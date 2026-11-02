// See decide_core.h.
#include "decide_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

struct dec_ctx {
    struct llama_model * model;
    struct llama_context * lctx;
    const struct llama_vocab * vocab;
    mtmd_context * mctx;
    int n_batch;
    double last_ms;
};

static int dec_verbose(void) { const char * v = getenv("DEC_VERBOSE"); return v && v[0] && v[0] != '0'; }

static void quiet_log(enum ggml_log_level level, const char * text, void * user) {
    (void) user;
    if (level >= GGML_LOG_LEVEL_ERROR || dec_verbose()) fputs(text, stderr);
}

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

dec_ctx * dec_load(const char * model_path, const char * mmproj_path, int n_threads, int n_gpu_layers, int n_ctx,
                   int image_max_tokens, int use_gpu_mm) {
    llama_log_set(quiet_log, NULL);
    mtmd_helper_log_set(quiet_log, NULL);
    ggml_backend_load_all();
    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    // DEC_NO_REPACK=1: keep CPU weights in the mmap'd file instead of repacked anonymous copies (CPU_REPACK buffers roughly double
    // the resident weight memory: 9.2 GB vs 6.3 GB max RSS for Q4_K_M + mmproj q8_0 on x86). Same speed on x86 in our runs
    // (ARM unmeasured).
    const char * nr = getenv("DEC_NO_REPACK");
    if (nr && nr[0] == '1') mp.use_extra_bufts = false;
    struct llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return NULL;

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx > 0 ? n_ctx : 2048;
    cp.n_batch = cp.n_ctx;            // the whole prompt in one logical batch
    cp.n_ubatch = 512;
    cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    cp.no_perf = 1;
    cp.op_offload = n_gpu_layers != 0; // CPU mode must not push big batched matmuls to a GPU behind our back
    struct llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) { llama_model_free(model); return NULL; }

    dec_ctx * d = calloc(1, sizeof(*d));
    d->model = model; d->lctx = lctx; d->vocab = llama_model_get_vocab(model); d->n_batch = cp.n_batch;
    if (mmproj_path && mmproj_path[0]) {
        struct mtmd_context_params mp2 = mtmd_context_params_default();
        mp2.use_gpu = use_gpu_mm != 0;
        mp2.n_threads = n_threads;
        mp2.print_timings = dec_verbose();
        mp2.warmup = 0;
        if (image_max_tokens > 0) mp2.image_max_tokens = image_max_tokens;
        d->mctx = mtmd_init_from_file(mmproj_path, model, mp2);
        if (!d->mctx) { dec_free(d); return NULL; }
    }
    return d;
}

void dec_free(dec_ctx * d) {
    if (!d) return;
    if (d->mctx) mtmd_free(d->mctx);
    if (d->lctx) llama_free(d->lctx);
    if (d->model) llama_model_free(d->model);
    free(d);
}

int32_t dec_token_id(dec_ctx * d, const char * text) {
    llama_token toks[8];
    int n = llama_tokenize(d->vocab, text, (int32_t) strlen(text), toks, 8, /*add_special*/ 0, /*parse_special*/ 0);
    return n == 1 ? toks[0] : -1;
}

float dec_softcap(dec_ctx * d) {
    char buf[64];
    char key[128];
    char arch[64] = {0};
    if (llama_model_meta_val_str(d->model, "general.architecture", arch, sizeof(arch)) < 0) return 0.0f;
    snprintf(key, sizeof(key), "%s.final_logit_softcapping", arch);
    if (llama_model_meta_val_str(d->model, key, buf, sizeof(buf)) < 0) return 0.0f;
    return (float) atof(buf);
}

int dec_has_vision(dec_ctx * d) { return d->mctx && mtmd_support_vision(d->mctx); }
int dec_has_audio(dec_ctx * d) { return d->mctx && mtmd_support_audio(d->mctx); }
double dec_last_ms(dec_ctx * d) { return d->last_ms; }

// copy of prompt with the first `tag` replaced by `marker`
static char * replace_once(const char * s, const char * tag, const char * marker) {
    const char * p = strstr(s, tag);
    if (!p) return NULL;
    size_t a = (size_t) (p - s), lt = strlen(tag), lm = strlen(marker), ls = strlen(s);
    char * out = malloc(ls - lt + lm + 1);
    memcpy(out, s, a); memcpy(out + a, marker, lm); memcpy(out + a + lm, p + lt, ls - a - lt + 1);
    return out;
}

int32_t dec_eval(dec_ctx * d, const char * prompt, int media, const void * data, int32_t nx, int32_t ny,
                 const int32_t * ids, int32_t n_ids, float * out) {
    double t0 = now_ms();
    llama_memory_clear(llama_get_memory(d->lctx), true);
    int32_t n_pos = 0;

    if (media == DEC_TEXT && !d->mctx) {
        int32_t cap = (int32_t) strlen(prompt) + 8;
        llama_token * toks = malloc(sizeof(llama_token) * cap);
        int32_t n = llama_tokenize(d->vocab, prompt, (int32_t) strlen(prompt), toks, cap, 0, 1);
        if (n <= 0 || n > d->n_batch) { free(toks); return -2; }
        int rc = llama_decode(d->lctx, llama_batch_get_one(toks, n));   // only the last token's logits are output
        free(toks);
        if (rc != 0) return -3;
        n_pos = n;
    } else {
        if (!d->mctx) return -4;
        char * text = NULL;
        mtmd_bitmap * bmp = NULL;
        if (media == DEC_IMAGE_RGB) {
            text = replace_once(prompt, "<|image|>", mtmd_default_marker());
            bmp = mtmd_bitmap_init((uint32_t) nx, (uint32_t) ny, (const unsigned char *) data);
        } else if (media == DEC_AUDIO_PCM16K) {
            text = replace_once(prompt, "<|audio|>", mtmd_default_marker());
            bmp = mtmd_bitmap_init_from_audio((size_t) nx, (const float *) data);
        } else {
            text = strdup(prompt);
        }
        if (!text) { if (bmp) mtmd_bitmap_free(bmp); return -5; }
        struct mtmd_input_text in = { text, strlen(text), /*add_special*/ false, /*parse_special*/ true };
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        const mtmd_bitmap * bmps[1] = { bmp };
        int32_t rc = mtmd_tokenize(d->mctx, chunks, &in, bmp ? bmps : NULL, bmp ? 1 : 0);
        free(text);
        if (bmp) mtmd_bitmap_free(bmp);
        if (rc != 0) { mtmd_input_chunks_free(chunks); return -6; }
        llama_pos new_n_past = 0;
        rc = mtmd_helper_eval_chunks(d->mctx, d->lctx, chunks, 0, 0, d->n_batch, true, &new_n_past);
        mtmd_input_chunks_free(chunks);
        if (rc != 0) return -7;
        n_pos = new_n_past;
    }
    const float * logits = llama_get_logits_ith(d->lctx, -1);
    if (!logits) return -8;
    for (int i = 0; i < n_ids; i++) out[i] = logits[ids[i]];
    d->last_ms = now_ms() - t0;
    return n_pos;
}

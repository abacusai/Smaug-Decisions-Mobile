// decide_core: a tiny C API over llama.cpp + libmtmd for decision models (one prefill, read the logits of a few tokens at the
// last position). Shared by llama_decide.py (ctypes) and the Android app's llama.cpp backend (JNI), so both run the exact
// same code path.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dec_ctx dec_ctx;

enum dec_media { DEC_TEXT = 0, DEC_IMAGE_RGB = 1, DEC_AUDIO_PCM16K = 2 };

// mmproj_path may be NULL or "" (text only). n_gpu_layers: 0 = CPU, -1 = all. image_max_tokens: 280 matches the HF processor.
// use_gpu_mm: run the vision/audio encoders on the GPU backend if one is compiled in. Returns NULL on failure.
dec_ctx * dec_load(const char * model_path, const char * mmproj_path, int n_threads, int n_gpu_layers, int n_ctx,
                   int image_max_tokens, int use_gpu_mm);
void dec_free(dec_ctx * d);

// id of `text` if it tokenizes (without specials) to exactly one token, else -1
int32_t dec_token_id(dec_ctx * d, const char * text);
// final_logit_softcapping stored in the GGUF (0 if none). llama.cpp's gemma4 graph already applies it to the logits it returns.
float dec_softcap(dec_ctx * d);
int dec_has_vision(dec_ctx * d);
int dec_has_audio(dec_ctx * d);

// prompt: the rendered prompt verbatim (starts with "<bos>"; contains "<|image|>" or "<|audio|>" exactly once when media
// is given). media: DEC_TEXT (data NULL), DEC_IMAGE_RGB (data = nx*ny*3 bytes RGB), DEC_AUDIO_PCM16K (data = nx float32 samples,
// mono 16 kHz, ny ignored). Writes the last-position logits of ids[0..n_ids) into out. Returns the number of prompt positions (>0)
// or a negative error code.
int32_t dec_eval(dec_ctx * d, const char * prompt, int media, const void * data, int32_t nx, int32_t ny,
                 const int32_t * ids, int32_t n_ids, float * out);

// milliseconds spent in the last dec_eval (encode + prefill)
double dec_last_ms(dec_ctx * d);

#ifdef __cplusplus
}
#endif

// A thin C layer between Swift and the LiteRT-LM C API. It compiles against both
//  - the 1.0 C API of a source build at 2c1c9a6 (with or without our harness patch), and
//  - the pre-1.0 C API of the stock prebuilt CLiteRTLM.xcframework v0.18.1 (the fallback),
// so the Swift code never sees which one is linked. Every int-returning call returns 0 on success; on failure it returns
// nonzero and decide_last_error() says why. Not thread-safe: call from one serial queue.
#ifndef DECIDE_SHIM_H_
#define DECIDE_SHIM_H_

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 1 when the linked headers carry the harness patch (multimodal prefill + raw letter logits), else 0.
int decide_has_harness(void);
// 1 when built against the 1.0 C API (source build), 0 for the pre-1.0 prebuilt.
int decide_c_api_v1(void);

// Engine on the CPU backend; vision / audio = also create that CPU encoder (each costs memory only when attached). NULL on failure.
// backend: "cpu" (default when NULL) or "gpu" for the decoder; the encoders stay on the CPU.
void* decide_engine_create(const char* model_path, const char* backend, const char* cache_dir, int max_tokens, int threads, bool vision,
                           bool audio, bool benchmark);
void decide_engine_delete(void* engine);
// The token id when `text` is exactly one token, else -1.
int decide_single_token(void* engine, const char* text);
// The number of tokens of `text` (no BOS added), or -1.
int decide_count_tokens(void* engine, const char* text);
// Token ids of `text` (no BOS added): copies up to `cap` ids to `out` and returns the full count, or -1.
int decide_tokenize(void* engine, const char* text, int* out, size_t cap);

// A raw session with apply_prompt_template=false (the runtime still prepends BOS). NULL on failure.
void* decide_session_create(void* engine);
void decide_session_delete(void* session);
// Raw text prefill (stock API; text only).
int decide_prefill_text(void* session, const char* prompt);
// Harness only: prefill with <|image|>/<|audio|> placeholders expanded from media_json
// ('[{"type":"audio","path":"/x.wav"}]', or "" for text). visual_token_budget <= 0 = model default.
int decide_prefill_multimodal(void* session, void* engine, const char* prompt, const char* media_json,
                              int visual_token_budget);
// Prefill token count of the last prefill turn (engine created with benchmark=true, 1.0 API), else -1.
// Call it right after the prefill, before scoring.
int decide_last_prefill_tokens(void* session);
// log p(target | everything prefilled so far). Score a session at most once (LiteRT-LM #3561).
int decide_score_text(void* session, const char* target, float* out_logp);
// Harness only: raw logits of `ids` at the most recent scoring step (process-global), plus the full-vocab logsumexp.
int decide_last_logits(const int* ids, size_t n, float* out_z, float* out_logsumexp);

const char* decide_last_error(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DECIDE_SHIM_H_

#include "DecideShim.h"

#include <stdio.h>
#include <string.h>

#include <CLiteRTLM/engine.h>
#include <CLiteRTLM/error_reporter.h>
#include <CLiteRTLM/experimental.h>

#if defined(LITERT_LM_C_API_VERSION_MAJOR) && LITERT_LM_C_API_VERSION_MAJOR >= 1
#define DECIDE_V1 1
#else
#define DECIDE_V1 0
#endif

#if defined(LITERT_LM_HARNESS_PATCH) && !DECIDE_V1
#error "the harness patch is only expected on top of the 1.0 C API"
#endif

static _Thread_local char g_error[1024];

static int fail(const char* what) {
  const char* m = litert_lm_get_last_error_message();
  snprintf(g_error, sizeof g_error, "%s failed: %s", what, (m && *m) ? m : "(no error message)");
  return -1;
}

static int fail_msg(const char* what, const char* msg) {
  snprintf(g_error, sizeof g_error, "%s: %s", what, msg);
  return -1;
}

const char* decide_last_error(void) { return g_error; }

#if DECIDE_V1
#define OK(call, what) \
  do { if ((call) != kLiteRtLmStatusOk) return fail(what); } while (0)
#endif

int decide_has_harness(void) {
#ifdef LITERT_LM_HARNESS_PATCH
  return 1;
#else
  return 0;
#endif
}

int decide_c_api_v1(void) { return DECIDE_V1; }

void* decide_engine_create(const char* model_path, const char* backend, const char* cache_dir, int max_tokens, int threads, bool vision,
                           bool audio, bool benchmark) {
  const char* main_backend = backend ? backend : "cpu";
  const char* vision_backend = vision ? "cpu" : NULL;
  const char* audio_backend = audio ? "cpu" : NULL;
  LiteRtLmEngine* engine = NULL;
#if DECIDE_V1
  LiteRtLmEngineSettings* s = NULL;
  if (litert_lm_engine_settings_create(model_path, main_backend, vision_backend, audio_backend, &s) != kLiteRtLmStatusOk) {
    fail("litert_lm_engine_settings_create");
    return NULL;
  }
  int bad = litert_lm_engine_settings_set_max_num_tokens(s, max_tokens) != kLiteRtLmStatusOk ||
            litert_lm_engine_settings_set_num_threads(s, threads) != kLiteRtLmStatusOk ||
            (cache_dir && litert_lm_engine_settings_set_cache_dir(s, cache_dir) != kLiteRtLmStatusOk) ||
            (benchmark && litert_lm_engine_settings_enable_benchmark(s) != kLiteRtLmStatusOk);
  if (bad) {
    fail("engine settings");
  } else if (litert_lm_engine_create(s, &engine) != kLiteRtLmStatusOk) {
    fail("litert_lm_engine_create");
    engine = NULL;
  }
  litert_lm_engine_settings_delete(s);
#else
  (void)benchmark;
  LiteRtLmEngineSettings* s = litert_lm_engine_settings_create(model_path, main_backend, vision_backend, audio_backend);
  if (!s) {
    fail("litert_lm_engine_settings_create");
    return NULL;
  }
  litert_lm_engine_settings_set_max_num_tokens(s, max_tokens);
  litert_lm_engine_settings_set_num_threads(s, threads);
  if (cache_dir) litert_lm_engine_settings_set_cache_dir(s, cache_dir);
  engine = litert_lm_engine_create(s);
  if (!engine) fail("litert_lm_engine_create");
  litert_lm_engine_settings_delete(s);
#endif
  return engine;
}

void decide_engine_delete(void* engine) {
  if (engine) litert_lm_engine_delete((LiteRtLmEngine*)engine);
}

// Tokenizes `text`; on success *out_n tokens are copied to out (up to cap) and 0 is returned.
static int tokenize(void* engine, const char* text, int* out, size_t cap, size_t* out_n) {
  size_t n = 0;
  const int* ids = NULL;
#if DECIDE_V1
  LiteRtLmTokenizeResult* r = NULL;
  OK(litert_lm_engine_tokenize((LiteRtLmEngine*)engine, text, &r), "litert_lm_engine_tokenize");
  if (litert_lm_tokenize_result_get_num_tokens(r, &n) != kLiteRtLmStatusOk ||
      litert_lm_tokenize_result_get_tokens(r, &ids) != kLiteRtLmStatusOk) {
    litert_lm_tokenize_result_delete(r);
    return fail("litert_lm_tokenize_result_get_*");
  }
#else
  LiteRtLmTokenizeResult* r = litert_lm_engine_tokenize((LiteRtLmEngine*)engine, text);
  if (!r) return fail("litert_lm_engine_tokenize");
  n = litert_lm_tokenize_result_get_num_tokens(r);
  ids = litert_lm_tokenize_result_get_tokens(r);
#endif
  for (size_t i = 0; i < n && i < cap; ++i) out[i] = ids[i];
  *out_n = n;
  litert_lm_tokenize_result_delete(r);
  return 0;
}

int decide_single_token(void* engine, const char* text) {
  int id = -1;
  size_t n = 0;
  if (tokenize(engine, text, &id, 1, &n) != 0) return -1;
  if (n != 1) {
    fail_msg("decide_single_token", "text is not exactly one token");
    return -1;
  }
  return id;
}

int decide_count_tokens(void* engine, const char* text) {
  size_t n = 0;
  if (tokenize(engine, text, NULL, 0, &n) != 0) return -1;
  return (int)n;
}

int decide_tokenize(void* engine, const char* text, int* out, size_t cap) {
  size_t n = 0;
  if (tokenize(engine, text, out, cap, &n) != 0) return -1;
  return (int)n;
}

void* decide_session_create(void* engine) {
  LiteRtLmSession* session = NULL;
#if DECIDE_V1
  LiteRtLmSessionConfig* cfg = NULL;
  if (litert_lm_session_config_create(&cfg) != kLiteRtLmStatusOk) {
    fail("litert_lm_session_config_create");
    return NULL;
  }
  if (litert_lm_session_config_set_apply_prompt_template(cfg, false) != kLiteRtLmStatusOk) {
    fail("litert_lm_session_config_set_apply_prompt_template");
  } else if (litert_lm_engine_create_session((LiteRtLmEngine*)engine, cfg, &session) != kLiteRtLmStatusOk) {
    fail("litert_lm_engine_create_session");
    session = NULL;
  }
  litert_lm_session_config_delete(cfg);
#else
  LiteRtLmSessionConfig* cfg = litert_lm_session_config_create();
  if (!cfg) {
    fail("litert_lm_session_config_create");
    return NULL;
  }
  litert_lm_session_config_set_apply_prompt_template(cfg, false);
  session = litert_lm_engine_create_session((LiteRtLmEngine*)engine, cfg);
  if (!session) fail("litert_lm_engine_create_session");
  litert_lm_session_config_delete(cfg);
#endif
  return session;
}

void decide_session_delete(void* session) {
  if (session) litert_lm_session_delete((LiteRtLmSession*)session);
}

int decide_prefill_text(void* session, const char* prompt) {
  LiteRtLmInputData* in = NULL;
#if DECIDE_V1
  OK(litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt, strlen(prompt), &in), "litert_lm_input_data_create");
  LiteRtLmStatusCode st = litert_lm_session_run_prefill((LiteRtLmSession*)session, (const LiteRtLmInputData* const*)&in, 1);
  litert_lm_input_data_delete(in);
  return st == kLiteRtLmStatusOk ? 0 : fail("litert_lm_session_run_prefill");
#else
  in = litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt, strlen(prompt));
  if (!in) return fail("litert_lm_input_data_create");
  int st = litert_lm_session_run_prefill((LiteRtLmSession*)session, (const LiteRtLmInputData* const*)&in, 1);
  litert_lm_input_data_delete(in);
  return st == 0 ? 0 : fail("litert_lm_session_run_prefill");
#endif
}

int decide_prefill_multimodal(void* session, void* engine, const char* prompt, const char* media_json,
                              int visual_token_budget) {
#ifdef LITERT_LM_HARNESS_PATCH
  OK(litert_lm_experimental_session_run_prefill_multimodal((LiteRtLmSession*)session, (LiteRtLmEngine*)engine, prompt,
                                                           media_json, visual_token_budget),
     "litert_lm_experimental_session_run_prefill_multimodal");
  return 0;
#else
  (void)session; (void)engine; (void)prompt; (void)media_json; (void)visual_token_budget;
  return fail_msg("decide_prefill_multimodal", "this CLiteRTLM build has no harness patch (images/audio need it)");
#endif
}

int decide_last_prefill_tokens(void* session) {
#if DECIDE_V1
  LiteRtLmBenchmarkInfo* info = NULL;
  if (litert_lm_session_get_benchmark_info((LiteRtLmSession*)session, &info) != kLiteRtLmStatusOk || !info) return -1;
  size_t turns = 0, count = 0;
  int out = -1;
  if (litert_lm_benchmark_info_get_num_prefill_turns(info, &turns) == kLiteRtLmStatusOk && turns > 0 &&
      litert_lm_benchmark_info_get_prefill_token_count_at(info, turns - 1, &count) == kLiteRtLmStatusOk) {
    out = (int)count;
  }
  litert_lm_benchmark_info_delete(info);
  return out;
#else
  (void)session;
  return -1;
#endif
}

int decide_score_text(void* session, const char* target, float* out_logp) {
  const char* targets[1] = {target};
#if DECIDE_V1
  LiteRtLmResponses* r = NULL;
  OK(litert_lm_session_run_text_scoring((LiteRtLmSession*)session, targets, 1, false, &r),
     "litert_lm_session_run_text_scoring");
  LiteRtLmStatusCode st = litert_lm_responses_get_score_at(r, 0, out_logp);
  litert_lm_responses_delete(r);
  return st == kLiteRtLmStatusOk ? 0 : fail("litert_lm_responses_get_score_at");
#else
  LiteRtLmResponses* r = litert_lm_session_run_text_scoring((LiteRtLmSession*)session, targets, 1, false);
  if (!r) return fail("litert_lm_session_run_text_scoring");
  *out_logp = litert_lm_responses_get_score_at(r, 0);
  litert_lm_responses_delete(r);
  return 0;
#endif
}

int decide_last_logits(const int* ids, size_t n, float* out_z, float* out_logsumexp) {
#ifdef LITERT_LM_HARNESS_PATCH
  OK(litert_lm_experimental_get_last_scoring_logits(ids, n, out_z, out_logsumexp, NULL, NULL),
     "litert_lm_experimental_get_last_scoring_logits");
  return 0;
#else
  (void)ids; (void)n; (void)out_z; (void)out_logsumexp;
  return fail_msg("decide_last_logits", "this CLiteRTLM build has no harness patch");
#endif
}

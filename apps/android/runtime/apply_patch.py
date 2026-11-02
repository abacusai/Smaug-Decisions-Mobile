"""Applies the LiteRT-LM harness patch (multimodal raw-session prefill + raw letter logits) to a LiteRT-LM checkout at 2c1c9a67
(== litert-lm 0.19.0). Usage: python apply_patch.py <repo>"""
import pathlib, sys
R = pathlib.Path(sys.argv[1])
def edit(rel, old, new, count=1):
    p = R / rel; s = p.read_text()
    assert s.count(old) >= 1, (rel, old[:60])
    if new in s: return
    p.write_text(s.replace(old, new, count))

# 1. Capture the raw logits row used by text scoring.
edit("runtime/components/scoring_cpu_util.h",
"""    float temperature);
}  // namespace litert::lm""",
"""    float temperature);

// [harness patch] Raw logits (batch row 0, full vocab) seen by the most recent
// ComputeLogLikelihood call, i.e. the next-token logits at the scoring step.
std::vector<float> HarnessGetLastScoringLogits();
}  // namespace litert::lm""")
edit("runtime/components/scoring_cpu_util.cc", "#include <iterator>\n", "#include <iterator>\n#include <mutex>\n")
edit("runtime/components/scoring_cpu_util.cc",
"""namespace litert::lm {

absl::StatusOr<std::vector<float>> ComputeLogLikelihood(""",
"""namespace litert::lm {

namespace {
std::mutex& HarnessMutex() { static std::mutex* m = new std::mutex(); return *m; }
std::vector<float>& HarnessLogits() { static auto* v = new std::vector<float>(); return *v; }
}  // namespace

std::vector<float> HarnessGetLastScoringLogits() {
  std::lock_guard<std::mutex> lock(HarnessMutex());
  return HarnessLogits();
}

absl::StatusOr<std::vector<float>> ComputeLogLikelihood(""")
edit("runtime/components/scoring_cpu_util.cc",
"""  const int vocab_size = logits.size() / batch_size;
""",
"""  const int vocab_size = logits.size() / batch_size;
  {
    std::lock_guard<std::mutex> lock(HarnessMutex());
    HarnessLogits().assign(logits.begin(), logits.begin() + vocab_size);
  }
""")

# 2. C API additions.
edit("c/conversation.cc", "#include <cstring>\n", "#include <cmath>\n#include <cstring>\n#include <variant>\n#include <vector>\n")
edit("c/conversation.cc", "#include \"runtime/conversation/model_data_processor/gemma4_data_processor_config.h\"\n",
     "#include \"runtime/conversation/model_data_processor/gemma4_data_processor_config.h\"\n"
     "#include \"runtime/conversation/model_data_processor/model_data_processor_factory.h\"\n"
     "#include \"runtime/components/scoring_cpu_util.h\"\n")
s = (R / "c/conversation.cc").read_text()
if "litert_lm_experimental_session_run_prefill_multimodal" not in s:
    idx = s.rfind("}  // extern \"C\"")
    assert idx > 0
    add = r"""
// [harness patch] Prefills `rendered_prompt` (no chat template is applied; the
// session adds BOS on the first turn) into a raw session. <|image|> / <|audio|>
// placeholders are replaced by the model data processor exactly as the
// Conversation API does, using `media_json`, a JSON array of
// {"type": "image"|"audio", "path": "..."} items in prompt order.
// visual_token_budget <= 0 keeps the model default.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_session_run_prefill_multimodal(
    LiteRtLmSession* session, LiteRtLmEngine* engine,
    const char* rendered_prompt, const char* media_json,
    int visual_token_budget) {
  if (!session || !session->session || !engine || !engine->engine ||
      !rendered_prompt) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session, engine or prompt.");
  }
  const auto& model_type =
      session->session->GetSessionConfig().GetLlmModelType();
  auto config = litert::lm::CreateDataProcessorConfigFromLlmModelType(model_type);
  if (!config.ok()) return litert::lm::c::ToCStatus(config.status());
  auto processor = litert::lm::CreateModelDataProcessor(
      *config, std::nullopt, &engine->engine->GetTokenizer());
  if (!processor.ok()) return litert::lm::c::ToCStatus(processor.status());
  nlohmann::ordered_json content = nlohmann::ordered_json::array();
  if (media_json && *media_json) {
    content = nlohmann::ordered_json::parse(media_json, nullptr, false);
    if (content.is_discarded() || !content.is_array()) {
      return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                        "media_json must be a JSON array.");
    }
  }
  nlohmann::ordered_json messages = nlohmann::ordered_json::array();
  messages.push_back({{"role", "user"}, {"content", content}});
  litert::lm::DataProcessorArguments args = std::monostate();
  if (visual_token_budget > 0 && model_type.has_gemma4()) {
    litert::lm::Gemma4DataProcessorArguments g4;
    g4.visual_token_budget = visual_token_budget;
    args = g4;
  }
  auto inputs =
      (*processor)->ToInputDataVector(rendered_prompt, messages, args);
  if (!inputs.ok()) return litert::lm::c::ToCStatus(inputs.status());
  auto status = session->session->RunPrefill(*inputs);
  if (!status.ok()) return litert::lm::c::ToCStatus(status);
  return kLiteRtLmStatusOk;
}

// [harness patch] Raw next-token logits from the most recent text-scoring
// step (process-global; call right after litert_lm_session_run_text_scoring).
// Writes logits[ids[i]] to out_logits[i]; optionally the full-vocab logsumexp,
// the max logit and the vocab size.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_get_last_scoring_logits(
    const int* ids, size_t num_ids, float* out_logits, float* out_logsumexp,
    float* out_max_logit, int* out_vocab_size) {
  std::vector<float> row = litert::lm::HarnessGetLastScoringLogits();
  if (row.empty()) {
    return litert::lm::c::ReturnError(absl::StatusCode::kFailedPrecondition,
                                      "No scoring logits captured yet.");
  }
  for (size_t i = 0; i < num_ids; ++i) {
    if (ids[i] < 0 || ids[i] >= static_cast<int>(row.size())) {
      return litert::lm::c::ReturnError(absl::StatusCode::kOutOfRange,
                                        "Token id out of range.");
    }
    if (out_logits) out_logits[i] = row[ids[i]];
  }
  double m = row[0];
  for (float v : row) m = v > m ? v : m;
  if (out_logsumexp) {
    double s = 0.0;
    for (float v : row) s += std::exp(static_cast<double>(v) - m);
    *out_logsumexp = static_cast<float>(m + std::log(s));
  }
  if (out_max_logit) *out_max_logit = static_cast<float>(m);
  if (out_vocab_size) *out_vocab_size = static_cast<int>(row.size());
  return kLiteRtLmStatusOk;
}

"""
    s = s[:idx] + add + s[idx:]
    (R / "c/conversation.cc").write_text(s)

# 3. Deps for the shared library target.
edit("c/BUILD", """        ":model_info_headers",
        "//runtime/engine:engine_impl_selected",
    ] + select({""", """        ":model_info_headers",
        "//runtime/engine:engine_impl_selected",
        "//runtime/components:scoring_cpu_util",
        "//runtime/conversation/model_data_processor:model_data_processor",
        "//runtime/conversation/model_data_processor:model_data_processor_factory",
    ] + select({""")
print("patched")

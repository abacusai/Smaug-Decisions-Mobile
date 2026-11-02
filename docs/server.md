# Server

A small HTTP server ([`server/decisions_server.py`](../server/decisions_server.py)) that puts a decision model, or the cascade,
behind one endpoint, `POST /v1/decisions`: one input (text, one audio clip or
one image), up to 64 typed questions, a calibrated answer per question. Nothing is generated, so every answer is a probability
distribution over the options you sent.

```bash
pip install -e ".[server]"                                   # from the repository root
python server/decisions_server.py --model abacusai/Smaug-Decisions --port 8000
```

`--model` takes a Hugging Face repo id or a local checkpoint directory; `abacusai/Smaug-Audio-Decisions-0.7B` works the same way. The server
binds to `127.0.0.1` by default; pass `--host 0.0.0.0` to expose it, and set an API key when you do (see below).

**Cascade.** Add `--fast` to serve the cascade: audio questions go to the fast tier first and escalate to `--model` when its
confidence is below `--threshold`; text and image questions go straight to `--model`. Both models are loaded at start-up.

```bash
python server/decisions_server.py --model abacusai/Smaug-Decisions --fast abacusai/Smaug-Audio-Decisions-0.7B --threshold 0.9 --port 8000
```

Every answer then carries `"tier": "fast"` or `"tier": "big"`. On an H100 this answers Earshot at Smaug-Decisions' level (60.1) in
130 ms per audio question on average, against 210 ms for Smaug-Decisions alone. Set the threshold on your own traffic with
[`scripts/calibrate.py`](choosing-a-threshold.md).

| Endpoint | Purpose |
|---|---|
| `POST /v1/decisions` | Evaluate an input against typed questions |
| `GET /health` | Liveness and the model name (never needs a key) |
| `GET /v1/models` | The served model |
| `GET /demo` | Phone-first demo page: turn gate, room awareness, image checks, text routing |
| `GET /playground` | Live turn-taking from the microphone and free-form questions about a clip |

## Request

```bash
curl -s localhost:8000/v1/decisions -H 'content-type: application/json' -d '{
  "input": [
    {"role": "developer", "content": "You triage a support line."},
    {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "<base64 wav>", "format": "wav"}}]}
  ],
  "questions": [
    {"name": "done", "type": "predicate", "instructions": "Has the speaker finished their turn?"},
    {"name": "mood", "type": "score", "instructions": "How angry is the speaker?",
     "levels": [{"label": "calm"}, {"label": "annoyed"}, {"label": "angry"}]},
    {"name": "team", "type": "choice", "instructions": "Which team should handle this?",
     "choices": [{"value": "billing", "description": "refunds and payments"}, {"value": "shipping"}]},
    {"name": "sounds", "type": "multi", "instructions": "Which of these can be heard?",
     "choices": [{"value": "speech"}, {"value": "music"}, {"value": "traffic"}]}
  ]
}'
```

**Input.** A plain string, or a list of messages whose `content` is a string or a list of parts:

| Part | Shape | Notes |
|---|---|---|
| text | `{"type": "input_text", "text": "..."}` | `developer` / `system` text becomes the rules; `user` text becomes the context when there is media, otherwise the text input |
| audio | `{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}` | wav, flac or ogg; resampled to 16 kHz mono; the last 30 s are kept |
| image | `{"type": "input_image", "image_url": "data:image/png;base64,..."}` | data URLs only; remote URLs are never fetched |

One audio clip or one image per request, not both.

**Questions.** Every question has a unique `name`, a `type` and `instructions`:

| Type | Options | Answer |
|---|---|---|
| `predicate` | Yes / No | `probability` that the statement is true |
| `choice` | `choices: [{value, description?}]` | `choice`, `probabilities`, `confidence` |
| `score` | `levels: [{label, description?}]`, lowest first | `score` (expected level index), `probabilities`, `confidence` |
| `multi` | `choices: [{value, description?}]` | `selected` (probability above 0.5) and an independent `probabilities` entry per option |

Confidence for `choice` is `(k * max_p - 1) / (k - 1)`: 0 at a uniform answer, 1 when certain. For `score` it is
`1 - sum_i p_i |i - peak| / mean_i |i - (k - 1) / 2|`, which also penalises probability mass far from the peak level. A question
with more than 20 options gets a `refusal` answer; the other questions in the request are still answered.

## Response

```json
{
  "model": "abacusai/Smaug-Decisions",
  "answers": [
    {"type": "predicate", "name": "done", "probability": 0.91},
    {"type": "score", "name": "mood", "score": 0.4, "confidence": 0.71, "probabilities": [{"value": 0, "label": "calm", "probability": 0.68}, "..."]},
    {"type": "choice", "name": "team", "choice": "billing", "confidence": 0.62, "probabilities": [{"value": "billing", "probability": 0.81}, "..."]},
    {"type": "multi", "name": "sounds", "selected": ["speech"], "probabilities": [{"value": "speech", "probability": 0.97}, "..."]}
  ],
  "latency_ms": 812.4
}
```

Errors use `{"error": {"message", "type"}}` with status 400 (`invalid_request_error`) or 401 (`authentication_error`).

## Operating notes

- **Authentication.** `DECISIONS_API_KEY=<key>` (or `DECISIONS_API_KEY_FILE=<path>` for a mounted secret) makes `/v1/decisions`
  require `Authorization: Bearer <key>`. The demo pages read the key from a `#key=...` URL fragment or an input field.
- **Latency.** Each question is one forward pass (two when the cascade escalates), run one after another. On an H100, Smaug-Decisions
  takes about 210 ms per audio question and Smaug-Audio-Decisions-0.7B about 10 ms.
- **Concurrency.** One model per process; requests are serialised with a lock. Run several processes behind a load balancer to
  scale out.
- **Privacy.** The server keeps nothing: audio and images live only for the duration of the request.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/test_server.py
```

The tests replace the model with a fake that returns fixed probabilities, so they need neither a GPU nor the weights. They cover
request parsing, every question type, refusals, malformed requests, authentication and the cascade's tier field.

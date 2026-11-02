# Python

The `smaug_decide` package runs both models and the cascade with PyTorch and Transformers, on NVIDIA GPUs (bfloat16), Apple
Silicon (MPS) and CPUs. The device is picked automatically.

```bash
git clone https://github.com/abacusai/Smaug-Decisions-Mobile && cd Smaug-Decisions-Mobile
pip install -e .                       # Python 3.10+: torch, torchvision, transformers>=5.20, numpy, pillow, soundfile, huggingface_hub
```

`from_pretrained` downloads the weights on first use: Smaug-Decisions is about 17 GB, Smaug-Audio-Decisions-0.7B about 3 GB. To fetch them ahead of
time, use `python scripts/download.py --model decisions` (or `audio`) and pass the local directory instead of the repo id.

## One model

```python
from smaug_decide import SmaugDecide

model = SmaugDecide.from_pretrained("abacusai/Smaug-Decisions")            # or a local directory; device="cuda" | "mps" | "cpu"

model("call.wav", "Has the speaker finished their turn?")                                  # {'Yes': 0.93, 'No': 0.07}
model("song.wav", "Which genre is playing?", ["rock", "jazz", "classical"])                 # one answer: sums to 1
model("room.wav", "Which sounds are present?", ["speech", "music", "alarm"], multi=True)    # select-all: one probability each
model("receipt.jpg", "Does the total exceed $50?")
model(
    "The order never arrived and nobody answers the phone.",
    "How upset is the customer?",
    ["calm", "annoyed", "angry"],
    descriptions=[None, "frustrated but polite", "hostile or threatening"],
    rules="Judge the tone, not the facts.",
    input_type="text",
)
```

`model(inp, question, options=('Yes', 'No'), descriptions=None, multi=False, rules=None, context=None, input_type=None,
long_audio='last')` returns `{option: probability}`.

| Argument | Meaning |
|---|---|
| `inp` | An audio path, a `(samples, sample_rate)` pair or a 16 kHz float array; an image path or PIL image; or a string of text |
| `options` | 2 to 20 options for Smaug-Decisions (lettered A..T in the prompt); 2 or more for Smaug-Audio-Decisions-0.7B |
| `descriptions` | One short description per option (or `None`), shown next to it; settles edge cases |
| `multi` | `True` for select-all questions: independent probabilities. `False`: one answer, probabilities sum to 1 |
| `rules` | Standing instructions for every input ("Escalate anything involving money") |
| `context` | Facts about this input ("A caller on a support line") |
| `input_type` | `'audio'`, `'image'` or `'text'`. Without it, a string naming an existing file is read as that file: **always pass it for input you did not write yourself** |
| `long_audio` | Clips over 30 s: `'last'` keeps the last 30 s (how the clip ends), `'mean'` averages 30 s windows (whole-clip questions), `'any'` takes each option's maximum over windows (select-all presence questions) |

`model.logits(...)` takes the same arguments and returns the raw option logits before softmax or sigmoid.

## The cascade

```python
from smaug_decide import Cascade

cascade = Cascade("abacusai/Smaug-Audio-Decisions-0.7B", "abacusai/Smaug-Decisions", threshold=0.9)
r = cascade("call.wav", "Has the speaker finished their turn?")
r.probabilities        # {'Yes': ..., 'No': ...}
r.tier                 # 'fast': Smaug-Audio-Decisions-0.7B answered; 'big': Smaug-Decisions answered
r.fast_confidence      # the audio model's confidence, or None when the input was routed straight to Smaug-Decisions
r.routed               # True for text and images, which skip the audio model
```

- Audio goes to Smaug-Audio-Decisions-0.7B first. Its answer stands when its confidence is at least `threshold`; otherwise the question is
  asked again of Smaug-Decisions.
- Text and images go straight to Smaug-Decisions (the audio model is near chance on text). Pass `fast_inputs=None` to send everything to the audio model
  first, or e.g. `fast_inputs=('audio', 'image')`.
- Confidence (`smaug_decide.confidence`) is the top probability for one-answer questions and the least certain option's
  `max(p, 1 - p)` for select-all questions. The phone apps port this rule exactly.
- The models load on first use, so a cascade that never escalates never loads Smaug-Decisions. Pass `SmaugDecide` instances instead of
  repo ids to load both up front.
- The threshold trades latency for accuracy. 0.9 matched Smaug-Decisions on Earshot while escalating 60% of the questions; measure it
  on your own examples with [`scripts/calibrate.py`](choosing-a-threshold.md).

## Command line

```bash
python scripts/decide.py call.wav "Has the speaker finished their turn?"                       # Smaug-Decisions
python scripts/decide.py call.wav "Has the speaker finished their turn?" --model cascade        # the cascade
python scripts/decide.py room.wav "Which sounds are present?" --options speech music alarm --multi --model audio
python scripts/decide.py "Refund me now." "Which team?" --options billing shipping --type text --dry-run   # prompt only
```

For batches, the package's own command line reads a JSONL file of cases and writes one result per line:

```bash
python -m smaug_decide --model abacusai/Smaug-Decisions --cases cases.jsonl --out results.jsonl
# cases.jsonl: {"id", "input", "question", "options", "descriptions"?, "multi"?, "rules"?, "context"?} per line
```

The output format is also the reference format the parity scripts in [`scripts/convert/`](../scripts/convert) and
[`llamacpp/`](../llamacpp) read.

## Writing good questions

- **Say what to judge.** `rules` holds standing instructions ("Judge the tone, not the facts"); `context` holds facts about this
  particular input.
- **Describe ambiguous options.** `descriptions` settle edge cases ("not music: there is no music in the clip").
- **Include a way out.** If none of the options may apply, add one that says so ("none of these").
- **Use `multi=True` only for select-all questions.** Its probabilities are independent; for one-answer questions they sum to 1.
- **Keep prompts short on phones.** The LiteRT-LM builds hold about 2,000 tokens; the package trims long rules, context and text,
  keeping the head and the tail.

## For runtime authors

Everything a new runtime needs to reproduce a decision is plain Python without torch:

| Module | What it defines |
|---|---|
| [`smaug_decide/prompt.py`](../smaug_decide/prompt.py) | The exact prompt for each readout (`render_letters`, `render_marker`, `build_prompt`) and the character budget |
| [`smaug_decide/readout.py`](../smaug_decide/readout.py) | Softcap, softmax and `sigmoid(z - b)`; window combination for long audio |
| [`smaug_decide/cascade.py`](../smaug_decide/cascade.py) | The confidence rule and the routing |
| [`smaug_decide/mel.py`](../smaug_decide/mel.py) | The audio front end (log-mel features) in numpy, matching the Transformers feature extractor |
| [`smaug_decide/calibration.py`](../smaug_decide/calibration.py) | Scoring a cascade threshold from saved answers |

The tests in [`tests/`](../tests) pin these down and run without weights: `pip install -e ".[dev]" && pytest tests`.

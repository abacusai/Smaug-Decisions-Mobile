# Examples

Short, runnable scripts for the Python API. Each takes your own file as input and downloads the model from Hugging Face on first
use (Smaug-Decisions is about 17 GB, Smaug-Audio-Decisions-0.7B about 3 GB).

```bash
pip install -e .            # from the repository root
python examples/audio_predicate.py call.wav
```

| Script | What it shows |
|---|---|
| [`audio_predicate.py`](audio_predicate.py) | Yes / No about a clip: has the speaker finished their turn? |
| [`audio_choice.py`](audio_choice.py) | One answer out of several, with option descriptions |
| [`audio_select_all.py`](audio_select_all.py) | Select all that apply: an independent probability per option |
| [`image.py`](image.py) | Questions about a photo or a screenshot |
| [`text_with_rules.py`](text_with_rules.py) | Text input with developer rules and option descriptions |
| [`cascade.py`](cascade.py) | Smaug-Audio-Decisions-0.7B first for audio, escalate to Smaug-Decisions when its confidence is below a threshold |
| [`turn_gate.py`](turn_gate.py) | A voice-agent turn gate replayed over a recording: decide at every pause |

## Writing good questions

- **Say what to judge.** `rules` holds standing instructions ("Judge the tone, not the facts"); `context` holds facts about this
  particular input ("A caller on a support line").
- **Describe ambiguous options.** `descriptions` are shown next to each option and settle edge cases ("not music: there is no
  music in the clip").
- **Include a way out.** If none of the options may apply, add one that says so ("none of these", "not music").
- **Use `multi=True` only for select-all questions.** Its probabilities are independent; for one-answer questions they sum to 1.
- **Thresholds are yours to set.** The probabilities are calibrated, so a threshold means what it says, but pick it on a few
  hundred of your own examples before acting on single decisions ([choosing a threshold](../docs/choosing-a-threshold.md)).

More: [docs/python.md](../docs/python.md).

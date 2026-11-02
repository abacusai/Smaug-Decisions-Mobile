# Choosing a threshold

The cascade has one knob, tau. Smaug-Audio-Decisions-0.7B's answer to an audio question stands when its confidence is at least tau;
below it, the question goes to Smaug-Decisions. A higher tau escalates more: closer to Smaug-Decisions' accuracy, slower on average.

Confidence is the top probability for a one-answer question and, for a select-all question, the least certain option's
`max(p, 1 - p)` (every option has to be clear for the audio model to answer). Text and image questions never use tau: they always go
to Smaug-Decisions.

## What it looks like on Earshot

Our audio benchmark, 6,671 items, on an H100 (mean latency = the audio model + escalation rate x Smaug-Decisions):

| tau | Escalated to Smaug-Decisions | Earshot | ECE | Mean latency |
|---|---|---|---|---|
| The audio model alone | 0% | 53.0 | 0.086 | 9.7 ms |
| 0.80 | 46% | 59.2 | 0.071 | 101 ms |
| 0.85 | 52% | 59.4 | 0.072 | 114 ms |
| **0.90** | **60%** | **60.1** | 0.071 | **130 ms** |
| 0.95 | 72% | 60.2 | 0.069 | 153 ms |
| Smaug-Decisions alone | 100% | 60.1 | 0.069 | 210 ms |

On a phone the trade is steeper, because Smaug-Decisions costs far more than the audio model there. On the iPhone 14 Pro Max, the audio model takes about
0.33 s and an escalated audio decision about 3-5 s more, so at 60% escalation the mean is roughly 2.5-3.5 s per audio question
(an estimate from the measured per-tier latencies in [devices.md](devices.md)).

Earshot is a mix of many question types. Your questions, audio and accuracy needs will put the best tau somewhere else.

## Calibrate on your own examples

[`scripts/calibrate.py`](../scripts/calibrate.py) runs both models once on your labeled examples and scores every tau from 0.50
to 0.99.

1. Collect a few hundred audio clips that look like your traffic, with the question you will ask and the right answer. Write
   one JSON object per line (media paths are relative to the file):

   ```json
   {"input": "clips/0001.wav", "question": "Has the speaker finished their turn?", "options": ["Yes", "No"], "label": "Yes"}
   {"input": "clips/0002.wav", "question": "Which sounds are present?", "options": ["speech", "music", "alarm"], "multi": true, "label": ["speech", "alarm"]}
   ```

   `descriptions`, `rules` and `context` are optional, as in the Python API. A select-all answer counts as correct when the
   options above 0.5 are exactly the labeled set.

2. Check the file, then run:

   ```bash
   python scripts/calibrate.py examples.jsonl --dry-run
   python scripts/calibrate.py examples.jsonl --out answers.jsonl --json sweep.json
   ```

   Both models' answers and latencies are saved to `answers.jsonl`; re-sweep later without the models with
   `python scripts/calibrate.py --answers answers.jsonl`.

3. Read the table (illustrative numbers):

   ```
          tau  accuracy  escalated   mean ms
   0.7B audio model only     0.812       0.0%      11.8
         0.50     0.812       0.0%      11.8
          ...
         0.88     0.861      55.1%     128.4  <- suggested
          ...
      Smaug-Decisions     0.866     100.0%     212.0

   Suggested tau = 0.88: accuracy 0.861 (Smaug-Decisions alone 0.866), 55% escalated, on 400 examples.
   ```

   The suggestion is the lowest-escalation tau whose accuracy is within `--tolerance` (default 0.005) of Smaug-Decisions alone. Raise
   the tolerance to trade accuracy for speed; set it to 0 to ask for Smaug-Decisions' accuracy.

The latency column is measured on the machine that ran the script. For a phone, replace `fast_ms` and `big_ms` in
`answers.jsonl` with that phone's per-tier latencies (see [devices.md](devices.md)) and sweep again with `--answers`.

## Tips

- **Use enough examples.** With 200 examples, one accuracy point is two examples; differences smaller than that are noise.
  The 95% interval of an accuracy near 0.85 on 400 examples is about +/- 3.5 points.
- **One tau per question type** when they behave differently: run calibrate on each question's examples and give each its own
  `Cascade(..., threshold=...)`. A turn-taking gate and a sound-tagging question rarely want the same value.
- **Select-all questions escalate more,** because every option has to be clear. If that is too slow, ask fewer options per
  question.
- **Recheck after your traffic changes:** new microphones, languages or noise conditions move the audio model's confidence.
- **The probabilities are calibrated,** so the threshold on the final answer (for example "act when P(finished) >= 0.8") is a
  separate decision from tau. Pick it the same way, on your own labeled examples.

"""SmaugDecide: a calibrated typed decision over audio, an image or text in one forward pass, a probability per option, no
generated text. The same class loads both published models and picks the readout from the checkpoint's readout.json:

- letters (abacusai/Smaug-Decisions): LM-head logits of the option letters A..T at the answer position, tanh softcap, then softmax for
  one-answer questions or sigmoid(z - b) for select-all questions (`final_logit_softcapping` and `multi_bias_b` in readout.json).
- marker (abacusai/Smaug-Audio-Decisions-0.7B): a linear head (head.pt) scores the hidden state at each option's marker token, then
  softmax over the options or sigmoid per option.

Runs on NVIDIA GPUs (bfloat16), Apple Silicon (MPS) and CPU; the device is picked automatically.
"""

import json
import os

import torch
from huggingface_hub import snapshot_download

from .media import audio_chunks, classify_input, load_audio
from .prompt import LETTERS, MARKER, build_prompt
from .readout import combine_windows, option_probs, readout_kind


# the Transformers checkpoint at the repository root; other formats in subfolders stay on the Hub
WEIGHT_FILES = ['*.json', '*.jinja', 'model*.safetensors', 'head.pt']


class SmaugDecide:
    def __init__(self, model, processor, readout, device, head=None):
        self.model, self.proc, self.device = model.eval(), processor, device
        self.readout = readout_kind(readout)
        tok = processor.tokenizer
        if self.readout == 'letters':
            ids = [tok(L, add_special_tokens=False)['input_ids'] for L in LETTERS]
            assert all(len(i) == 1 for i in ids), 'every option letter must be a single token'
            self.letter_ids = torch.tensor([i[0] for i in ids], device=device)
            self.cap, self.b = readout.get('final_logit_softcapping'), float(readout['multi_bias_b'])
        else:
            # a no-op when the saved tokenizer already has the marker; keeps it a single atomic token
            tok.add_tokens([MARKER], special_tokens=True)
            self.marker_id = tok.convert_tokens_to_ids(MARKER)
            assert tok(MARKER, add_special_tokens=False)['input_ids'] == [self.marker_id], (
                'the marker must tokenise to its single id'
            )
            self.w, self.b = head['weight'].float().to(device), head['bias'].float().to(device)

    @classmethod
    def from_pretrained(cls, name_or_path, device=None):
        from transformers import AutoModel, AutoModelForMultimodalLM, AutoProcessor

        path = name_or_path if os.path.isdir(name_or_path) else snapshot_download(name_or_path, allow_patterns=WEIGHT_FILES)
        device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        with open(os.path.join(path, 'readout.json')) as f:
            readout = json.load(f)
        processor = AutoProcessor.from_pretrained(path)
        if readout_kind(readout) == 'marker':
            dtype = torch.bfloat16 if device == 'cuda' else torch.float32
            model = AutoModel.from_pretrained(path, dtype=dtype).to(device)
            head = torch.load(os.path.join(path, 'head.pt'), map_location='cpu', weights_only=True)
            return cls(model, processor, readout, device, head=head)
        model = AutoModelForMultimodalLM.from_pretrained(path, dtype=torch.bfloat16).to(device)
        return cls(model, processor, readout, device)

    @torch.no_grad()
    def logits(
        self, inp, question, options=('Yes', 'No'), descriptions=None, multi=False, rules=None, context=None, input_type=None
    ):
        """Raw option logits (one float per option) before softmax / sigmoid."""
        text, audio, image = build_prompt(self.readout, inp, question, options, descriptions, multi, rules, context, input_type)
        kw = {}
        if audio is not None:
            kw['audio'] = [audio]
        if image is not None:
            kw['images'] = [image]
        if self.readout == 'letters':
            enc = self.proc(text=[text], return_tensors='pt', add_special_tokens=False, **kw).to(self.device)
            h = self.model.model(**enc, use_cache=False).last_hidden_state[0, -1].float()  # hidden state at the answer position
            z = h @ self.model.lm_head.weight[self.letter_ids[: len(options)]].float().T  # LM-head logits of the letters only
        else:
            enc = self.proc(text=[text], return_tensors='pt', **kw).to(self.device)
            marks = (enc['input_ids'][0] == self.marker_id).nonzero().flatten()
            assert len(marks) == len(options), 'every option needs its marker (the prompt was truncated)'
            h = self.model(**enc).last_hidden_state[0, marks].float()  # one hidden state per option
            z = (h @ self.w.T + self.b).squeeze(-1)
        return z.tolist()

    def __call__(
        self,
        inp,
        question,
        options=('Yes', 'No'),
        descriptions=None,
        multi=False,
        rules=None,
        context=None,
        input_type=None,
        long_audio='last',
    ):
        """Returns {option: probability}.

        inp: an audio path / (samples, sr) / 16 kHz array, an image path / PIL image, or a plain string. One-answer questions
        (multi=False) give probabilities that sum to 1; select-all questions (multi=True) give one independent probability per
        option. input_type ('audio', 'image' or 'text') skips auto-detection; always pass it for untrusted input.

        long_audio: how audio longer than 30 s is handled. 'last' (default) keeps the final 30 s, right for questions about how
        the clip ends (turn complete, next speaker). 'mean' runs every 30 s window and averages the probabilities (whole-clip
        questions: genre, language, mood). 'any' takes each option's highest probability across windows (select-all presence
        questions: "which sounds occur anywhere?"; multi=True only).
        """
        options = list(options)
        if long_audio not in ('last', 'mean', 'any'):
            raise ValueError("long_audio must be 'last', 'mean' or 'any'")
        if long_audio != 'last' and (input_type or classify_input(inp)) == 'audio':
            if long_audio == 'any' and not multi:
                raise ValueError("long_audio='any' is for select-all questions (multi=True)")
            chunks = audio_chunks(load_audio(inp, max_seconds=None))
            if len(chunks) > 1:
                ps = [self(c, question, options, descriptions, multi, rules, context, input_type='audio') for c in chunks]
                return combine_windows(ps, long_audio)
        z = self.logits(inp, question, options, descriptions, multi, rules, context, input_type)
        if self.readout == 'letters':
            p = option_probs(z, multi, self.cap, self.b)
        else:
            p = option_probs(z, multi)  # same softmax / sigmoid, no softcap and no bias: the head's own bias is already in z
        return dict(zip(options, p))

"""Smaug-Decisions on llama.cpp: the same prompt as smaug_decide.render_letters, one prefill through llama.cpp (+ libmtmd for
images and audio), the GGUF's LM-head logits of the letters A..T at the last position, softmax (one answer) or sigmoid(z - b)
(select-all).

    from llama_decide import LlamaDecide
    m = LlamaDecide('smaug-decisions-Q4_K_M-imat.gguf', mmproj='mmproj-smaug-decisions-f16.gguf', readout='Smaug-Decisions/readout.json')
    m('The order never arrived.', 'How upset is the customer?', ['calm', 'annoyed', 'angry'])
    m('clip.wav', 'Has the speaker finished their turn?')
    m('photo.jpg', 'Is there a person in the picture?')

Needs libdecide.so (decide_core.c built against llama.cpp, see build_core.sh); set LLAMA_DECIDE_LIB to its path. llama.cpp's
gemma4 graph already applies the final logit softcap (30) to the logits it returns, so it is not applied again here unless the
GGUF lacks the key. No torch / transformers needed.
"""

import ctypes
import json
import os

import numpy as np


LETTERS = 'ABCDEFGHIJKLMNOPQRST'
SAMPLE_RATE, MAX_AUDIO_SECONDS = 16000, 30.0
MAX_TOKENS, AUDIO_TOKENS_PER_SECOND, IMAGE_TOKENS = 1280, 25, 280
PREAMBLE = (
    'You make decisions for apps. Read the rules, the context and the input, then answer the question by choosing '
    'from the lettered options.'
)
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')
AUDIO_SUFFIXES = ('.wav', '.flac', '.ogg', '.mp3', '.m4a', '.aiff', '.aif')
DEC_TEXT, DEC_IMAGE_RGB, DEC_AUDIO_PCM16K = 0, 1, 2


# --- verbatim from smaug_decide/prompt.py (render_letters, trim) and smaug_decide/media.py (load_audio), so this file needs only
# numpy; parity_llama.py and tests/test_package.py assert the rendered prompts are identical ---
def trim(s, n):
    if len(s) <= n:
        return s
    return s[: int(n * 0.7)] + ' ... ' + s[-int(n * 0.3) :]


def render(question, options, descriptions, multi, media_tag, rules, context, text_input, char_budget):
    parts = [PREAMBLE]
    if rules:
        parts.append(f'Rules: {trim(rules, char_budget // 2)}')
    if context:
        parts.append(f'Context: {trim(context, char_budget)}')
    if media_tag:
        parts.append(f'Input: {media_tag}')
    elif text_input:
        parts.append(f'Input: {trim(text_input, char_budget)}')
    lines = [f'Question: {question}' + (' Select all that apply.' if multi else '')]
    for k, (opt, desc) in enumerate(zip(options, descriptions)):
        lines.append(f'{LETTERS[k]}) {opt}' + (f' — {desc}' if desc else ''))
    parts.append('\n'.join(lines))
    parts.append('Answer with the letters of all options that apply.' if multi else 'Answer with the letter of the best option.')
    return '<bos><|turn>user\n' + '\n'.join(parts) + '<turn|>\n<|turn>model\n'


def load_audio(x):
    if isinstance(x, str):
        import soundfile as sf

        w, sr = sf.read(x, dtype='float32')
    elif isinstance(x, tuple):
        w, sr = x
    else:
        w, sr = x, SAMPLE_RATE
    w = np.asarray(w, dtype=np.float32)
    if w.ndim > 1:
        w = w.mean(1)
    if sr != SAMPLE_RATE:
        n = int(round(len(w) * SAMPLE_RATE / sr))
        w = np.interp(np.linspace(0.0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)
    return w[-int(MAX_AUDIO_SECONDS * SAMPLE_RATE) :]


# --- end verbatim ---


def hf_image_size(height, width, patch=16, pool=3, max_soft_tokens=IMAGE_TOKENS):
    """transformers' Gemma4 get_aspect_ratio_preserving_size: scale (up or down) to fill max_soft_tokens * pool^2 patches, floor
    each side to a multiple of pool * patch. llama.cpp's own gemma4v preprocessing only rounds to the nearest multiple and never
    upscales, so a 500x500 image becomes ~100 tokens instead of HF's 256; resizing here first gives llama.cpp HF's exact grid."""
    import math

    max_patches, side = max_soft_tokens * pool * pool, pool * patch
    f = math.sqrt(max_patches * patch * patch / (height * width))
    th, tw = int(math.floor(f * height / side)) * side, int(math.floor(f * width / side)) * side
    max_side = (max_patches // (pool * pool)) * side
    if th == 0:
        th, tw = side, min(int(math.floor(width / height)) * side, max_side)
    elif tw == 0:
        tw, th = side, min(int(math.floor(height / width)) * side, max_side)
    return th, tw


def hf_resize(image):
    """PIL bicubic (antialiased when shrinking) to the HF grid; HF uses torchvision bicubic with antialias=True."""
    from PIL import Image

    th, tw = hf_image_size(image.height, image.width)
    return image if (th, tw) == (image.height, image.width) else image.resize((tw, th), Image.BICUBIC)


def load_lib(path=None):
    path = path or os.environ.get('LLAMA_DECIDE_LIB') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libdecide.so')
    lib = ctypes.CDLL(path)
    lib.dec_load.restype = ctypes.c_void_p
    lib.dec_load.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.dec_free.argtypes = [ctypes.c_void_p]
    lib.dec_token_id.restype = ctypes.c_int32
    lib.dec_token_id.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.dec_softcap.restype = ctypes.c_float
    lib.dec_softcap.argtypes = [ctypes.c_void_p]
    lib.dec_has_vision.argtypes = lib.dec_has_audio.argtypes = [ctypes.c_void_p]
    lib.dec_last_ms.restype = ctypes.c_double
    lib.dec_last_ms.argtypes = [ctypes.c_void_p]
    lib.dec_eval.restype = ctypes.c_int32
    lib.dec_eval.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ]
    return lib


class LlamaDecide:
    def __init__(
        self,
        gguf,
        mmproj=None,
        b=None,
        readout=None,
        n_threads=None,
        n_gpu_layers=0,
        n_ctx=2048,
        image_max_tokens=IMAGE_TOKENS,
        gpu_mm=None,
        lib=None,
        hf_image_resize=True,
    ):
        """b: the select-all threshold (multi_bias_b in readout.json); or pass readout=path to the checkpoint's readout.json."""
        self.lib = load_lib(lib)
        self.hf_resize = hf_image_resize
        n_threads = n_threads or os.cpu_count()
        gpu_mm = (n_gpu_layers != 0) if gpu_mm is None else gpu_mm
        self.d = self.lib.dec_load(
            gguf.encode(), (mmproj or '').encode(), n_threads, n_gpu_layers, n_ctx, image_max_tokens, int(gpu_mm)
        )
        if not self.d:
            raise RuntimeError(f'llama.cpp failed to load {gguf} / {mmproj}')
        ids = [self.lib.dec_token_id(self.d, L.encode()) for L in LETTERS]
        assert min(ids) >= 0, f'a letter is not a single token: {ids}'
        self.letter_ids = (ctypes.c_int32 * len(ids))(*ids)
        ro = json.load(open(readout)) if readout else {}
        b = b if b is not None else ro.get('multi_bias_b')
        self.b = None if b is None else float(b)  # select-all questions need it
        cap_in_graph = self.lib.dec_softcap(self.d)
        self.cap = None if cap_in_graph else ro.get('final_logit_softcapping')  # gemma4 GGUFs carry 30.0, applied in the graph
        self.last_tokens, self.last_ms = 0, 0.0

    def close(self):
        if self.d:
            self.lib.dec_free(self.d)
            self.d = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # interpreter shutdown: the library may already be gone
            pass

    def letter_logits(self, inp, question, options=('Yes', 'No'), descriptions=None, multi=False, rules=None, context=None):
        """Returns (z for the options' letters, rendered prompt). Same input routing and char budget as smaug_decide."""
        from PIL import Image

        options = list(options)
        assert 2 <= len(options) <= len(LETTERS), f'2 to {len(LETTERS)} options'
        descriptions = list(descriptions) if descriptions else [None] * len(options)
        audio = image = text_input = None
        if isinstance(inp, Image.Image) or (isinstance(inp, str) and inp.lower().endswith(IMAGE_SUFFIXES)):
            image = (inp if isinstance(inp, Image.Image) else Image.open(inp)).convert('RGB')
        elif isinstance(inp, str) and not inp.lower().endswith(AUDIO_SUFFIXES) and not os.path.exists(inp):
            text_input = inp
        else:
            audio = load_audio(inp)
        media_tokens = (
            len(audio) / SAMPLE_RATE * AUDIO_TOKENS_PER_SECOND + 2
            if audio is not None
            else (IMAGE_TOKENS + 2 if image is not None else 0)
        )
        tag = '<|audio|>' if audio is not None else '<|image|>' if image is not None else None
        budget = int(max(400, (MAX_TOKENS - media_tokens - 80 - 10 * len(options)) * 3.5))
        text = render(question, options, descriptions, multi, tag, rules, context, text_input, budget)
        out = (ctypes.c_float * len(options))()
        if audio is not None:
            buf = np.ascontiguousarray(audio, dtype=np.float32)
            n = self.lib.dec_eval(
                self.d, text.encode(), DEC_AUDIO_PCM16K, buf.ctypes.data, len(buf), 0, self.letter_ids, len(options), out
            )
        elif image is not None:
            image = hf_resize(image) if self.hf_resize else image
            buf = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
            n = self.lib.dec_eval(
                self.d,
                text.encode(),
                DEC_IMAGE_RGB,
                buf.ctypes.data,
                image.width,
                image.height,
                self.letter_ids,
                len(options),
                out,
            )
        else:
            n = self.lib.dec_eval(self.d, text.encode(), DEC_TEXT, None, 0, 0, self.letter_ids, len(options), out)
        if n <= 0:
            raise RuntimeError(f'dec_eval failed: {n}')
        self.last_tokens, self.last_ms = n, self.lib.dec_last_ms(self.d)
        z = np.array(out[:], dtype=np.float64)
        if self.cap:
            z = np.tanh(z / self.cap) * self.cap
        return z, text

    def __call__(self, inp, question, options=('Yes', 'No'), descriptions=None, multi=False, rules=None, context=None):
        options = list(options)
        z, _ = self.letter_logits(inp, question, options, descriptions, multi, rules, context)
        if multi:
            if self.b is None:
                raise ValueError('select-all questions need the threshold: pass readout=<readout.json> or b=')
            p = 1 / (1 + np.exp(-(z - self.b)))
        else:
            e = np.exp(z - z.max())
            p = e / e.sum()
        return dict(zip(options, p.tolist()))


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--mmproj', help='needed for image and audio inputs')
    ap.add_argument('--readout', help="the checkpoint's readout.json (select-all threshold)")
    ap.add_argument('--ngl', type=int, default=0)
    ap.add_argument('--threads', type=int)
    ap.add_argument('input')
    ap.add_argument('question')
    ap.add_argument('options', nargs='*', default=['Yes', 'No'])
    ap.add_argument('--multi', action='store_true')
    a = ap.parse_args()
    m = LlamaDecide(a.gguf, a.mmproj, readout=a.readout, n_gpu_layers=a.ngl, n_threads=a.threads)
    print(
        json.dumps(m(a.input, a.question, a.options or ['Yes', 'No'], multi=a.multi), indent=1),
        f'{m.last_tokens} positions, {m.last_ms:.0f} ms',
    )

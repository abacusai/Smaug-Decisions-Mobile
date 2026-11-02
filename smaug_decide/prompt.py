"""Prompt rendering for both model families. The prompts must match what the models expect exactly; any runtime (PyTorch, MLX,
Core ML, llama.cpp) reproduces a decision by rendering the same text and reading the same positions.

Letter readout (Smaug-Decisions): the options are lettered A..T and the answer is read from the LM-head logits of those letters at the
last position. Marker readout (Smaug-Audio-Decisions-0.7B): every option line starts with the marker token <unused0> and a linear head
scores the hidden state at each marker.
"""

from .media import SAMPLE_RATE, classify_input, load_audio, load_image


LETTERS = 'ABCDEFGHIJKLMNOPQRST'  # up to 20 options for the letter readout
MARKER = '<unused0>'  # option marker for the marker readout
AUDIO_TAG, IMAGE_TAG = '<|audio|>', '<|image|>'
AUDIO_TOKENS_PER_SECOND, IMAGE_TOKENS = 25, 280
PREAMBLE = (
    'You make decisions for apps. Read the rules, the context and the input, then answer the question by choosing '
    'from the lettered options.'
)
# token budget per readout: (max tokens, fixed overhead, per-option overhead, minimum character budget)
BUDGETS = {'letters': (1280, 80, 10, 400), 'marker': (1536, 60, 8, 300)}


def trim(s, n):
    """Keep the head and the tail of an over-long string (70/30), so both the start and the end of a long context survive."""
    if len(s) <= n:
        return s
    return s[: int(n * 0.7)] + ' ... ' + s[-int(n * 0.3) :]


def char_budget(readout, media_tokens, n_options):
    max_tokens, fixed, per_option, floor = BUDGETS[readout]
    return int(max(floor, (max_tokens - media_tokens - fixed - per_option * n_options) * 3.5))


def render_letters(question, options, descriptions, multi, media_tag, rules, context, text_input, budget):
    """Smaug-Decisions prompt: preamble, optional rules / context, the input, the lettered options, the answer instruction."""
    parts = [PREAMBLE]
    if rules:
        parts.append(f'Rules: {trim(rules, budget // 2)}')
    if context:
        parts.append(f'Context: {trim(context, budget)}')
    if media_tag:
        parts.append(f'Input: {media_tag}')
    elif text_input:
        parts.append(f'Input: {trim(text_input, budget)}')
    lines = [f'Question: {question}' + (' Select all that apply.' if multi else '')]
    for k, (opt, desc) in enumerate(zip(options, descriptions)):
        lines.append(f'{LETTERS[k]}) {opt}' + (f' — {desc}' if desc else ''))
    parts.append('\n'.join(lines))
    parts.append('Answer with the letters of all options that apply.' if multi else 'Answer with the letter of the best option.')
    return '<bos><|turn>user\n' + '\n'.join(parts) + '<turn|>\n<|turn>model\n'


def render_marker(question, options, descriptions, multi, media_tag, rules, context, text_input, budget):
    """Smaug-Audio-Decisions-0.7B prompt: a compact task line, the input, the question, one marker-prefixed line per option."""
    pre = 'task: decide | '
    if rules:
        pre += f'rules: {trim(rules, budget // 2)}\n'
    if context:
        pre += f'context: {trim(context, budget)}\n'
    pre += 'input: ' + (media_tag or trim(text_input or '', budget))
    post = f'\nquestion: {question}' + (' (select all that apply)' if multi else '') + '\n'
    for opt, desc in zip(options, descriptions):
        post += f'{MARKER} {opt}' + (f': {desc}' if desc else '') + '\n'
    return pre + post


RENDER = {'letters': render_letters, 'marker': render_marker}


def build_prompt(readout, inp, question, options, descriptions=None, multi=False, rules=None, context=None, input_type=None):
    """Render the prompt for one decision. Returns (text, audio or None, image or None).

    inp: an audio path / (samples, sample_rate) / 16 kHz float array, an image path / PIL image, or a plain string (text input).
    input_type: 'audio', 'image' or 'text' to skip auto-detection. Pass it whenever the input comes from someone else: a string is
    otherwise read as a file when it names one that exists.
    """
    options = list(options)
    if len(options) < 2:
        raise ValueError('a decision needs at least two options')
    if readout == 'letters' and len(options) > len(LETTERS):
        raise ValueError(f'the letter readout answers 2 to {len(LETTERS)} options')
    descriptions = list(descriptions) if descriptions else [None] * len(options)
    if len(descriptions) != len(options):
        raise ValueError('descriptions must have one entry per option')
    if input_type not in (None, 'audio', 'image', 'text'):
        raise ValueError("input_type must be 'audio', 'image' or 'text'")
    kind = input_type or classify_input(inp)
    audio = load_audio(inp) if kind == 'audio' else None
    image = load_image(inp) if kind == 'image' else None
    text_input = inp if kind == 'text' else None
    if audio is not None:
        media_tokens, tag = len(audio) / SAMPLE_RATE * AUDIO_TOKENS_PER_SECOND + 2, AUDIO_TAG
    elif image is not None:
        media_tokens, tag = IMAGE_TOKENS + 2, IMAGE_TAG
    else:
        media_tokens, tag = 0, None
    budget = char_budget(readout, media_tokens, len(options))
    text = RENDER[readout](question, options, descriptions, multi, tag, rules, context, text_input, budget)
    return text, audio, image

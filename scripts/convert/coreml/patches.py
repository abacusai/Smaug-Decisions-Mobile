"""Patches that make EmbeddingGemma 2 traceable and convertible to Core ML. Importing this module applies them.

- Mask builders: transformers builds bidirectional (sliding-window) masks with vmap / new_ones, which do not convert; these
  replacements build the same additive masks with plain tensor ops, and pass a ready 4-D mask through unchanged.
- RMSNorm: the Neural Engine reduces in fp16, where mean(x^2) overflows once |x| reaches about 300 over 1024 dims; each row is
  scaled by its max first (identical math: x / rms(x) == xs / sqrt(mean(xs^2) + eps / s^2)).
- coremltools ops: Tensor.unfold and movedim as static gathers / transposes, and aten::Int on shape-[1] tensors.
- freeze_audio_mask: bakes the audio tower's local-attention mask for one input size into the graph.

Import this only in a conversion process. The replacement mask builder ignores the audio tower's sliding-window mask function, so
the patched audio tower is correct only after freeze_audio_mask; never import it where the stock model serves as a reference
(parity.py does not).
"""

import importlib

import numpy as np
import torch
import transformers.masking_utils as mu


MASK_NEG = -30000.0  # fp16-safe "minus infinity" for additive masks; exp() underflows to 0 either way


def _module(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


EMBEDDING_GEMMA2 = _module('transformers.models.embedding_gemma2.modeling_embedding_gemma2')
G4 = importlib.import_module('transformers.models.gemma4.modeling_gemma4')


def _additive(allowed, dtype):
    return torch.where(allowed, torch.zeros((), dtype=dtype), torch.full((), torch.finfo(dtype).min, dtype=dtype))


def bidirectional_mask(config, inputs_embeds, attention_mask, *args, **kwargs):
    if attention_mask is None or attention_mask.dim() == 4:
        return attention_mask
    b, t = inputs_embeds.shape[0], inputs_embeds.shape[1]
    allowed = attention_mask.bool()[:, None, None, :].expand(b, 1, t, attention_mask.shape[-1])
    return _additive(allowed, inputs_embeds.dtype)


def bidirectional_sliding_mask(config, inputs_embeds, attention_mask, *args, **kwargs):
    b, t = inputs_embeds.shape[0], inputs_embeds.shape[1]
    w = config.sliding_window
    if attention_mask is None and t - 1 <= w:
        return None
    idx = torch.arange(t)
    allowed = ((idx[None, :] - idx[:, None]).abs() <= w)[None, None]
    if attention_mask is not None:
        allowed = allowed & attention_mask.bool()[:, None, None, :]
    return _additive(allowed.expand(b, 1, t, t), inputs_embeds.dtype)


ORIG_BIDIRECTIONAL_MASK = mu.create_bidirectional_mask

for _mod in (mu, EMBEDDING_GEMMA2, G4):
    if _mod is None:
        continue
    if hasattr(_mod, 'create_bidirectional_mask'):
        _mod.create_bidirectional_mask = bidirectional_mask
    if hasattr(_mod, 'create_bidirectional_sliding_window_mask'):
        _mod.create_bidirectional_sliding_window_mask = bidirectional_sliding_mask


def _norm_fp16_safe(self, hidden_states):
    s = hidden_states.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    xs = hidden_states / s
    e = (self.eps**0.5) / s
    return xs * torch.pow(xs.pow(2).mean(-1, keepdim=True) + e * e, -0.5)


G4.Gemma4RMSNorm._norm = _norm_fp16_safe
if EMBEDDING_GEMMA2 is not None and hasattr(EMBEDDING_GEMMA2, 'EmbeddingGemma2RMSNorm'):
    EMBEDDING_GEMMA2.EmbeddingGemma2RMSNorm._norm = _norm_fp16_safe


def freeze_audio_mask(tower, features, mask):
    """The audio tower's local-window blocked mask is built with a gather on float indices; bake the exact one for this size.
    Call again (after `unfreeze_audio_mask`) for every other window size."""
    tower.__dict__.pop('_convert_4d_mask_to_blocked_5d', None)
    with torch.no_grad():
        hidden, out_mask = tower.subsample_conv_projection(features, mask > 0.5)
        m4 = ORIG_BIDIRECTIONAL_MASK(
            config=tower.config,
            inputs_embeds=hidden,
            attention_mask=out_mask,
            and_mask_function=G4.sliding_window_mask_function(
                (tower.config.attention_context_left - 1, tower.config.attention_context_right)
            ),
        )
        m5 = type(tower)._convert_4d_mask_to_blocked_5d(tower, m4)
    tower._convert_4d_mask_to_blocked_5d = lambda _m: m5


def unfreeze_audio_mask(tower):
    tower.__dict__.pop('_convert_4d_mask_to_blocked_5d', None)


def register_coremltools_ops():
    """Torch ops that coremltools does not convert out of the box (or converts wrongly for these graphs)."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops as ct_ops
    from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op

    orig_cast = ct_ops._cast

    def cast_size1(
        context, node, dtype, dtype_name
    ):  # newer torch traces aten::Int on shape-[1] tensors; coremltools expects 0-d
        x = ct_ops._get_inputs(context, node, expected=1)[0]
        if x.val is not None and getattr(x.val, 'size', 0) == 1 and getattr(x.val, 'ndim', 0) > 0:
            context.add(mb.const(val=dtype(x.val.reshape(()).item()), name=node.name))
            return
        orig_cast(context, node, dtype, dtype_name)

    ct_ops._cast = cast_size1

    @register_torch_op(override=True)
    def unfold(context, node):  # Tensor.unfold on a static shape == gather with constant window indices + reshape + transpose
        x, dim, size, step = ct_ops._get_inputs(context, node, expected=4)
        dim, size, step = int(dim.val), int(size.val), int(step.val)
        shape = list(x.shape)
        rank = len(shape)
        dim = dim % rank
        n = (shape[dim] - size) // step + 1
        idx = (np.arange(n)[:, None] * step + np.arange(size)[None, :]).reshape(-1).astype(np.int32)
        g = mb.gather(x=x, indices=idx, axis=dim)
        r = mb.reshape(x=g, shape=shape[:dim] + [n, size] + shape[dim + 1 :])
        perm = list(range(dim + 1)) + list(range(dim + 2, rank + 1)) + [dim + 1]
        context.add(mb.transpose(x=r, perm=perm, name=node.name))

    @register_torch_op(override=True)
    def movedim(context, node):
        x, src, dst = ct_ops._get_inputs(context, node, expected=3)
        rank = len(x.shape)
        src = [int(v) % rank for v in np.atleast_1d(src.val)]
        dst = [int(v) % rank for v in np.atleast_1d(dst.val)]
        rest = iter([d for d in range(rank) if d not in src])
        perm = [None] * rank
        for s_, d_ in zip(src, dst):
            perm[d_] = s_
        perm = [p if p is not None else next(rest) for p in perm]
        context.add(mb.transpose(x=x, perm=perm, name=node.name))

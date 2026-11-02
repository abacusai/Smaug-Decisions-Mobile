"""Weight quantisation variants of the exported fast tier (export_onnx.py output) for ONNX Runtime Mobile.

    int8      quantize_dynamic on the weight MatMuls only, int8 per-channel weights, activations quantised at run time
              (MatMulInteger): the classic
              ORT-Mobile int8 path, fast integer kernels on every arm64 CPU (dotprod / i8mm when present)
    q8w       MatMulNBits 8-bit, block 32, symmetric (weight-only; accuracy_level 4 = int8 activations inside the kernel)
    q4w       MatMulNBits 4-bit, block 32 (weight-only, accuracy_level 4; KleidiAI kernels on arm64)
Convolutions and the tiny projections stay fp32. The fp16 embedding table is copied unchanged.
usage: quantize_onnx.py --src onnx_fp32 --out-root . --variants int8,q8w,q4w
"""
import argparse
import json
import os
import shutil

import onnx


def nbits(src, dst, bits):
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
    from onnxruntime.quantization import QuantFormat
    m = onnx.load(src)
    cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=4, quant_format=QuantFormat.QOperator,
                                       bits=bits)
    q = MatMulNBitsQuantizer(m, algo_config=cfg)
    q.process()
    onnx.save_model(q.model.model, dst, save_as_external_data=False)


def weight_matmuls(path):
    """MatMul / Gemm nodes with a constant weight input. Activation x activation products (attention scores q.k and w.v) are left
    in fp32: quantising them dynamically costs about 10x the probability error on the trunk (measured, see docs/devices.md)."""
    m = onnx.load(path, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    consts = {o for n in m.graph.node if n.op_type == 'Constant' for o in n.output}
    return [n.name for n in m.graph.node if n.op_type in ('MatMul', 'Gemm') and any(i in inits or i in consts for i in n.input[1:2])]


def dyn8(src, dst):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=['MatMul', 'Gemm'],
                     nodes_to_quantize=weight_matmuls(src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--out-root', default='.')
    ap.add_argument('--variants', default='int8,q8w,q4w')
    a = ap.parse_args()
    sizes = {}
    for v in a.variants.split(','):
        out = os.path.join(a.out_root, f'onnx_{v}')
        os.makedirs(out, exist_ok=True)
        for unit in ('audio', 'trunk'):
            src, dst = os.path.join(a.src, f'{unit}.onnx'), os.path.join(out, f'{unit}.onnx')
            if os.path.exists(dst):
                continue
            {'int8': dyn8, 'q8w': lambda s, d: nbits(s, d, 8), 'q4w': lambda s, d: nbits(s, d, 4)}[v](src, dst)
            print('wrote', dst, round(os.path.getsize(dst) / 1e6, 1), 'MB', flush=True)
        for f in ('embed_tokens_f16.bin', 'head.json'):
            if not os.path.exists(os.path.join(out, f)):
                os.link(os.path.join(a.src, f), os.path.join(out, f))
        sizes[v] = {f: round(os.path.getsize(os.path.join(out, f)) / 1e6, 1) for f in sorted(os.listdir(out))}
    print(json.dumps(sizes, indent=1))


if __name__ == '__main__':
    main()

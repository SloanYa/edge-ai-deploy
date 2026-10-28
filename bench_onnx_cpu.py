#!/usr/bin/env python3
"""W1-2:在 Mac 上用 ONNX Runtime CPU 跑一遍,验证链路通不通。

⚠️ 这个结果只能用来"确认模型导出正确",不能写进简历。
   简历要的延迟数字必须在 NVIDIA GPU 上用 TensorRT 测得。

用法:
    python bench_onnx_cpu.py --onnx best.onnx
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if not Path(args.onnx).exists():
        print(f"找不到文件:{args.onnx}")
        return 1

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]

    # 动态维度先填 1,其余按模型声明
    shape = [1 if not isinstance(d, int) or d <= 0 else d for d in inp.shape]
    x = np.random.randn(*shape).astype(np.float32)

    for _ in range(args.warmup):
        sess.run(None, {inp.name: x})

    t0 = time.perf_counter()
    for _ in range(args.iters):
        sess.run(None, {inp.name: x})
    avg = (time.perf_counter() - t0) / args.iters

    print(f"输入:{inp.name} {shape}")
    print(f"输出数:{len(sess.get_outputs())}")
    print(f"CPU 纯推理平均:{avg * 1000:.2f} ms  ({1 / avg:.1f} FPS)")
    print("\n注意:这是 CPU + 随机输入,仅验证链路。GPU 数据要去租机器测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

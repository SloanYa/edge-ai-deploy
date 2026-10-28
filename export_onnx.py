#!/usr/bin/env python3
"""W1-1:导出 YOLOv8 权重为 ONNX,并做完整性检查。

用法:
    python export_onnx.py --weights /path/to/best.pt

产出:
    best.onnx,以及终端打印的输入/输出名、形状、算子统计。
    —— 这些名字下一步写 trtexec 命令时必须原样使用。
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path


def export(weights: str, imgsz: int, opset: int, batch: int, simplify: bool) -> Path:
    from ultralytics import YOLO

    model = YOLO(weights)
    out = model.export(
        format="onnx",
        opset=opset,
        simplify=simplify,
        imgsz=imgsz,
        dynamic=False,   # 固定输入尺寸,推理路径最短
        batch=batch,
    )
    out = Path(out)
    # ultralytics 默认把 onnx 写在权重同级目录,复制一份到当前目录方便后续操作
    local = Path.cwd() / out.name
    if out.resolve() != local.resolve():
        shutil.copy2(out, local)
        print(f"已复制到当前目录:{local}")
        return local
    return out


def inspect(path: Path) -> None:
    import onnx
    from onnx import shape_inference

    print("\n" + "=" * 60)
    print(f"文件:{path}")
    print(f"大小:{path.stat().st_size / 1024 / 1024:.1f} MB")

    model = onnx.load(str(path))
    onnx.checker.check_model(model)          # 结构完整性,不通过会直接抛异常
    print("checker:通过")

    inferred = shape_inference.infer_shapes(model)
    graph = inferred.graph

    print(f"opset:{model.opset_import[0].version}")
    print(f"节点数:{len(graph.node)}")

    def show(items, tag):
        print(f"\n{tag}(共 {len(items)} 个):")
        for v in items:
            dims = [
                d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in v.type.tensor_type.shape.dim
            ]
            dtype = onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type)
            print(f"  name={v.name!r}  shape={dims}  dtype={dtype}")

    show(graph.input, "INPUT")
    show(graph.output, "OUTPUT")

    counter = Counter(n.op_type for n in graph.node)
    print("\n算子分布(top 15):")
    for op, cnt in counter.most_common(15):
        print(f"  {op:<24}{cnt}")

    risky = {"NonMaxSuppression", "TopK", "Resize", "ScatterND", "GridSample"}
    hits = risky & set(counter)
    if hits:
        print(f"\n⚠️  含 TensorRT 较敏感的算子:{sorted(hits)}")
        print("   若后面构建 engine 失败,优先怀疑它们。")
    else:
        print("\n未发现高风险算子。")

    print("=" * 60)
    print("把上面 INPUT / OUTPUT 的 name 抄下来,下一步要用。\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="YOLOv8 权重路径,如 best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--no-simplify", action="store_true", help="导出时不简化图")
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.exists():
        # 形如 yolov8n.pt 的官方模型名交给 ultralytics 自动下载
        if w.parent == Path(".") and w.suffix == ".pt":
            print(f"{args.weights} 本地不存在,尝试从 ultralytics 下载…")
        else:
            print(f"找不到权重文件:{args.weights}", file=sys.stderr)
            return 1

    onnx_path = export(
        args.weights, args.imgsz, args.opset, args.batch, not args.no_simplify
    )
    inspect(onnx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

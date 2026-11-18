# edge-ai-deploy

边缘 AI 部署链路实践：把自训练的 YOLOv8 缺陷检测模型从 PyTorch 一路部署到端侧硬件，并记录每一级的延迟与精度。

当前阶段：**W1 — 部署链路打通（ONNX 导出 + 基线测速）**

---

## 任务背景

- 任务：NEU-DET 钢材表面缺陷检测（6 类）
- 模型：YOLOv8n，自训练权重
- 目标平台：NVIDIA GPU（TensorRT）→ 后续 Zynq-7020 / PYNQ-Z2（Vitis AI DPU）

## 模型规格

| 项目 | 值 |
|---|---|
| 输入名 / 形状 | `images` / `[1, 3, 224, 224]` float32 |
| 输出名 / 形状 | `output0` / `[1, 10, 1029]` float32 |
| ONNX 大小 | 12 MB，234 节点，opset 12 |
| 输出维度含义 | 10 = 4 个框坐标 + 6 个类别；1029 = 28² + 14² + 7²（stride 8/16/32 三尺度） |

> 训练分辨率 200，被 stride=32 向上对齐到 **224**。导出与构建 engine 时输入尺寸必须一致，否则 trtexec 直接报 shape 不匹配。

## 已完成的实测

端到端口径（letterbox + 推理 + NMS），50 次预热 + 300 次正式；精度在 NEU-DET 180 张测试集上 `val`。

| 阶段 | 环境 | 延迟 | FPS | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| ONNX Runtime FP32 | Apple Silicon CPU | 43.22 ms | 23.1 | — | — |
| PyTorch FP32 | RTX 3080 (CUDA 11.7) | 32.88 ms | 30.4 | 0.7493 | 0.4060 |
| PyTorch FP16 | RTX 3080 | 31.07 ms | 32.2 | 0.7471 | 0.4062 |
| ONNX Runtime FP32 (CUDA EP) | RTX 3080 | **17.53 ms** | **57.1** | 0.7039 | 0.3681 |
| TensorRT FP32 | NVIDIA GPU | 待测 | 待测 | 待测 | 待测 |
| TensorRT FP16 | NVIDIA GPU | 待测 | 待测 | 待测 | 待测 |
| TensorRT INT8 (PTQ) | NVIDIA GPU | 待测 | 待测 | 待测 | 待测 |

**三条结论**

1. 图优化有效：ORT 相对 PyTorch FP32 端到端 **1.88×** 提速（32.88 → 17.53 ms）。
2. FP16 对 yolov8n@224 几乎无收益（仅快 5.8%，纯前向甚至更慢）。真·FP16 精度损失仅 **0.22 mAP50**。
3. ORT 的 mAP 低 4.5 点不是量化损失，而是**后处理实现差异**（ORT 走 numpy NMS，PyTorch 走 torchvision NMS）；用两个不同版本导出器重测结果完全一致（0.7039），可排除导出问题。

纯前向拆解（剥离前后处理，`cudnn.benchmark` + `autocast`）：

| 输入 | FP32 | FP16 |
|---|---|---|
| 224×224 | 23.13 ms | 29.46 ms |
| 640×640 | 23.54 ms | 28.92 ms |

224 与 640 前向耗时几乎相同 → 瓶颈是 **kernel launch 开销**而非算力（该卡实测 matmul 20.9 TFLOPS FP32 / 59.5 TFLOPS FP16，满血）。这正是 TensorRT 内核融合要解决的问题。

> CPU 数字与 GPU 数字不可直接相比，加速比只在同一硬件、同一精度口径下成立。

## 目录结构

```
.
├── export_onnx.py      # 导出 ONNX 并做完整性检查（checker / 输入输出 / 算子统计）
├── bench_onnx_cpu.py   # ONNX Runtime CPU 基线测速
├── bench_gpu.py        # GPU 端到端延迟 + 精度（PyTorch FP32 / FP16 / ORT-GPU）
├── bench_onnx.py       # 单独重测 ORT-GPU（换导出器对比用）
├── bench_pure.py       # 纯前向计时，剥离前后处理，用于定位瓶颈
├── gpu_diag.py         # GPU 规格与 matmul 吞吐自检（确认卡是否满血）
├── neu_det.yaml        # 验证集配置（路径需按本机修改）
├── bench_results.json  # 上述脚本的原始结果
├── .gitignore          # 排除 *.pt / *.onnx / *.engine（体积大、可复现）
└── README.md
```

模型文件不入库，按下方命令可自行复现。

## 复现步骤

```bash
# 1. 建环境
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics onnx onnxruntime numpy
pip freeze > requirements.txt

# 2. 导出 ONNX（imgsz 用 224，与训练分辨率对齐后的值）
python export_onnx.py --weights best.pt --imgsz 224 --opset 12

# 3. CPU 基线测速
python bench_onnx_cpu.py --onnx best.onnx --warmup 5 --iters 50
```

导出脚本会打印输入/输出名、形状和算子分布；若出现 `NonMaxSuppression`、`TopK`、`Resize` 等算子会被标为 TensorRT 敏感算子，构建 engine 失败时优先怀疑它们。

## 踩坑记录

1. **两个 `best.pt` 不是同一份**
   - 5.9 MB 那份是标准 `yolov8n`，官方 ultralytics 可正常读取并导出
   - 54 MB 那份训练时改过网络（加了 `GAM_Attention`），官方环境反序列化直接报
     `AttributeError: Can't get attribute 'GAM_Attention'`
   - 原则：**先用标准件跑通，再换自己的零件**

2. **输入尺寸必须对齐 stride**
   训练 `imgsz=200` 不能直接拿来导出，ONNX 的静态 shape 是 224。后面 trtexec 的
   `--minShapes/--optShapes/--maxShapes` 三处都要写 `1x3x224x224`。

3. **Mac 跑不了 TensorRT**
   TensorRT 只支持 NVIDIA GPU。Mac 端只做导出与 CPU 链路验证，GPU 相关的构建与测速放到租用实例上做。

## 下一步

- [x] 在 GPU 上跑通 PyTorch FP32 / FP16 / ONNX Runtime 三配置，补齐延迟与精度
- [ ] TensorRT FP32 / FP16 engine（Windows 版 TensorRT 无 pip wheel、zip 已下架，改 Linux 实例补）
- [ ] W2：INT8 PTQ（对称/非对称、per-tensor/per-channel 对比），记录量化前后 mAP 差
- [ ] W3：迁移到 PYNQ-Z2，Vitis AI 量化 → xmodel → DPU 推理

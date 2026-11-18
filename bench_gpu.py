# -*- coding: utf-8 -*-
# W1 GPU 基准:RTX 3080,YOLOv8n(NEU-DET 6 类),端到端延迟 + 精度
import os, time, glob, json
os.environ['YOLO_VERBOSE'] = 'False'
import torch
from ultralytics import YOLO
import onnxruntime as ort

BASE = r'C:\Users\87143\deploy_w1'
PT   = os.path.join(BASE, 'best.pt')
ONNX = os.path.join(BASE, 'best.onnx')
DATA = os.path.join(BASE, 'neu_det.yaml')
IMGSZ = 224
WARM, IT = 50, 300

IMGS = sorted(glob.glob(os.path.join(BASE, 'data', 'NEU-DET', 'test', 'images', '*')))
print('test images:', len(IMGS), '| cuda:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0))

# 确认 ONNX Runtime 真的在用 GPU
_sess = ort.InferenceSession(ONNX, providers=['CUDAExecutionProvider',
                                              'CPUExecutionProvider'])
print('ORT active providers:', _sess.get_providers())
del _sess

def bench_e2e(model, half=False):
    for i in range(WARM):
        model(IMGS[i % len(IMGS)], imgsz=IMGSZ, half=half, verbose=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for i in range(IT):
        model(IMGS[i % len(IMGS)], imgsz=IMGSZ, half=half, verbose=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    el = time.perf_counter() - t0
    return el / IT * 1000, IT / el

def get_map(model, half=False):
    r = model.val(data=DATA, imgsz=IMGSZ, half=half, split='val',
                  workers=0, verbose=False, plots=False)
    return float(r.box.map50), float(r.box.map)

def vram_mb():
    return torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

results = {}

print('\n== [1/3] PyTorch FP32 ==')
torch.cuda.reset_peak_memory_stats()
m = YOLO(PT)
lat, fps = bench_e2e(m, half=False)
map50, map5095 = get_map(m, half=False)
results['PyTorch_FP32'] = dict(latency_ms=round(lat, 2), fps=round(fps, 1),
                               mAP50=round(map50, 4), mAP5095=round(map5095, 4),
                               vram_MB=round(vram_mb(), 1))
print(results['PyTorch_FP32'])

print('\n== [2/3] PyTorch FP16 ==')
torch.cuda.reset_peak_memory_stats()
lat, fps = bench_e2e(m, half=True)
map50, map5095 = get_map(m, half=True)
results['PyTorch_FP16'] = dict(latency_ms=round(lat, 2), fps=round(fps, 1),
                               mAP50=round(map50, 4), mAP5095=round(map5095, 4),
                               vram_MB=round(vram_mb(), 1))
print(results['PyTorch_FP16'])

print('\n== [3/3] ONNX Runtime CUDA EP FP32 ==')
torch.cuda.reset_peak_memory_stats()
mo = YOLO(ONNX, task='detect')
lat, fps = bench_e2e(mo, half=False)
map50, map5095 = get_map(mo, half=False)
results['ORT_GPU_FP32'] = dict(latency_ms=round(lat, 2), fps=round(fps, 1),
                               mAP50=round(map50, 4), mAP5095=round(map5095, 4),
                               vram_MB=round(vram_mb(), 1))
print(results['ORT_GPU_FP32'])

with open(os.path.join(BASE, 'bench_results.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('\nSAVED:', os.path.join(BASE, 'bench_results.json'))

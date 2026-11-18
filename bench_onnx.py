# -*- coding: utf-8 -*-
import os, time, glob, json
os.environ['YOLO_VERBOSE'] = 'False'
import torch
from ultralytics import YOLO
import onnxruntime as ort

BASE = r'C:\Users\87143\deploy_w1'
ONNX = os.path.join(BASE, 'best.onnx')
DATA = os.path.join(BASE, 'neu_det.yaml')
IMGSZ, WARM, IT = 224, 50, 300
IMGS = sorted(glob.glob(os.path.join(BASE, 'data', 'NEU-DET', 'test', 'images', '*')))

_s = ort.InferenceSession(ONNX, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
print('ORT providers:', _s.get_providers()); del _s

m = YOLO(ONNX, task='detect')
for i in range(WARM): m(IMGS[i % len(IMGS)], imgsz=IMGSZ, verbose=False)
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(IT): m(IMGS[i % len(IMGS)], imgsz=IMGSZ, verbose=False)
torch.cuda.synchronize()
el = time.perf_counter() - t0
lat, fps = el/IT*1000, IT/el
r = m.val(data=DATA, imgsz=IMGSZ, split='val', workers=0, verbose=False, plots=False)
out = dict(latency_ms=round(lat,2), fps=round(fps,1),
           mAP50=round(float(r.box.map50),4), mAP5095=round(float(r.box.map),4),
           vram_MB=round(torch.cuda.max_memory_allocated()/1024/1024,1))
print('ORT_GPU_FP32_NEW:', out)
res = json.load(open(os.path.join(BASE,'bench_results.json')))
res['ORT_GPU_FP32_newexport'] = out
json.dump(res, open(os.path.join(BASE,'bench_results.json'),'w'), indent=2, ensure_ascii=False)
print('SAVED')

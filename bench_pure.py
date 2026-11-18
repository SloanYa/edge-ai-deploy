# -*- coding: utf-8 -*-
# 纯前向推理计时 v2:cudnn.benchmark + autocast,224 与 640 两个输入尺寸
import os, time, json
os.environ['YOLO_VERBOSE'] = 'False'
import torch
from ultralytics import YOLO
torch.backends.cudnn.benchmark = True

BASE = r'C:\Users\87143\deploy_w1'
m = YOLO(os.path.join(BASE, 'best.pt'))
net = m.model.eval().cuda()
WARM, IT = 200, 500

def run(size, half):
    x = torch.randn(1, 3, size, size).cuda()
    def f():
        with torch.no_grad():
            if half:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    net(x)
            else:
                net(x)
    for _ in range(WARM): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(IT): f()
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    return el/IT*1000

out = {}
for size in (224, 640):
    f32 = run(size, False); f16 = run(size, True)
    out[f'forward_{size}_FP32_ms'] = round(f32, 3)
    out[f'forward_{size}_FP16_ms'] = round(f16, 3)
    out[f'forward_{size}_fp16_gain_pct'] = round((f32-f16)/f32*100, 1)
print(out)
res = json.load(open(os.path.join(BASE,'bench_results.json')))
res['pure_forward_v2'] = out
json.dump(res, open(os.path.join(BASE,'bench_results.json'),'w'), indent=2, ensure_ascii=False)
print('SAVED')

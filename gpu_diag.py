# -*- coding: utf-8 -*-
import torch, time
p = torch.cuda.get_device_properties(0)
print('GPU:', p.name, '| SM', p.major, p.minor, '| SMs', p.multi_processor_count,
      '| Mem', round(p.total_memory/1024**3,1), 'GB')
# 原始 matmul 吞吐 (4096^3)
a = torch.randn(4096, 4096, device='cuda'); b = torch.randn(4096, 4096, device='cuda')
for _ in range(10): torch.matmul(a, b)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50): torch.matmul(a, b)
torch.cuda.synchronize(); dt = (time.perf_counter()-t0)/50
print(f'matmul 4096^3 FP32: {dt*1000:.2f} ms, {2*4096**3/dt/1e12:.1f} TFLOPS')
ah, bh = a.half(), b.half()
for _ in range(10): torch.matmul(ah, bh)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50): torch.matmul(ah, bh)
torch.cuda.synchronize(); dt = (time.perf_counter()-t0)/50
print(f'matmul 4096^3 FP16: {dt*1000:.2f} ms, {2*4096**3/dt/1e12:.1f} TFLOPS')
print('current clock MHz:', torch.cuda.clock_rate()//1000)

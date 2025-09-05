# rm -rf /tmp/torchinductor* && rm -rf torch_compile_debug* && TORCHINDUCTOR_MAX_AUTOTUNE=1 TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS="TRITON" TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_BENCHMARK_KERNEL=1 python conv.py

import torch
import torch.nn.functional as F
 
torch._inductor.config.max_autotune = True
 
@torch.compile(mode="max-autotune")
def conv_forward(x, weight):
    return F.conv2d(x, weight, padding=1)
 
#x = torch.randn(1, 3, 32, 32).cuda()
#weight = torch.randn(16, 3, 3, 3).cuda()
#output = conv_forward(x, weight)

# Row 17 of production
#convbfp16 -n 128 -c 384 -H 24 -W 48 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -t 1 -b 0 -F 1

#  triton_convolution2d_4 0.2090 ms 100.0% ALLOW_TF32=True, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, GROUPS=1, KERNEL_H=1, KERNEL_W=1, PADDING_H=0, PADDING_W=0, STRIDE_H=1, STRIDE_W=1, UNROLL=True, matrix_instr_nonkdim=0, num_stages=2, num_warps=8

# Row 21 of production
#convbfp16 -n 16 -c 768 -H 48 -W 32 -k 2048 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -t 1 -b 0 -F 1

#  triton_convolution2d_4 4.1383 ms 100.0% ALLOW_TF32=True, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, GROUPS=1, KERNEL_H=3, KERNEL_W=3, PADDING_H=1, PADDING_W=1, STRIDE_H=1, STRIDE_W=1, UNROLL=False, matrix_instr_nonkdim=0, num_stages=2, num_warps=8
 
x = torch.randn(16, 768, 48, 32, dtype=torch.bfloat16).cuda()
weight = torch.randn(2048, 768, 3, 3, dtype=torch.bfloat16).cuda()
output = conv_forward(x, weight)
 
print(f"Output shape: {output.shape}")

import re
import sys

import triton
from triton.backends.compiler import GPUTarget

if len(sys.argv) != 3:
    raise SystemExit(f"usage: {sys.argv[0]} <kernel.ttgir> <out.hsaco>")

ttgir = sys.argv[1]
out = sys.argv[2]
k = triton.compile(ttgir, target=GPUTarget("hip", "gfx950", 64))
amd = k.asm["amdgcn"]
open(out, "wb").write(k.asm["hsaco"])
print("kernel name :", k.metadata.name)
print("shared bytes:", k.metadata.shared)
print("ds_read_b64 :", len(re.findall(r"ds_read_b64 v\[", amd)))
print("s_barrier   :", len(re.findall(r"s_barrier", amd)))
open(out + ".amdgcn", "w").write(amd)

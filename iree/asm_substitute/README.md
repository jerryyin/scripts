# IREE Assembly Hijacking

Inject modified GPU assembly into IREE compilation for performance experiments.

## Important

**The assembly MUST come from the SAME compilation as your baseline vmfb.**
IREE compilation can produce slightly different code between runs due to:
- Hash-based ordering
- Optimization non-determinism
- Other factors

Always use the assembly dumped from your baseline compilation.

## Quick Start

```bash
# 1. Compile and dump assembly
iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx950 \
    --iree-llvmgpu-use-direct-load \
    --iree-hal-dump-executable-intermediates-to=./ir_dump \
    input.mlir -o baseline.vmfb

# 2. Copy and modify assembly
cp ./ir_dump/module_*_rocm_hsaco_fb.rocmasm modified.s
vim modified.s  # Make changes

# 3. Hijack compilation
./hijack_asm.sh modified.s input.mlir modified.vmfb matmul_dispatch_0 \
    "--iree-llvmgpu-use-direct-load"

# 4. Test
iree-run-module --module=baseline.vmfb --device=hip --input=... --output=@baseline.bin
iree-run-module --module=modified.vmfb --device=hip --input=... --output=@modified.bin
```

## How It Works

1. **Assemble**: `clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c modified.s -o modified.o`
2. **Link**: `ld.lld -shared modified.o -o modified.hsaco`
3. **Substitute**: `iree-compile ... --iree-hal-substitute-executable-object=<exec>=modified.hsaco`

The `--iree-hal-substitute-executable-object` flag completely replaces the GPU binary, so structural changes (adding/removing instructions) work fine.

## Finding the Executable Name

The exec name is derived from the `.rocmasm` filename:
- `module_matmul_dispatch_0_rocm_hsaco_fb.rocmasm` → `matmul_dispatch_0`

Or look in the MLIR for `stream.executable @<name>` or `hal.executable @<name>`.

## Example: Experimenting with vmcnt Placement

```bash
# Get original assembly
cat original.s | grep -A5 "s_waitcnt vmcnt"
#   s_waitcnt vmcnt(0)
#   s_barrier
#   buffer_load_dwordx4 ...

# Create modified version with vmcnt after loads
cat > modified.s << 'EOF'
... (copy everything before the loop)
.LBB0_1:
    s_mov_b64 s[20:21], s[2:3]
    ... (setup)
    buffer_load_dwordx4 ...   # Issue loads FIRST
    buffer_load_dwordx4 ...
    buffer_load_dwordx4 ...
    buffer_load_dwordx4 ...
    s_waitcnt vmcnt(4)        # Wait for OLD loads
    s_barrier
    ... (rest of loop)
EOF

# Test
./hijack_asm.sh modified.s input.mlir test.vmfb
```

## Common Assembly Patterns

```asm
# Wait for vector memory (global loads)
s_waitcnt vmcnt(0)      # Wait for all
s_waitcnt vmcnt(4)      # Wait until ≤4 outstanding

# Wait for LDS operations
s_waitcnt lgkmcnt(0)

# Workgroup barrier
s_barrier

# Load global → LDS
buffer_load_dwordx4 v10, s[4:7], 0 offen lds

# Read from LDS
ds_read2_b32 v[84:85], v91 offset1:4

# Matrix multiply
v_mfma_f32_16x16x4_f32 v[48:51], v84, v80, v[48:51]
```

## Environment Variables

- `ROCM_LLVM` - Path to ROCm LLVM tools (default: `/opt/rocm/llvm/bin`)
- `GPU_TARGET` - GPU architecture (default: `gfx950`)
- `IREE_COMPILE` - Path to iree-compile (default: from PATH)

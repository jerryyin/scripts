#!/usr/bin/env python3
"""
Flash Attention Forward Kernel Tool

Compile and/or verify the Flash Attention forward kernel.

Usage:
    # Compile only (default) - generates IR artifacts
    python flash_attention_tool.py
    
    # Compile with specific parameters
    python flash_attention_tool.py --batch 4 --heads 32 --seq-len 1024 --head-dim 128 --causal
    
    # Verify correctness against PyTorch
    python flash_attention_tool.py --verify
    
    # Both compile and verify
    python flash_attention_tool.py --verify --batch 2 --heads 4 --seq-len 512
    
    # List all 48 test combinations
    python flash_attention_tool.py --list-configs

Environment Variables:
    TRITON_SAVETEMPS_DIR: Directory to save IR artifacts (default: ./ir_output)
"""

import argparse
import os
import sys
from dataclasses import dataclass
from itertools import product
from typing import Optional

# Add triton to path
sys.path.insert(0, "/root/triton-mi450/python")

import triton
import triton.language as tl

# Only import torch if needed for verification
torch = None


def lazy_import_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch


# =============================================================================
# Kernel Definition (from official tutorial)
# =============================================================================

@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    desc_k, desc_v,
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    
    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(sm_scale, M,
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr):
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = desc_q.load([qo_offset_y, 0])
    
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,
                                        desc_k, desc_v,
                                        offset_y, dtype, start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        4 - STAGE, offs_m, offs_n, N_CTX)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,
                                        desc_k, desc_v,
                                        offset_y, dtype, start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        2, offs_m, offs_n, N_CTX)
    
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AttentionConfig:
    """Configuration for Flash Attention kernel."""
    batch: int = 1          # Z in tutorial
    heads: int = 2          # H in tutorial
    seq_len: int = 128      # N_CTX in tutorial
    head_dim: int = 64      # HEAD_DIM
    causal: bool = False
    block_m: int = 128      # BLOCK_M (tile size)
    block_n: int = 64       # BLOCK_N (tile size)
    num_stages: int = 1     # Pipeline stages (affects buffering)
    
    def __str__(self):
        return (f"B{self.batch}_H{self.heads}_N{self.seq_len}_D{self.head_dim}_"
                f"{'causal' if self.causal else 'full'}_BM{self.block_m}_BN{self.block_n}"
                f"_NS{self.num_stages}")
    
    @property
    def stage(self) -> int:
        """STAGE parameter: 3 for causal, 1 for non-causal."""
        return 3 if self.causal else 1


def get_all_test_configs() -> list[AttentionConfig]:
    """
    Generate all 48 test configurations from the tutorial.
    
    Based on pytest parametrize:
        Z: [1, 4]           -> 2
        H: [2, 48]          -> 2
        N_CTX: [128, 1024, 2048] -> 3 (HIP)
        HEAD_DIM: [64, 128] -> 2
        causal: [False, True] -> 2
        
    Total: 2 × 2 × 3 × 2 × 2 = 48
    """
    configs = []
    
    for batch, heads, seq_len, head_dim, causal in product(
        [1, 4],                 # Z (batch)
        [2, 48],                # H (heads)
        [128, 1024, 2048],      # N_CTX (seq_len)
        [64, 128],              # HEAD_DIM
        [False, True],          # causal
    ):
        configs.append(AttentionConfig(
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            head_dim=head_dim,
            causal=causal,
        ))
    
    return configs


def resolve_output_dir(out_dir: Optional[str]) -> str:
    """
    Resolve output directory with consistent priority.
    
    Priority: explicit arg > TRITON_SAVETEMPS_DIR env var > ./ir_output
    
    Args:
        out_dir: Explicitly specified output directory, or None
    
    Returns:
        Resolved absolute path to output directory
    """
    if out_dir:
        return os.path.abspath(out_dir)
    
    env_dir = os.environ.get("TRITON_SAVETEMPS_DIR", "").strip()
    if env_dir:
        return os.path.abspath(env_dir)
    
    return os.path.abspath(os.path.join(os.getcwd(), "ir_output"))


# =============================================================================
# Compile Function
# =============================================================================

def compile_kernel(config: AttentionConfig, save_dir: str) -> dict:
    """
    Compile the attention kernel and save IR artifacts.
    
    Args:
        config: Attention configuration
        save_dir: Directory to save IR (must be provided)
    
    Returns:
        Dict with compilation info and artifact paths.
    """
    from triton.compiler import ASTSource, compile, make_backend
    
    target = triton.runtime.driver.active.get_current_target()
    
    # Initialize kernel binder
    _attn_fwd.create_binder()
    
    # Signature
    signature = {
        "sm_scale": "fp32",
        "M": "*fp32",
        "Z": "i32",
        "H": "i32",
        "desc_q": "*fp16",
        "desc_k": "*fp16",
        "desc_v": "*fp16",
        "desc_o": "*fp16",
        "N_CTX": "i32",
    }
    
    # Constexprs
    constexprs = {
        "HEAD_DIM": config.head_dim,
        "BLOCK_M": config.block_m,
        "BLOCK_N": config.block_n,
        "STAGE": config.stage,
    }
    
    # Create source and compile
    src = ASTSource(fn=_attn_fwd, signature=signature, constexprs=constexprs)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 4, "num_stages": config.num_stages})
    
    compiled = compile(src, target=target, options=options.__dict__)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save artifacts with natural file extensions
    artifacts = {}
    prefix = f"attn_fwd_{config}"
    
    # Map stage names to file extensions
    # "source" is MLIR with debug locations, not Python
    ext_map = {"source": "mlir"}
    
    if hasattr(compiled, 'asm'):
        for stage, content in compiled.asm.items():
            ext = ext_map.get(stage, stage)  # e.g., "source" -> "py", others unchanged
            filepath = os.path.join(save_dir, f"{prefix}.{ext}")
            
            if isinstance(content, str):
                with open(filepath, 'w') as f:
                    f.write(content)
            elif isinstance(content, bytes):
                with open(filepath, 'wb') as f:
                    f.write(content)
            
            artifacts[stage] = filepath
    
    return {
        "target": str(target),
        "config": str(config),
        "artifacts": artifacts,
        "save_dir": save_dir,
    }


def analyze_asm(asm_filepath: str) -> dict:
    """
    Analyze assembly file for key instructions.
    
    Args:
        asm_filepath: Path to the .amdgcn assembly file
    
    Returns:
        Dict with instruction counts.
    """
    patterns = {
        "TDM load": "tensor_load_to_lds",
        "TDM store": "tensor_store_from_lds", 
        "WMMA": "v_wmma",
        "ds_load_tr": "ds_load_tr",
    }
    
    with open(asm_filepath, 'r') as f:
        content = f.read()
    
    results = {}
    for name, pattern in patterns.items():
        count = content.lower().count(pattern.lower())
        if count > 0:
            results[name] = count
    
    return results


def analyze_all_configs(save_dir: str) -> int:
    """
    Compile all 48 test configs and print instruction stats table.
    
    Args:
        save_dir: Directory to save IR artifacts (must be provided)
    
    Returns:
        Exit code (0 for success)
    """
    configs = get_all_test_configs()
    target = triton.runtime.driver.active.get_current_target()
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 90)
    print("Flash Attention - Analyze All 48 Configurations")
    print(f"Target: {target}")
    print(f"Output: {save_dir}")
    print("=" * 90)
    
    # Print header
    print(f"\n{'ID':>3} | {'B':>1} | {'H':>2} | {'N':>4} | {'D':>3} | {'Mode':>6} | "
          f"{'TDM ld':>6} | {'TDM st':>6} | {'WMMA':>4} | {'ds_tr':>5}")
    print("-" * 90)
    
    for i, cfg in enumerate(configs):
        # Compile
        result = compile_kernel(cfg, save_dir=save_dir)
        
        # Get the .amdgcn file
        asm_file = result['artifacts'].get('amdgcn')
        if asm_file and os.path.exists(asm_file):
            stats = analyze_asm(asm_file)
        else:
            stats = {}
        
        causal_str = "causal" if cfg.causal else "full"
        tdm_ld = stats.get("TDM load", 0)
        tdm_st = stats.get("TDM store", 0)
        wmma = stats.get("WMMA", 0)
        ds_tr = stats.get("ds_load_tr", 0)
        
        print(f"{i:3d} | {cfg.batch:1d} | {cfg.heads:2d} | {cfg.seq_len:4d} | {cfg.head_dim:3d} | "
              f"{causal_str:>6} | {tdm_ld:6d} | {tdm_st:6d} | {wmma:4d} | {ds_tr:5d}")
    
    print("-" * 90)
    print(f"\nAll {len(configs)} configurations compiled successfully.")
    print(f"IR artifacts saved to: {save_dir}")
    
    return 0


# =============================================================================
# Verify Function
# =============================================================================

def verify_kernel(config: AttentionConfig, sm_scale: float = 0.5) -> dict:
    """
    Verify Triton kernel against PyTorch reference.
    
    Returns:
        Dict with verification results.
    """
    torch = lazy_import_torch()
    
    device = triton.runtime.driver.active.get_active_torch_device()
    
    # Create random inputs
    torch.manual_seed(42)
    q = torch.randn((config.batch, config.heads, config.seq_len, config.head_dim),
                    dtype=torch.float16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    # PyTorch reference
    p = torch.matmul(q.float(), k.float().transpose(2, 3)) * sm_scale
    if config.causal:
        mask = torch.tril(torch.ones((config.seq_len, config.seq_len), device=device))
        p[:, :, mask == 0] = float("-inf")
    p = torch.softmax(p, dim=-1)
    ref_out = torch.matmul(p, v.float()).half()
    
    # Triton kernel
    o = torch.empty_like(q)
    M = torch.empty((config.batch, config.heads, config.seq_len), 
                    device=device, dtype=torch.float32)
    
    grid = (triton.cdiv(config.seq_len, config.block_m), config.batch * config.heads)
    
    _attn_fwd[grid](
        sm_scale, M,
        config.batch, config.heads, q, k, v, o, config.seq_len,
        HEAD_DIM=config.head_dim,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        STAGE=config.stage,
        num_warps=4,
        num_stages=config.num_stages,
    )
    
    # Compare
    max_diff = (o - ref_out).abs().max().item()
    mean_diff = (o - ref_out).abs().mean().item()
    atol = 1e-2
    passed = max_diff < atol
    
    return {
        "passed": passed,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "atol": atol,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Flash Attention Forward Kernel Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Mode selection
    parser.add_argument("--verify", action="store_true",
                        help="Verify correctness against PyTorch reference")
    parser.add_argument("--list-configs", action="store_true",
                        help="List all 48 test configurations and exit")
    parser.add_argument("--config-id", type=int, metavar="N",
                        help="Use config #N from the 48 test configs (0-47)")
    parser.add_argument("--analyze-all", action="store_true",
                        help="Compile all 48 configs and print instruction stats table")
    
    # Kernel parameters
    parser.add_argument("--batch", "-b", type=int, default=1,
                        help="Batch size Z (default: 1)")
    parser.add_argument("--heads", "-H", type=int, default=2,
                        help="Number of heads H (default: 2)")
    parser.add_argument("--seq-len", "-n", type=int, default=128,
                        help="Sequence length N_CTX (default: 128)")
    parser.add_argument("--head-dim", "-d", type=int, default=64,
                        help="Head dimension HEAD_DIM (default: 64)")
    parser.add_argument("--causal", "-c", action="store_true",
                        help="Use causal masking")
    
    # Tile sizes
    parser.add_argument("--block-m", type=int, default=128,
                        help="BLOCK_M tile size (default: 128)")
    parser.add_argument("--block-n", type=int, default=64,
                        help="BLOCK_N tile size (default: 64)")
    parser.add_argument("--num-stages", type=int, default=1,
                        help="Pipeline stages (default: 1)")
    
    # Output
    parser.add_argument("--out-dir", "-o", type=str, default=None,
                        help="Output directory for IR (default: TRITON_SAVETEMPS_DIR or ./ir_output)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output")
    
    args = parser.parse_args()
    
    # List all configs
    if args.list_configs:
        configs = get_all_test_configs()
        print(f"All {len(configs)} test configurations:")
        print("-" * 80)
        for i, cfg in enumerate(configs):
            causal_str = "causal" if cfg.causal else "full  "
            print(f"  {i:2d}: B={cfg.batch}, H={cfg.heads:2d}, N={cfg.seq_len:4d}, "
                  f"D={cfg.head_dim:3d}, {causal_str}")
        print("-" * 80)
        print("\nUse --config-id N to compile a specific configuration.")
        return 0
    
    # Resolve output directory once
    out_dir = resolve_output_dir(args.out_dir)
    
    # Analyze all configs
    if args.analyze_all:
        return analyze_all_configs(out_dir)
    
    # Build config
    if args.config_id is not None:
        configs = get_all_test_configs()
        if args.config_id < 0 or args.config_id >= len(configs):
            print(f"Error: --config-id must be 0-{len(configs)-1}")
            return 1
        config = configs[args.config_id]
        config.num_stages = args.num_stages
    else:
        config = AttentionConfig(
            batch=args.batch,
            heads=args.heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            causal=args.causal,
            block_m=args.block_m,
            block_n=args.block_n,
            num_stages=args.num_stages,
        )
    
    # Get target info
    target = triton.runtime.driver.active.get_current_target()
    
    if not args.quiet:
        print("=" * 60)
        print("Flash Attention Forward Kernel Tool")
        print(f"Target: {target}")
        print(f"Config: {config}")
        print("=" * 60)
    
    # Compile
    if not args.quiet:
        print("\n[Compiling...]")
    
    result = compile_kernel(config, save_dir=out_dir)
    
    if not args.quiet:
        print(f"  Saved to: {result['save_dir']}")
        for stage, path in result['artifacts'].items():
            size = os.path.getsize(path)
            print(f"    {stage}: {os.path.basename(path)} ({size:,} bytes)")
    
    # Analyze assembly
    asm_file = result['artifacts'].get('amdgcn')
    if asm_file and os.path.exists(asm_file) and not args.quiet:
        ir_analysis = analyze_asm(asm_file)
        if ir_analysis:
            print("\n[IR Analysis]")
            for instr, count in ir_analysis.items():
                print(f"    {instr}: {count}")
    
    # Verify if requested
    if args.verify:
        if not args.quiet:
            print("\n[Verifying...]")
        
        verify_result = verify_kernel(config)
        
        status = "✅ PASS" if verify_result["passed"] else "❌ FAIL"
        print(f"  {status}")
        print(f"    max_diff:  {verify_result['max_diff']:.6f}")
        print(f"    mean_diff: {verify_result['mean_diff']:.6f}")
        print(f"    atol:      {verify_result['atol']}")
        
        if not verify_result["passed"]:
            return 1
    
    if not args.quiet:
        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

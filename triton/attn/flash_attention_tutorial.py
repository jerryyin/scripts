"""
==============================================================================
Flash Attention Tutorial: From Intuition to Triton Code
==============================================================================

This tutorial explains Flash Attention step by step, assuming you only know:
- Attention is: GEMM → Softmax → GEMM
- Basic Python/NumPy

We'll build up from the naive algorithm to the full Flash Attention kernel.

==============================================================================
PART 1: What is Attention?
==============================================================================

In transformers, attention lets each position "look at" all other positions
and decide which ones are relevant. For a sequence of tokens:

    Input: "The cat sat on the mat"
    
When processing "sat", attention might focus on "cat" (who sat?) and "mat" (where?).

The formula for scaled dot-product attention is:

    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V

Where:
- Q (Query):  [M, D] - "What am I looking for?"
- K (Key):    [N, D] - "What do I contain?"  
- V (Value):  [N, D] - "What information do I have?"
- Output:     [M, D] - Weighted combination of values

The steps are:
1. Compute similarity: S = Q @ K.T              → [M, N] matrix
2. Scale:              S = S / sqrt(D)          → prevents huge values
3. Softmax:            P = softmax(S, dim=-1)   → each row sums to 1
4. Weight values:      O = P @ V                → [M, D] output
"""

import torch
import triton
import triton.language as tl
import numpy as np


# ==============================================================================
# PART 2: Naive Attention in Python
# ==============================================================================

def naive_attention(Q, K, V):
    """
    Naive attention implementation - simple but memory-hungry.
    
    Memory usage: O(M * N) for the attention matrix!
    For a 4K sequence, that's 4096 * 4096 = 16M elements per head.
    """
    M, D = Q.shape
    N, _ = K.shape
    
    # Step 1: Compute all pairwise similarities
    # Q @ K.T: [M, D] @ [D, N] = [M, N]
    scores = Q @ K.T
    
    # Step 2: Scale to prevent softmax saturation
    scale = 1.0 / np.sqrt(D)
    scores = scores * scale
    
    # Step 3: Softmax along the key dimension (each row sums to 1)
    # This tells us: for each query position, what's the weight of each key?
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))  # subtract max for stability
    attention_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    # Step 4: Weighted sum of values
    # P @ V: [M, N] @ [N, D] = [M, D]
    output = attention_weights @ V
    
    return output


# ==============================================================================
# PART 3: The Memory Problem
# ==============================================================================
"""
The problem with naive attention:

    Sequence Length    Attention Matrix Size    Memory (FP32)
    ─────────────────────────────────────────────────────────
    512                512 × 512 = 262K         1 MB
    2,048              2K × 2K = 4M             16 MB
    8,192              8K × 8K = 67M            256 MB
    32,768             32K × 32K = 1B           4 GB  ← per head!

For GPT-4 with 96 heads and 32K context, you'd need 384 GB just for attention!

Flash Attention solves this by NEVER materializing the full [M, N] matrix.
Instead, it processes attention in blocks and uses a clever trick called
"online softmax" to accumulate results without storing everything.
"""


# ==============================================================================
# PART 4: The Key Insight - Online Softmax
# ==============================================================================
"""
Normal softmax requires knowing ALL values first:
    
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
                       ↑                    ↑
                  Need to know max      Need to know sum
                  of ALL elements       of ALL exp values

BUT - we can compute softmax INCREMENTALLY using this trick:

    When we see new values, we can UPDATE our running result!

Let's say we've processed elements [x₀, x₁, x₂] and now see x₃:

    Old max: m = max(x₀, x₁, x₂)
    New max: m' = max(m, x₃)
    
    Old sum: l = exp(x₀-m) + exp(x₁-m) + exp(x₂-m)
    New sum: l' = l × exp(m - m') + exp(x₃ - m')
                     ↑
             Correction factor! Rescales old values

This is called "online softmax" and it's the heart of Flash Attention.
"""


def online_softmax_demo():
    """Demonstrate online softmax computation."""
    
    # True values
    x = np.array([1.0, 3.0, 2.0, 4.0])
    
    # Ground truth softmax
    true_softmax = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
    print(f"True softmax: {true_softmax}")
    print(f"Sum: {true_softmax.sum()}")
    
    # Online computation
    m = float('-inf')  # running max
    l = 0.0            # running sum
    
    for i, xi in enumerate(x):
        # Update max
        m_new = max(m, xi)
        
        # Correct the old sum for the new max, then add new element
        l = l * np.exp(m - m_new) + np.exp(xi - m_new)
        
        m = m_new
        print(f"After x[{i}]={xi}: max={m:.2f}, sum_exp={l:.4f}")
    
    # Final softmax
    online_softmax = np.exp(x - m) / l
    print(f"\nOnline softmax: {online_softmax}")
    print(f"Difference: {np.abs(true_softmax - online_softmax).max():.2e}")


# ==============================================================================
# PART 5: Flash Attention Algorithm
# ==============================================================================
"""
Flash Attention applies online softmax to attention:

    Instead of computing the full [M, N] attention matrix:
    
    1. Load a BLOCK of Q values (e.g., 64 rows)
    2. FOR each block of K, V:
       a. Compute QK^T for this block only         → [BLOCK_M, BLOCK_N]
       b. Update running max, sum, and output
    3. After all K,V blocks: normalize output
    
    Memory: Only need O(BLOCK_M × BLOCK_N) instead of O(M × N)
    
    ASCII visualization:
    
                K₀      K₁      K₂      K₃
              ┌──────┬──────┬──────┬──────┐
         Q₀  │  ★   │  →   │  →   │  →   │   Process block by block
              ├──────┼──────┼──────┼──────┤   ★ = current block
         Q₁  │      │      │      │      │   → = will process next
              └──────┴──────┴──────┴──────┘
              
    Each GPU thread block processes one Q block (row of blocks).
    It iterates through ALL K,V blocks, updating its running softmax.
"""


def flash_attention_python(Q, K, V, BLOCK_M=64, BLOCK_N=64):
    """
    Flash Attention in pure Python - LINE-BY-LINE identical to Triton version.
    
    Compare this directly with flash_attention_kernel() below.
    Every variable name and operation matches exactly.
    """
    M, D = Q.shape
    N_CTX, _ = K.shape
    HEAD_DIM = D
    
    # Output tensor
    Out = np.zeros((M, HEAD_DIM), dtype=np.float32)
    
    # Scale factor (same as Triton)
    qk_scale = 1.0 / np.sqrt(HEAD_DIM)
    
    # Grid: number of Q blocks (in Triton, this is the grid size)
    num_programs = (M + BLOCK_M - 1) // BLOCK_M
    
    # ==========================================================================
    # OUTER LOOP: In Triton, each iteration is a separate thread block
    # ==========================================================================
    for start_m in range(num_programs):  # ← tl.program_id(0) in Triton
        
        # ----------------------------------------------------------------------
        # Step 1: Compute offsets (same logic as Triton)
        # ----------------------------------------------------------------------
        offs_m = start_m * BLOCK_M + np.arange(BLOCK_M)  # ← tl.arange(0, BLOCK_M)
        offs_d = np.arange(HEAD_DIM)                      # ← tl.arange(0, HEAD_DIM)
        
        # ----------------------------------------------------------------------
        # Step 2: Load Q block
        # ----------------------------------------------------------------------
        mask_m = offs_m < M
        q = np.where(mask_m[:, None], Q[np.minimum(offs_m, M-1), :], 0.0)
        
        # ----------------------------------------------------------------------
        # Step 3: Initialize running statistics
        # ----------------------------------------------------------------------
        m_i = np.full(BLOCK_M, float("-inf"), dtype=np.float32)  # Max scores
        l_i = np.zeros(BLOCK_M, dtype=np.float32) + 1.0          # Sum of exp
        acc = np.zeros((BLOCK_M, HEAD_DIM), dtype=np.float32)     # Output acc
        
        # ----------------------------------------------------------------------
        # Step 4: Iterate through ALL K, V blocks
        # ----------------------------------------------------------------------
        for start_n in range(0, N_CTX, BLOCK_N):
            offs_n = np.arange(BLOCK_N)
            
            # Load K block (transposed): [HEAD_DIM, BLOCK_N]
            mask_n = (start_n + offs_n) < N_CTX
            k_indices = np.minimum(start_n + offs_n, N_CTX-1)
            k = np.where(mask_n[None, :], K[k_indices, :].T, 0.0)  # [D, BLOCK_N]
            
            # Compute QK^T: [BLOCK_M, D] @ [D, BLOCK_N] = [BLOCK_M, BLOCK_N]
            qk = q @ k
            
            # Online softmax: new max
            m_ij = np.maximum(m_i, np.max(qk, axis=1) * qk_scale)
            
            # Scale QK and compute exp
            qk = qk * qk_scale - m_ij[:, None]
            p = np.exp(qk)
            
            # Correction factor
            alpha = np.exp(m_i - m_ij)
            
            # Update running sum
            l_ij = np.sum(p, axis=1)
            l_i = l_i * alpha + l_ij
            
            # Update max
            m_i = m_ij
            
            # Rescale accumulator
            acc = acc * alpha[:, None]
            
            # Load V block: [BLOCK_N, HEAD_DIM]
            v = np.where(mask_n[:, None], V[k_indices, :], 0.0)
            
            # Accumulate P @ V
            acc = acc + p @ v
        
        # ----------------------------------------------------------------------
        # Step 5: Normalize and store
        # ----------------------------------------------------------------------
        acc = acc / l_i[:, None]
        
        # Store (with mask)
        for i, idx in enumerate(offs_m):
            if idx < M:
                Out[idx, :] = acc[i, :]
    
    return Out


# ==============================================================================
# PART 6: Flash Attention in Triton - The Simplest Version
# ==============================================================================
"""
Now let's translate to Triton. The key mappings are:

    Python              →    Triton
    ─────────────────────────────────────────────
    for q_start...      →    tl.program_id(0) - each thread block handles one Q block
    np.zeros()          →    tl.zeros()
    Q[start:end]        →    tl.load(Q + offsets)
    Q @ K.T             →    tl.dot(q, k.T)
    np.exp()            →    tl.math.exp2() - base-2 is faster on GPU
    np.maximum()        →    tl.maximum()
    O[start:end] = ...  →    tl.store()
"""


@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    stride_qm, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_om, stride_od,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel - LINE-BY-LINE identical to Python version.
    
    Compare this directly with flash_attention_python() above.
    Every variable name and operation matches exactly.
    """
    # Scale factor (same as Python)
    qk_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
    
    # ==========================================================================
    # OUTER LOOP: Each thread block handles one iteration (parallelized)
    # ==========================================================================
    start_m = tl.program_id(0)  # ← for start_m in range(num_programs) in Python
    
    # --------------------------------------------------------------------------
    # Step 1: Compute offsets (same logic as Python)
    # --------------------------------------------------------------------------
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # ← np.arange(BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)                      # ← np.arange(HEAD_DIM)
    
    # --------------------------------------------------------------------------
    # Step 2: Load Q block
    # --------------------------------------------------------------------------
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # --------------------------------------------------------------------------
    # Step 3: Initialize running statistics
    # --------------------------------------------------------------------------
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # Max scores
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0          # Sum of exp
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)       # Output acc
    
    # --------------------------------------------------------------------------
    # Step 4: Iterate through ALL K, V blocks
    # --------------------------------------------------------------------------
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = tl.arange(0, BLOCK_N)
        
        # Load K block (transposed): [HEAD_DIM, BLOCK_N]
        k_ptrs = K + (start_n + offs_n[None, :]) * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)
        
        # Compute QK^T: [BLOCK_M, D] @ [D, BLOCK_N] = [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k)
        
        # Online softmax: new max
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        
        # Scale QK and compute exp
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp(qk)  # Using exp (same as Python np.exp)
        
        # Correction factor
        alpha = tl.math.exp(m_i - m_ij)
        
        # Update running sum
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        
        # Update max
        m_i = m_ij
        
        # Rescale accumulator
        acc = acc * alpha[:, None]
        
        # Load V block: [BLOCK_N, HEAD_DIM]
        v_ptrs = V + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        
        # Accumulate P @ V
        p = p.to(v.dtype)
        acc = acc + tl.dot(p, v)
    
    # --------------------------------------------------------------------------
    # Step 5: Normalize and store
    # --------------------------------------------------------------------------
    acc = acc / l_i[:, None]
    
    # Store (with mask)
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def flash_attention_triton(Q, K, V):
    """
    Wrapper to call the Triton Flash Attention kernel.
    """
    M, D = Q.shape
    N, _ = K.shape
    
    Out = torch.empty_like(Q)
    
    # Block sizes - these are tuning parameters
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Grid: one thread block per Q block
    grid = (triton.cdiv(M, BLOCK_M),)
    
    flash_attention_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        Out.stride(0), Out.stride(1),
        N,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    
    return Out


# ==============================================================================
# PART 7: Putting It All Together
# ==============================================================================

def test_flash_attention():
    """Test all implementations against each other."""
    
    print("=" * 70)
    print("Flash Attention Tutorial - Testing Implementations")
    print("=" * 70)
    
    # Test dimensions
    M, N, D = 256, 256, 64  # Sequence lengths and head dimension
    
    # Create random Q, K, V
    torch.manual_seed(42)
    Q_torch = torch.randn(M, D, device='cuda', dtype=torch.float32)
    K_torch = torch.randn(N, D, device='cuda', dtype=torch.float32)
    V_torch = torch.randn(N, D, device='cuda', dtype=torch.float32)
    
    Q_np = Q_torch.cpu().numpy()
    K_np = K_torch.cpu().numpy()
    V_np = V_torch.cpu().numpy()
    
    print(f"\nTest setup: M={M}, N={N}, D={D}")
    
    # 1. Naive Python
    print("\n1. Naive Attention (Python)...")
    out_naive = naive_attention(Q_np, K_np, V_np)
    print(f"   Shape: {out_naive.shape}")
    
    # 2. Flash Attention Python
    print("\n2. Flash Attention (Python)...")
    out_flash_py = flash_attention_python(Q_np, K_np, V_np, BLOCK_M=64, BLOCK_N=64)
    diff_flash_py = np.abs(out_naive - out_flash_py).max()
    print(f"   Shape: {out_flash_py.shape}")
    print(f"   Max diff from naive: {diff_flash_py:.2e} {'✓' if diff_flash_py < 1e-5 else '✗'}")
    
    # 3. Flash Attention Triton
    print("\n3. Flash Attention (Triton)...")
    out_triton = flash_attention_triton(Q_torch, K_torch, V_torch)
    diff_triton = (out_triton.cpu().numpy() - out_naive).max()
    print(f"   Shape: {out_triton.shape}")
    print(f"   Max diff from naive: {diff_triton:.2e} {'✓' if abs(diff_triton) < 1e-4 else '✗'}")
    
    # 4. PyTorch reference
    print("\n4. PyTorch scaled_dot_product_attention (reference)...")
    scale = 1.0 / np.sqrt(D)
    Q_4d = Q_torch.unsqueeze(0).unsqueeze(0)  # [1, 1, M, D]
    K_4d = K_torch.unsqueeze(0).unsqueeze(0)
    V_4d = V_torch.unsqueeze(0).unsqueeze(0)
    out_pytorch = torch.nn.functional.scaled_dot_product_attention(Q_4d, K_4d, V_4d)
    out_pytorch = out_pytorch.squeeze(0).squeeze(0)
    diff_pytorch = (out_pytorch.cpu().numpy() - out_naive).max()
    print(f"   Shape: {out_pytorch.shape}")
    print(f"   Max diff from naive: {diff_pytorch:.2e}")
    
    print("\n" + "=" * 70)
    print("Online Softmax Demo")
    print("=" * 70)
    online_softmax_demo()
    
    print("\n" + "=" * 70)
    print("Summary: How Flash Attention Saves Memory")
    print("=" * 70)
    print(f"""
    For M={M}, N={N}, D={D}:
    
    Naive attention:
    - Stores full [M, N] attention matrix: {M * N * 4:,} bytes
    
    Flash attention (BLOCK=64):
    - Stores only [BLOCK, N] at a time: {64 * N * 4:,} bytes
    - Memory savings: {M * N / (64 * N):.0f}x less!
    
    For real workloads (M=N=4096):
    - Naive: 64 MB per head
    - Flash: 1 MB per head
    """)


if __name__ == "__main__":
    test_flash_attention()

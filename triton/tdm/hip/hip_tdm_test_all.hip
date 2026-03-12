/**
 * Comprehensive HIP TDM test — 2D (d2) and 4D (full) load/store variants.
 * All tests use tile == tensor (no OOB) to isolate the store instruction.
 *
 * Tests:
 *   1. 2D TDM load  (d2 intrinsic)  — 8x32 fp16
 *   2. 2D TDM store (d2 intrinsic)  — 8x32 fp16
 *   3. 4D TDM load  (full intrinsic) — 2x4x8x32 fp16
 *   4. 4D TDM store (full intrinsic) — 2x4x8x32 fp16
 *   5. 2D TDM load  (full intrinsic, groups 2/3 zeroed) — 8x32 fp16
 *   6. 2D TDM store (full intrinsic, groups 2/3 zeroed) — 8x32 fp16
 */

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_gfx1250_TDM.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(expr) do {                                      \
    hipError_t _e = (expr);                                       \
    if (_e != hipSuccess) {                                       \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n",          \
                _e, hipGetErrorString(_e), __FILE__, __LINE__);   \
        exit(1);                                                  \
    }                                                             \
} while(0)

using int32x4_t = int __attribute__((ext_vector_type(4)));
using int32x8_t = int __attribute__((ext_vector_type(8)));
using float16_t = _Float16;

// ===== 2D config: 8 rows x 32 cols =====
constexpr int R2 = 8, C2 = 32, N2 = R2 * C2;

// ===== 4D config: 2 x 4 x 8 x 32 =====
constexpr int D0 = 2, D1 = 4, D2 = 8, D3 = 32;
constexpr int N4 = D0 * D1 * D2 * D3;

// ---- 2D descriptor helpers ----

__device__ void build_2d_desc(void* lds, void* global,
                              int32x4_t& g0, int32x8_t& g1) {
    gfx1250_TDM_GROUP0 grp0;
    grp0.m_count = 1;
    grp0.m_is_store = 0;
    grp0.m_is_restore = 0;
    grp0.m_nv = 0;
    grp0.m_scope_trait = 0;
    grp0.m_th = 0;
    grp0.m_gather_index_size = 0;
    grp0.m_gather_mode = 0;
    grp0.ldsAddr(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)));
    grp0.globalAddr(reinterpret_cast<uintptr_t>(global));
    grp0.m_type = 2;

    gfx1250_TDM_GROUP1 grp1;
    grp1.m_workgroup_mask = 0;
    grp1.m_data_size = 1;  // fp16
    grp1.m_atomic_barrier_enable = 0;
    grp1.m_iterate_enable = 0;
    grp1.m_pad_enable = 0;
    grp1.m_early_timeout = 0;
    grp1.m_pad_interval = 0;
    grp1.m_pad_amount = 0;
    grp1.m_atomic_barrier_address = 0;
    grp1.tensorDim0(C2);  // innermost
    grp1.tensorDim1(R2);
    grp1.tileDim0(C2);
    grp1.tileDim1(R2);
    grp1.tileDim2(0);
    grp1.tensorDim0Stride(C2);  // stride in elements
    grp1.tensorDim1Stride(0);

    g0 = __builtin_bit_cast(int32x4_t, grp0.m_bitfield);
    g1 = __builtin_bit_cast(int32x8_t, grp1.m_bitfield);
}

// ---- 4D descriptor helpers ----

__device__ void build_4d_desc(void* lds, void* global,
                              int32x4_t& g0, int32x8_t& g1,
                              int32x4_t& g2, int32x4_t& g3) {
    // Group 0: same as 2D
    gfx1250_TDM_GROUP0 grp0;
    grp0.m_count = 1;
    grp0.m_is_store = 0;
    grp0.m_is_restore = 0;
    grp0.m_nv = 0;
    grp0.m_scope_trait = 0;
    grp0.m_th = 0;
    grp0.m_gather_index_size = 0;
    grp0.m_gather_mode = 0;
    grp0.ldsAddr(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)));
    grp0.globalAddr(reinterpret_cast<uintptr_t>(global));
    grp0.m_type = 2;

    // Group 1: dims 0,1 (innermost two) + tile dims 0,1,2
    // HW dim ordering: dim0=innermost(D3), dim1=D2, dim2=D1, dim3=D0
    gfx1250_TDM_GROUP1 grp1;
    grp1.m_workgroup_mask = 0;
    grp1.m_data_size = 1;  // fp16
    grp1.m_atomic_barrier_enable = 0;
    grp1.m_iterate_enable = 0;
    grp1.m_pad_enable = 0;
    grp1.m_early_timeout = 0;
    grp1.m_pad_interval = 0;
    grp1.m_pad_amount = 0;
    grp1.m_atomic_barrier_address = 0;
    grp1.tensorDim0(D3);             // innermost tensor dim
    grp1.tensorDim1(D2);             // second dim
    grp1.tileDim0(D3);               // tile matches tensor
    grp1.tileDim1(D2);
    grp1.tileDim2(D1);
    grp1.tensorDim0Stride(D3);       // stride to next D2 row (in elements)
    grp1.tensorDim1Stride(D3 * D2);  // stride to next D1 slice

    // Group 2: tensor_dim2 (D1), tensor_dim3 (D0), tensor_dim2_stride, tile_dim3
    gfx1250_TDM_GROUP2 grp2;
    grp2.m_tensor_dim2 = D1;
    grp2.m_tensor_dim3 = D0;
    grp2.tensorDim2Stride(static_cast<uint64_t>(D3) * D2 * D1);  // stride to next D0 slice
    grp2.m_tile_dim3 = D0;

    // Group 3: tensor_dim3_stride (not needed for outermost), tensor_dim4, tile_dim4
    gfx1250_TDM_GROUP3 grp3;
    grp3.tensorDim3Stride(0);
    grp3.tensorDim4(0);
    grp3.m_sgpr2 = 0;
    grp3.m_sgpr3_reserved = 0;

    g0 = __builtin_bit_cast(int32x4_t, grp0.m_bitfield);
    g1 = __builtin_bit_cast(int32x8_t, grp1.m_bitfield);
    g2 = __builtin_bit_cast(int32x4_t, grp2.m_bitfield);
    g3 = __builtin_bit_cast(int32x4_t, grp3.m_bitfield);
}

// ===== Test 1: 2D load (d2 intrinsic) =====
__global__ void kernel_2d_load_d2(float16_t* src, float16_t* dst) {
    __shared__ float16_t lds[N2];
    int32x4_t g0; int32x8_t g1;
    build_2d_desc(lds, src, g0, g1);
    __builtin_amdgcn_tensor_load_to_lds_d2(g0, g1, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    for (int i = threadIdx.x; i < N2; i += blockDim.x) dst[i] = lds[i];
}

// ===== Test 2: 2D store (d2 intrinsic) =====
__global__ void kernel_2d_store_d2(float16_t* dst) {
    __shared__ float16_t lds[N2];
    for (int i = threadIdx.x; i < N2; i += blockDim.x)
        lds[i] = static_cast<float16_t>(i);
    __syncthreads();
    int32x4_t g0; int32x8_t g1;
    build_2d_desc(lds, dst, g0, g1);
    __builtin_amdgcn_tensor_store_from_lds_d2(g0, g1, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
}

// ===== Test 3: 4D load (full intrinsic) =====
__global__ void kernel_4d_load_full(float16_t* src, float16_t* dst) {
    __shared__ float16_t lds[N4];
    int32x4_t g0, g2, g3; int32x8_t g1;
    build_4d_desc(lds, src, g0, g1, g2, g3);
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    for (int i = threadIdx.x; i < N4; i += blockDim.x) dst[i] = lds[i];
}

// ===== Test 4: 4D store (full intrinsic) =====
__global__ void kernel_4d_store_full(float16_t* dst) {
    __shared__ float16_t lds[N4];
    for (int i = threadIdx.x; i < N4; i += blockDim.x)
        lds[i] = static_cast<float16_t>(i % 256);
    __syncthreads();
    int32x4_t g0, g2, g3; int32x8_t g1;
    build_4d_desc(lds, dst, g0, g1, g2, g3);
    __builtin_amdgcn_tensor_store_from_lds(g0, g1, g2, g3, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
}

// ===== Test 5: 2D load via full intrinsic (groups 2/3 zeroed) =====
__global__ void kernel_2d_load_full(float16_t* src, float16_t* dst) {
    __shared__ float16_t lds[N2];
    int32x4_t g0; int32x8_t g1;
    build_2d_desc(lds, src, g0, g1);
    int32x4_t g2 = {0, 0, 0, 0};
    int32x4_t g3 = {0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    for (int i = threadIdx.x; i < N2; i += blockDim.x) dst[i] = lds[i];
}

// ===== Test 6: 2D store via full intrinsic (groups 2/3 zeroed) =====
__global__ void kernel_2d_store_full(float16_t* dst) {
    __shared__ float16_t lds[N2];
    for (int i = threadIdx.x; i < N2; i += blockDim.x)
        lds[i] = static_cast<float16_t>(i);
    __syncthreads();
    int32x4_t g0; int32x8_t g1;
    build_2d_desc(lds, dst, g0, g1);
    int32x4_t g2 = {0, 0, 0, 0};
    int32x4_t g3 = {0, 0, 0, 0};
    __builtin_amdgcn_tensor_store_from_lds(g0, g1, g2, g3, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
}

bool verify(const float16_t* expected, const float16_t* actual, int n, const char* label) {
    int errs = 0;
    for (int i = 0; i < n; i++) {
        if (static_cast<float>(expected[i]) != static_cast<float>(actual[i])) {
            if (errs < 3)
                fprintf(stderr, "  [%s] mismatch at %d: expected %.1f got %.1f\n",
                        label, i, (float)expected[i], (float)actual[i]);
            errs++;
        }
    }
    if (errs == 0) printf("  [%s] PASS\n", label);
    else printf("  [%s] FAIL (%d/%d mismatches)\n", label, errs, n);
    return errs == 0;
}

struct TestResult { const char* name; bool ran; bool passed; };

int main(int argc, char** argv) {
    int test_filter = 0;  // 0 = all, N = run only test N
    if (argc > 1) test_filter = atoi(argv[1]);

    printf("HIP TDM comprehensive test (2D & 4D, load & store)\n");
    printf("  2D: %dx%d fp16 (%d elems)\n", R2, C2, N2);
    printf("  4D: %dx%dx%dx%d fp16 (%d elems)\n", D0, D1, D2, D3, N4);
    if (test_filter) printf("  Running only test %d\n", test_filter);
    printf("\n");

    int maxN = (N4 > N2) ? N4 : N2;
    float16_t* h_src = new float16_t[maxN];
    float16_t* h_dst = new float16_t[maxN];
    float16_t* h_exp = new float16_t[maxN];

    float16_t *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_src, maxN * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&d_dst, maxN * sizeof(float16_t)));

    TestResult results[6] = {};
    int ok = 0, total = 0;

    auto run_test = [&](int idx, const char* name, auto fn) {
        if (test_filter && test_filter != idx) return;
        results[idx-1] = {name, true, false};
        total++;
        printf("Test %d: %s\n", idx, name);
        fflush(stdout);
        fn();
        fflush(stdout);
    };

    // --- Test 1: 2D load (d2) ---
    run_test(1, "2D load (d2 intrinsic)", [&]{
        for (int i = 0; i < N2; i++) h_src[i] = static_cast<float16_t>(i + 1);
        HIP_CHECK(hipMemcpy(d_src, h_src, N2 * sizeof(float16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_dst, 0, N2 * sizeof(float16_t)));
        kernel_2d_load_d2<<<1, 32>>>(d_src, d_dst);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst, d_dst, N2 * sizeof(float16_t), hipMemcpyDeviceToHost));
        bool p = verify(h_src, h_dst, N2, "2d_load_d2");
        results[0].passed = p; ok += p;
    });

    // --- Test 2: 2D store (d2) ---
    run_test(2, "2D store (d2 intrinsic)", [&]{
        for (int i = 0; i < N2; i++) h_exp[i] = static_cast<float16_t>(i);
        HIP_CHECK(hipMemset(d_dst, 0, N2 * sizeof(float16_t)));
        kernel_2d_store_d2<<<1, 32>>>(d_dst);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst, d_dst, N2 * sizeof(float16_t), hipMemcpyDeviceToHost));
        bool p = verify(h_exp, h_dst, N2, "2d_store_d2");
        results[1].passed = p; ok += p;
    });

    // --- Test 3: 4D load (full) ---
    run_test(3, "4D load (full intrinsic)", [&]{
        for (int i = 0; i < N4; i++) h_src[i] = static_cast<float16_t>(i % 256);
        HIP_CHECK(hipMemcpy(d_src, h_src, N4 * sizeof(float16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_dst, 0, N4 * sizeof(float16_t)));
        kernel_4d_load_full<<<1, 32>>>(d_src, d_dst);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst, d_dst, N4 * sizeof(float16_t), hipMemcpyDeviceToHost));
        bool p = verify(h_src, h_dst, N4, "4d_load_full");
        results[2].passed = p; ok += p;
    });

    // --- Test 4: 4D store (full) ---
    run_test(4, "4D store (full intrinsic)", [&]{
        for (int i = 0; i < N4; i++) h_exp[i] = static_cast<float16_t>(i % 256);
        HIP_CHECK(hipMemset(d_dst, 0, N4 * sizeof(float16_t)));
        kernel_4d_store_full<<<1, 32>>>(d_dst);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst, d_dst, N4 * sizeof(float16_t), hipMemcpyDeviceToHost));
        bool p = verify(h_exp, h_dst, N4, "4d_store_full");
        results[3].passed = p; ok += p;
    });

    // --- Test 5: 2D load via full intrinsic ---
    run_test(5, "2D load (full intrinsic, grp2/3 zeroed)", [&]{
        for (int i = 0; i < N2; i++) h_src[i] = static_cast<float16_t>(i + 1);
        HIP_CHECK(hipMemcpy(d_src, h_src, N2 * sizeof(float16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_dst, 0, N2 * sizeof(float16_t)));
        kernel_2d_load_full<<<1, 32>>>(d_src, d_dst);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst, d_dst, N2 * sizeof(float16_t), hipMemcpyDeviceToHost));
        bool p = verify(h_src, h_dst, N2, "2d_load_full");
        results[4].passed = p; ok += p;
    });

    // --- Test 6: 2D store via full intrinsic ---
    run_test(6, "2D store (full intrinsic, grp2/3 zeroed)", [&]{
        for (int i = 0; i < N2; i++) h_exp[i] = static_cast<float16_t>(i);
        HIP_CHECK(hipMemset(d_dst, 0, N2 * sizeof(float16_t)));
        kernel_2d_store_full<<<1, 32>>>(d_dst);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst, d_dst, N2 * sizeof(float16_t), hipMemcpyDeviceToHost));
        bool p = verify(h_exp, h_dst, N2, "2d_store_full");
        results[5].passed = p; ok += p;
    });

    printf("\n=== Summary ===\n");
    for (int i = 0; i < 6; i++) {
        if (results[i].ran)
            printf("  Test %d (%s): %s\n", i+1, results[i].name,
                   results[i].passed ? "PASS" : "FAIL/CRASH");
    }
    printf("\nResult: %d / %d passed\n", ok, total);

    hipFree(d_src);
    hipFree(d_dst);
    delete[] h_src;
    delete[] h_dst;
    delete[] h_exp;
    return (ok == total) ? 0 : 1;
}

/**
 * HIP TDM 1D test — 1D load/store via TDM on gfx1250.
 *
 * 1D is represented as a 2D descriptor with dim1=1 (single row of N fp16).
 *
 * Tests:
 *   1. 1D load  — d2 intrinsic        (128 fp16)
 *   2. 1D load  — full intrinsic       (128 fp16, grp2/3 zeroed)
 *   3. 1D store — d2 intrinsic         (128 fp16)
 *   4. 1D store — full intrinsic       (128 fp16, grp2/3 zeroed)
 *   5. 1D store — d2 intrinsic, tiny   (32 fp16)
 *
 * Build:
 *   /opt/rocm/bin/hipcc -x hip --offload-arch=gfx1250 -o hip_tdm_1d hip_tdm_1d.hip
 *
 * Run all tests:
 *   source /am-ffm/am_env.sh && ./hip_tdm_1d
 *
 * Run single test (e.g. test 1):
 *   source /am-ffm/am_env.sh && ./hip_tdm_1d 1
 *
 * Expected results (AM v3.0.1):
 *   Tests 1-2 (loads)  — PASS
 *   Tests 3-5 (stores) — crash (assertion am_tc_token.h:372, CreateTcpQuadToken)
 *
 * Expected results (FFM):
 *   source /am-ffm/ffmlite_env.sh && ./hip_tdm_1d
 *   Tests 1-5 — all PASS
 */

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_gfx1250_TDM.h>
#include <cstdio>
#include <cstdlib>

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

__device__ void build_1d_desc(void* lds, void* global, int n,
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
    grp1.tensorDim0(n);
    grp1.tensorDim1(1);
    grp1.tileDim0(n);
    grp1.tileDim1(1);
    grp1.tileDim2(0);
    grp1.tensorDim0Stride(n);
    grp1.tensorDim1Stride(0);

    g0 = __builtin_bit_cast(int32x4_t, grp0.m_bitfield);
    g1 = __builtin_bit_cast(int32x8_t, grp1.m_bitfield);
}

constexpr int N = 128;

// Test 1: 1D load (d2 intrinsic)
__global__ void kernel_1d_load(float16_t* src, float16_t* dst) {
    __shared__ float16_t lds[N];
    int32x4_t g0; int32x8_t g1;
    build_1d_desc(lds, src, N, g0, g1);
    __builtin_amdgcn_tensor_load_to_lds_d2(g0, g1, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    for (int i = threadIdx.x; i < N; i += blockDim.x) dst[i] = lds[i];
}

// Test 2: 1D load (full intrinsic, grp2/3 zeroed)
__global__ void kernel_1d_load_full(float16_t* src, float16_t* dst) {
    __shared__ float16_t lds[N];
    int32x4_t g0; int32x8_t g1;
    build_1d_desc(lds, src, N, g0, g1);
    int32x4_t g2 = {0, 0, 0, 0};
    int32x4_t g3 = {0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    for (int i = threadIdx.x; i < N; i += blockDim.x) dst[i] = lds[i];
}

// Test 3: 1D store (d2 intrinsic)
__global__ void kernel_1d_store(float16_t* dst) {
    __shared__ float16_t lds[N];
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        lds[i] = static_cast<float16_t>(i);
    __syncthreads();
    int32x4_t g0; int32x8_t g1;
    build_1d_desc(lds, dst, N, g0, g1);
    __builtin_amdgcn_tensor_store_from_lds_d2(g0, g1, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
}

// Test 4: 1D store (full intrinsic, grp2/3 zeroed)
__global__ void kernel_1d_store_full(float16_t* dst) {
    __shared__ float16_t lds[N];
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        lds[i] = static_cast<float16_t>(i);
    __syncthreads();
    int32x4_t g0; int32x8_t g1;
    build_1d_desc(lds, dst, N, g0, g1);
    int32x4_t g2 = {0, 0, 0, 0};
    int32x4_t g3 = {0, 0, 0, 0};
    __builtin_amdgcn_tensor_store_from_lds(g0, g1, g2, g3, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
}

// Test 5: 1D store tiny (32 elements)
__global__ void kernel_1d_store_tiny(float16_t* dst) {
    __shared__ float16_t lds[32];
    for (int i = threadIdx.x; i < 32; i += blockDim.x)
        lds[i] = static_cast<float16_t>(i);
    __syncthreads();
    int32x4_t g0; int32x8_t g1;
    build_1d_desc(lds, dst, 32, g0, g1);
    __builtin_amdgcn_tensor_store_from_lds_d2(g0, g1, 0);
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

int main(int argc, char** argv) {
    int test_id = argc > 1 ? atoi(argv[1]) : 0;

    printf("HIP TDM 1D test (%d fp16 elements)\n", N);
    if (test_id) printf("Running only test %d\n", test_id);
    printf("\n");

    float16_t h_src[N], h_dst[N], h_exp[N];
    for (int i = 0; i < N; i++) { h_src[i] = (float16_t)(i+1); h_exp[i] = (float16_t)i; }

    float16_t *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_src, N * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&d_dst, N * sizeof(float16_t)));

    int ok = 0, total = 0;
    auto run = [&](int id, const char* name, auto fn) {
        if (test_id && test_id != id) return;
        total++;
        printf("Test %d: %s\n", id, name);
        fflush(stdout);
        ok += fn();
        fflush(stdout);
    };

    run(1, "1D load (d2)", [&]{
        HIP_CHECK(hipMemcpy(d_src, h_src, N*2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_dst, 0, N*2));
        kernel_1d_load<<<1, 32>>>(d_src, d_dst);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst, d_dst, N*2, hipMemcpyDeviceToHost));
        return verify(h_src, h_dst, N, "1d_load");
    });

    run(2, "1D load (full intrinsic, grp2/3=0)", [&]{
        HIP_CHECK(hipMemcpy(d_src, h_src, N*2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_dst, 0, N*2));
        kernel_1d_load_full<<<1, 32>>>(d_src, d_dst);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst, d_dst, N*2, hipMemcpyDeviceToHost));
        return verify(h_src, h_dst, N, "1d_load_full");
    });

    run(3, "1D store (d2)", [&]{
        HIP_CHECK(hipMemset(d_dst, 0, N*2));
        kernel_1d_store<<<1, 32>>>(d_dst);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst, d_dst, N*2, hipMemcpyDeviceToHost));
        return verify(h_exp, h_dst, N, "1d_store");
    });

    run(4, "1D store (full intrinsic, grp2/3=0)", [&]{
        HIP_CHECK(hipMemset(d_dst, 0, N*2));
        kernel_1d_store_full<<<1, 32>>>(d_dst);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst, d_dst, N*2, hipMemcpyDeviceToHost));
        return verify(h_exp, h_dst, N, "1d_store_full");
    });

    run(5, "1D store tiny (32 elems)", [&]{
        float16_t h_exp32[32], h_dst32[32];
        for (int i = 0; i < 32; i++) h_exp32[i] = (float16_t)i;
        HIP_CHECK(hipMemset(d_dst, 0, 32*2));
        kernel_1d_store_tiny<<<1, 32>>>(d_dst);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_dst32, d_dst, 32*2, hipMemcpyDeviceToHost));
        return verify(h_exp32, h_dst32, 32, "1d_store_tiny");
    });

    printf("\nResult: %d / %d passed\n", ok, total);

    hipFree(d_src); hipFree(d_dst);
    return (ok == total) ? 0 : 1;
}

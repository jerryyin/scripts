/**
 * Debug 4D TDM load — test smaller tile sizes to find the boundary.
 * Also test if the issue is per-warp tile size limits.
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

__device__ void build_4d_load_desc(void* lds, void* global,
    int d0, int d1, int d2, int d3,
    int32x4_t& g0, int32x8_t& g1, int32x4_t& g2, int32x4_t& g3) {

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
    // HW dim ordering: dim0=d3(innermost), dim1=d2, dim2=d1, dim3=d0
    grp1.tensorDim0(d3);
    grp1.tensorDim1(d2);
    grp1.tileDim0(d3);
    grp1.tileDim1(d2);
    grp1.tileDim2(d1);
    grp1.tensorDim0Stride(d3);
    grp1.tensorDim1Stride(d3 * d2);

    gfx1250_TDM_GROUP2 grp2;
    grp2.m_tensor_dim2 = d1;
    grp2.m_tensor_dim3 = d0;
    grp2.tensorDim2Stride(static_cast<uint64_t>(d3) * d2 * d1);
    grp2.m_tile_dim3 = d0;

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

// Template kernel for different 4D sizes
template<int D0, int D1, int D2, int D3>
__global__ void kernel_4d_load(float16_t* src, float16_t* dst) {
    constexpr int N = D0 * D1 * D2 * D3;
    __shared__ float16_t lds[N];
    int32x4_t g0, g2, g3; int32x8_t g1;
    build_4d_load_desc(lds, src, D0, D1, D2, D3, g0, g1, g2, g3);
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    for (int i = threadIdx.x; i < N; i += blockDim.x) dst[i] = lds[i];
}

bool verify(const float16_t* expected, const float16_t* actual, int n, const char* label) {
    int errs = 0;
    int first_bad = -1;
    for (int i = 0; i < n; i++) {
        if (static_cast<float>(expected[i]) != static_cast<float>(actual[i])) {
            if (first_bad < 0) first_bad = i;
            if (errs < 3)
                fprintf(stderr, "  [%s] mismatch at %d: expected %.1f got %.1f\n",
                        label, i, (float)expected[i], (float)actual[i]);
            errs++;
        }
    }
    if (errs == 0) printf("  [%s] PASS\n", label);
    else printf("  [%s] FAIL (%d/%d mismatches, first at %d)\n", label, errs, n, first_bad);
    return errs == 0;
}

template<int D0, int D1, int D2, int D3>
bool run_4d_test(float16_t* d_src, float16_t* d_dst, float16_t* h_src,
                 float16_t* h_dst, const char* label) {
    constexpr int N = D0 * D1 * D2 * D3;
    for (int i = 0; i < N; i++) h_src[i] = static_cast<float16_t>(i % 256);
    HIP_CHECK(hipMemcpy(d_src, h_src, N * sizeof(float16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_dst, 0, N * sizeof(float16_t)));
    kernel_4d_load<D0, D1, D2, D3><<<1, 32>>>(d_src, d_dst);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_dst, d_dst, N * sizeof(float16_t), hipMemcpyDeviceToHost));
    return verify(h_src, h_dst, N, label);
}

int main() {
    printf("4D TDM load debug — varying tile sizes\n\n");

    constexpr int maxN = 4096;
    float16_t *h_src = new float16_t[maxN];
    float16_t *h_dst = new float16_t[maxN];
    float16_t *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_src, maxN * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&d_dst, maxN * sizeof(float16_t)));

    int ok = 0, total = 0;

    printf("Test A: 1x1x8x32 (256 elems, same as 2D)\n");
    total++; ok += run_4d_test<1,1,8,32>(d_src, d_dst, h_src, h_dst, "1x1x8x32");

    printf("Test B: 1x1x4x32 (128 elems)\n");
    total++; ok += run_4d_test<1,1,4,32>(d_src, d_dst, h_src, h_dst, "1x1x4x32");

    printf("Test C: 1x2x8x32 (512 elems)\n");
    total++; ok += run_4d_test<1,2,8,32>(d_src, d_dst, h_src, h_dst, "1x2x8x32");

    printf("Test D: 1x4x8x32 (1024 elems)\n");
    total++; ok += run_4d_test<1,4,8,32>(d_src, d_dst, h_src, h_dst, "1x4x8x32");

    printf("Test E: 2x1x8x32 (512 elems)\n");
    total++; ok += run_4d_test<2,1,8,32>(d_src, d_dst, h_src, h_dst, "2x1x8x32");

    printf("Test F: 2x2x8x32 (1024 elems)\n");
    total++; ok += run_4d_test<2,2,8,32>(d_src, d_dst, h_src, h_dst, "2x2x8x32");

    printf("Test G: 2x4x8x32 (2048 elems, original failing)\n");
    total++; ok += run_4d_test<2,4,8,32>(d_src, d_dst, h_src, h_dst, "2x4x8x32");

    printf("\nResult: %d / %d passed\n", ok, total);

    hipFree(d_src); hipFree(d_dst);
    delete[] h_src; delete[] h_dst;
    return (ok == total) ? 0 : 1;
}

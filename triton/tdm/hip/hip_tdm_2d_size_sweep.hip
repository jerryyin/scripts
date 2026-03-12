/**
 * 2D TDM load size sweep — find size limits on AM.
 *
 * All tests use d2 intrinsic, tile == tensor (no OOB), fp16.
 * Each test is selectable by ID to avoid cascading hangs.
 *
 * Tests:
 *   1.  8x32   =   256 elems (  512B)
 *   2.  8x64   =   512 elems (   1KB)
 *   3.  8x128  =  1024 elems (   2KB)
 *   4.  8x256  =  2048 elems (   4KB)
 *   5.  4x512  =  2048 elems (   4KB, very wide)
 *   6. 16x128  =  2048 elems (   4KB)
 *   7. 32x64   =  2048 elems (   4KB, many rows)
 *   8. 64x32   =  2048 elems (   4KB, lots of rows)
 *   9. 24x128  =  3072 elems (   6KB)
 *  10. 16x256  =  4096 elems (   8KB)
 *  11. 32x128  =  4096 elems (   8KB)
 *
 * Build:
 *   /opt/rocm/bin/hipcc -x hip --offload-arch=gfx1250 -o hip_tdm_2d_size_sweep hip_tdm_2d_size_sweep.hip
 *
 * Run all tests:
 *   source /am-ffm/am_env.sh && ./hip_tdm_2d_size_sweep
 *
 * Run single test (e.g. test 3):
 *   source /am-ffm/am_env.sh && ./hip_tdm_2d_size_sweep 3
 *
 * Expected results (AM v3.0.1):
 *   All tests PASS (2D loads work up to at least 8KB)
 *
 * Expected results (FFM):
 *   source /am-ffm/ffmlite_env.sh && ./hip_tdm_2d_size_sweep
 *   All tests PASS
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

__device__ void build_2d_desc(void* lds, void* global, int rows, int cols,
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
    grp1.tensorDim0(cols);
    grp1.tensorDim1(rows);
    grp1.tileDim0(cols);
    grp1.tileDim1(rows);
    grp1.tileDim2(0);
    grp1.tensorDim0Stride(cols);
    grp1.tensorDim1Stride(0);

    g0 = __builtin_bit_cast(int32x4_t, grp0.m_bitfield);
    g1 = __builtin_bit_cast(int32x8_t, grp1.m_bitfield);
}

template<int ROWS, int COLS>
__global__ void kernel_2d_load(float16_t* src, float16_t* dst) {
    constexpr int N = ROWS * COLS;
    __shared__ float16_t lds[N];
    int32x4_t g0; int32x8_t g1;
    build_2d_desc(lds, src, ROWS, COLS, g0, g1);
    __builtin_amdgcn_tensor_load_to_lds_d2(g0, g1, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    for (int i = threadIdx.x; i < N; i += blockDim.x) dst[i] = lds[i];
}

bool verify(const float16_t* expected, const float16_t* actual, int n, const char* label) {
    int errs = 0, first_bad = -1;
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
    else printf("  [%s] FAIL (%d/%d, first at %d)\n", label, errs, n, first_bad);
    return errs == 0;
}

template<int ROWS, int COLS>
bool run_test(float16_t* d_src, float16_t* d_dst, float16_t* h_src,
              float16_t* h_dst, const char* label) {
    constexpr int N = ROWS * COLS;
    for (int i = 0; i < N; i++) h_src[i] = static_cast<float16_t>(i % 256);
    HIP_CHECK(hipMemcpy(d_src, h_src, N * sizeof(float16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_dst, 0, N * sizeof(float16_t)));
    kernel_2d_load<ROWS, COLS><<<1, 32>>>(d_src, d_dst);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_dst, d_dst, N * sizeof(float16_t), hipMemcpyDeviceToHost));
    return verify(h_src, h_dst, N, label);
}

int main(int argc, char** argv) {
    int test_id = argc > 1 ? atoi(argv[1]) : 0;

    printf("2D TDM load — size sweep\n");
    if (test_id) printf("Running only test %d\n", test_id);
    printf("\n");

    constexpr int maxN = 8192;
    float16_t *h_src = new float16_t[maxN];
    float16_t *h_dst = new float16_t[maxN];
    float16_t *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_src, maxN * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&d_dst, maxN * sizeof(float16_t)));

    int ok = 0, total = 0;
    auto run = [&](int id, const char* name, auto fn) {
        if (test_id && test_id != id) return;
        total++;
        printf("Test %d: %s\n", id, name);
        fflush(stdout);
        ok += fn();
        fflush(stdout);
    };

    run( 1, " 8x32   =  256 elems (512B)",        [&]{ return run_test< 8, 32>(d_src, d_dst, h_src, h_dst,  "8x32"); });
    run( 2, " 8x64   =  512 elems (1KB)",         [&]{ return run_test< 8, 64>(d_src, d_dst, h_src, h_dst,  "8x64"); });
    run( 3, " 8x128  = 1024 elems (2KB)",         [&]{ return run_test< 8,128>(d_src, d_dst, h_src, h_dst, "8x128"); });
    run( 4, " 8x256  = 2048 elems (4KB)",         [&]{ return run_test< 8,256>(d_src, d_dst, h_src, h_dst, "8x256"); });
    run( 5, " 4x512  = 2048 elems (4KB, wide)",   [&]{ return run_test< 4,512>(d_src, d_dst, h_src, h_dst, "4x512"); });
    run( 6, "16x128  = 2048 elems (4KB)",          [&]{ return run_test<16,128>(d_src, d_dst, h_src, h_dst,"16x128"); });
    run( 7, "32x64   = 2048 elems (4KB, many rows)",[&]{ return run_test<32, 64>(d_src, d_dst, h_src, h_dst,"32x64"); });
    run( 8, "64x32   = 2048 elems (4KB, tall)",    [&]{ return run_test<64, 32>(d_src, d_dst, h_src, h_dst,"64x32"); });
    run( 9, "24x128  = 3072 elems (6KB)",          [&]{ return run_test<24,128>(d_src, d_dst, h_src, h_dst,"24x128"); });
    run(10, "16x256  = 4096 elems (8KB)",          [&]{ return run_test<16,256>(d_src, d_dst, h_src, h_dst,"16x256"); });
    run(11, "32x128  = 4096 elems (8KB)",          [&]{ return run_test<32,128>(d_src, d_dst, h_src, h_dst,"32x128"); });

    printf("\nResult: %d / %d passed\n", ok, total);

    hipFree(d_src); hipFree(d_dst);
    delete[] h_src; delete[] h_dst;
    return (ok == total) ? 0 : 1;
}

// RUN: /opt/rocm/bin/hipcc -x hip --offload-arch=gfx1250 %s -o ds_load_tr8_demo && ./ds_load_tr8_demo

/**
 * ds_load_tr8_b64 Demo (gfx1250)
 * ===============================
 *
 * Demonstrates the ds_load_tr8_b64 transposed LDS load instruction for
 * 8-bit data (fp8/i8).  Each lane reads 64 bits (8 x i8) from LDS, and
 * the hardware transposes across 8-lane shuffle groups.
 *
 * Tile: 16x16 uint8 elements stored contiguously in LDS.
 *
 * Pre-transpose LDS access (same quadrant layout as ds_load_tr16_b128):
 *   Group 0 (lanes 0-7):   rows 0-7,   cols 0-7
 *   Group 1 (lanes 8-15):  rows 0-7,   cols 8-15
 *   Group 2 (lanes 16-23): rows 8-15,  cols 0-7
 *   Group 3 (lanes 24-31): rows 8-15,  cols 8-15
 *
 * Shuffle groups for the transpose are interleaved (NOT consecutive):
 *   Shuffle A: lanes {0,1,2,3, 8,9,10,11}
 *   Shuffle B: lanes {4,5,6,7, 12,13,14,15}
 *   Shuffle C: lanes {16,17,18,19, 24,25,26,27}
 *   Shuffle D: lanes {20,21,22,23, 28,29,30,31}
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define ROWS 16
#define COLS 16

typedef uint8_t u8_t;
typedef __attribute__((ext_vector_type(2))) unsigned int u32x2_t;

__device__ __forceinline__ u32x2_t ds_load_tr8_b64(unsigned int lds_offset) {
    u32x2_t result;
    asm volatile(
        "ds_load_tr8_b64 %0, %1"
        : "=v"(result)
        : "v"(lds_offset)
        : "memory"
    );
    return result;
}

__device__ __forceinline__ unsigned int lds_ptr_to_offset(const u8_t* ptr) {
    return static_cast<unsigned int>(reinterpret_cast<size_t>(ptr) & 0xFFFFFFFF);
}

__global__ void transpose_demo(const u8_t* __restrict__ input,
                               u8_t* __restrict__ output_tr) {
    __shared__ u8_t lds[ROWS][COLS];

    const int lane_id = threadIdx.x % 32;

    const int quadrant = lane_id / 8;
    const int lane_in_quad = lane_id % 8;
    const int row_offset = (quadrant / 2) * 8;
    const int col_offset = (quadrant % 2) * 8;
    const int row = row_offset + lane_in_quad;

    for (int c = 0; c < 8; c++) {
        lds[row][col_offset + c] = input[row * COLS + col_offset + c];
    }
    __syncthreads();

    unsigned int lds_offset = lds_ptr_to_offset(&lds[row][col_offset]);
    u32x2_t raw = ds_load_tr8_b64(lds_offset);

    u8_t transposed[8];
    __builtin_memcpy(transposed, &raw, 8);

    for (int i = 0; i < 8; i++) {
        output_tr[lane_id * 8 + i] = transposed[i];
    }
}

int main() {
    const int num_elements = ROWS * COLS;
    const int output_size = 32 * 8;

    u8_t *h_input = new u8_t[num_elements];
    u8_t *h_output_tr = new u8_t[output_size];

    printf("ds_load_tr8_b64 Demo\n");
    printf("Tile: %dx%d uint8, stride=%d bytes\n\n", ROWS, COLS, COLS);

    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            h_input[r * COLS + c] = (u8_t)(r * COLS + c + 1);

    u8_t *d_input, *d_output_tr;
    (void)hipMalloc(&d_input, num_elements * sizeof(u8_t));
    (void)hipMalloc(&d_output_tr, output_size * sizeof(u8_t));

    (void)hipMemcpy(d_input, h_input, num_elements * sizeof(u8_t), hipMemcpyHostToDevice);

    transpose_demo<<<1, 32>>>(d_input, d_output_tr);

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("Kernel failed: %s\n", hipGetErrorString(err));
        delete[] h_input;
        delete[] h_output_tr;
        (void)hipFree(d_input);
        (void)hipFree(d_output_tr);
        return 1;
    }

    (void)hipMemcpy(h_output_tr, d_output_tr, output_size * sizeof(u8_t), hipMemcpyDeviceToHost);

    printf("All lane outputs:\n\n");
    for (int lane = 0; lane < 32; lane++) {
        if (lane % 8 == 0) {
            const char* group_names[] = {
                "Group 0 (lanes 0-7, read rows 0-7, cols 0-7)",
                "Group 1 (lanes 8-15, read rows 0-7, cols 8-15)",
                "Group 2 (lanes 16-23, read rows 8-15, cols 0-7)",
                "Group 3 (lanes 24-31, read rows 8-15, cols 8-15)"
            };
            printf("  %s:\n", group_names[lane / 8]);
        }
        printf("    Lane %2d: {", lane);
        for (int i = 0; i < 8; i++)
            printf("%3d%s", h_output_tr[lane * 8 + i], i < 7 ? ", " : "");
        printf("}\n");
        if (lane % 8 == 7) printf("\n");
    }

    // Verify against the observed hardware transpose pattern.
    //
    // Within each 16-lane half-wave, the transpose mixes data between two
    // 8-lane groups:
    //
    //   Post[L][E] = Pre[src_lane][L % 8]
    //   where src_lane = (L/8 % 2)*4 + (E/4)*8 + (E%4)
    //         (relative to the half-wave base L/16 * 16)
    //
    // This means lanes {0-3, 8-11} form one "super-group" and
    // lanes {4-7, 12-15} form another, but the LDS READ addresses
    // are still determined by our kernel's quadrant mapping.
    int errors = 0;
    for (int lane = 0; lane < 32; lane++) {
        int half_base = (lane / 16) * 16;
        int local = lane % 16;
        int h = local / 8;
        int l = local % 8;

        for (int elem = 0; elem < 8; elem++) {
            int e_half = elem / 4;
            int e_lo = elem % 4;
            int src_local = h * 4 + e_half * 8 + e_lo;
            int src_lane = half_base + src_local;

            int src_q = src_lane / 8;
            int src_row = (src_q / 2) * 8 + (src_lane % 8);
            int src_col = (src_q % 2) * 8 + l;

            u8_t expected = (u8_t)(src_row * COLS + src_col + 1);
            u8_t got = h_output_tr[lane * 8 + elem];
            if (got != expected) {
                if (errors < 8)
                    printf("MISMATCH lane %d elem %d: got %d expected %d "
                           "(src row %d col %d)\n",
                           lane, elem, got, expected, src_row, src_col);
                errors++;
            }
        }
    }
    printf("Correctness: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);

    delete[] h_input;
    delete[] h_output_tr;
    (void)hipFree(d_input);
    (void)hipFree(d_output_tr);

    return errors > 0 ? 1 : 0;
}

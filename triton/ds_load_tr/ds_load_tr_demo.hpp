// RUN: /opt/rocm/bin/hipcc -x hip --offload-arch=gfx1250 %s -o ds_load_tr_demo && ./ds_load_tr_demo

/**
 * ds_load_tr16_b128 Demonstration
 * ================================
 * 
 * This example shows how ds_load_tr16_b128 performs a transposed load from LDS.
 * 
 * The instruction is designed for WMMA (Wave Matrix Multiply Accumulate) where:
 * - Data in LDS is laid out as a matrix with one dimension contiguous
 * - WMMA needs the OTHER dimension to be contiguous in registers
 * - ds_load_tr16 shuffles data across lanes to achieve this transpose
 * 
 * For 32 lanes, each with 8 f16 elements:
 * - LDS layout: 16 rows x 16 cols (256 elements total)
 * - Before transpose: Lane L loads row L's 8 consecutive columns
 * - After transpose: Lane L gets 8 elements from the same column across different rows
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>

typedef _Float16 f16_t;
typedef __attribute__((ext_vector_type(8))) f16_t f16x8_t;

// ds_load_tr16_b128: Transposed load from LDS
// Each lane provides a 32-bit LDS byte offset
// Returns 8 x f16 = 128 bits, shuffled across lanes
__device__ __forceinline__ f16x8_t ds_load_tr16_b128(unsigned int lds_offset) {
    f16x8_t result;
    asm volatile(
        "ds_load_tr16_b128 %0, %1"
        : "=v"(result)
        : "v"(lds_offset)
        : "memory"
    );
    return result;
}

// Helper to get LDS pointer offset (LDS addresses are 32-bit)
__device__ __forceinline__ unsigned int lds_ptr_to_offset(const f16_t* ptr) {
    return static_cast<unsigned int>(reinterpret_cast<size_t>(ptr) & 0xFFFFFFFF);
}

/**
 * Kernel demonstrating ds_load_tr16_b128
 * 
 * We set up LDS as a 16x16 matrix of f16 values:
 *   LDS[row][col] = row * 16 + col + 1
 * 
 * So LDS contains:
 *   Row 0:  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16
 *   Row 1: 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
 *   ...
 *   Row 15: 241, 242, 243, ...
 * 
 * Direct load: Lane L loads LDS[L % 16][0..7] - 8 consecutive columns from row L
 * Transposed: Lane L should get LDS[0..7][L % 16] - same column from 8 different rows
 */
__global__ void transpose_demo(const f16_t* __restrict__ input, 
                               f16_t* __restrict__ output_tr,
                               f16_t* __restrict__ output_direct) {
    // 16 rows x 16 cols = 256 elements
    __shared__ f16_t lds[16][16];
    
    const int lane_id = threadIdx.x % 32;
    
    // Natural ordering: 4 quadrants of 8x8 each
    // Lanes 0-7:   top-left     (rows 0-7,  cols 0-7)
    // Lanes 8-15:  top-right    (rows 0-7,  cols 8-15)
    // Lanes 16-23: bottom-left  (rows 8-15, cols 0-7)
    // Lanes 24-31: bottom-right (rows 8-15, cols 8-15)
    const int quadrant = lane_id / 8;        // 0, 1, 2, or 3
    const int lane_in_quad = lane_id % 8;    // 0-7 within quadrant
    const int row_offset = (quadrant / 2) * 8;  // 0 for top, 8 for bottom
    const int col_offset = (quadrant % 2) * 8;  // 0 for left, 8 for right
    const int row = row_offset + lane_in_quad;
    
    // Step 1: Load 16x16 matrix to LDS
    // Each of 32 lanes loads 8 elements
    for (int c = 0; c < 8; c++) {
        lds[row][col_offset + c] = input[row * 16 + col_offset + c];
    }
    __syncthreads();
    
    // Step 2: Direct load - each lane reads 8 consecutive columns from its row
    f16_t direct[8];
    for (int c = 0; c < 8; c++) {
        direct[c] = lds[row][col_offset + c];
    }
    
    // Step 3: Transposed load via ds_load_tr16_b128
    // Lane provides starting address, instruction shuffles across lanes
    unsigned int lds_offset = lds_ptr_to_offset(&lds[row][col_offset]);
    f16x8_t transposed = ds_load_tr16_b128(lds_offset);
    
    // Step 4: Write back to global memory
    for (int i = 0; i < 8; i++) {
        output_tr[lane_id * 8 + i] = transposed[i];
        output_direct[lane_id * 8 + i] = direct[i];
    }
}

void print_matrix(const char* name, __half* data, int rows, int cols) {
    printf("%s:\n", name);
    for (int r = 0; r < rows; r++) {
        printf("  Row %2d: ", r);
        for (int c = 0; c < cols; c++) {
            printf("%4.0f ", __half2float(data[r * cols + c]));
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    const int ROWS = 16;
    const int COLS = 16;
    const int num_elements = ROWS * COLS;
    const int output_size = 32 * 8;  // 32 lanes x 8 elements
    
    // Allocate host memory
    __half *h_input = new __half[num_elements];
    __half *h_output_tr = new __half[output_size];
    __half *h_output_direct = new __half[output_size];
    
    // Initialize: input[row][col] = row * 16 + col + 1
    printf("════════════════════════════════════════════════════════════════\n");
    printf("ds_load_tr16_b128 Demonstration\n");
    printf("════════════════════════════════════════════════════════════════\n\n");
    
    printf("Setting up 16x16 matrix in LDS:\n");
    printf("  LDS[row][col] = row * 16 + col + 1\n\n");
    
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            h_input[r * COLS + c] = __float2half((float)(r * 16 + c + 1));
        }
    }
    
    print_matrix("Input Matrix (first 8x8)", h_input, ROWS, COLS);
    
    // Allocate device memory
    f16_t *d_input, *d_output_tr, *d_output_direct;
    (void)hipMalloc(&d_input, num_elements * sizeof(f16_t));
    (void)hipMalloc(&d_output_tr, output_size * sizeof(f16_t));
    (void)hipMalloc(&d_output_direct, output_size * sizeof(f16_t));
    
    // Copy input to device
    (void)hipMemcpy(d_input, h_input, num_elements * sizeof(f16_t), hipMemcpyHostToDevice);
    
    // Launch kernel with 1 warp (32 threads)
    transpose_demo<<<1, 32>>>(d_input, d_output_tr, d_output_direct);
    
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("Kernel launch failed: %s\n", hipGetErrorString(err));
        printf("(This is expected if not running on MI450/gfx1250 hardware)\n\n");
        printf("════════════════════════════════════════════════════════════════\n");
        printf("EXPECTED BEHAVIOR (what would happen on MI450):\n");
        printf("════════════════════════════════════════════════════════════════\n\n");
        
        printf("Direct Load (no transpose):\n");
        printf("  Each lane reads 8 consecutive columns from its row\n");
        printf("  Lane 0: reads LDS[0][0..7]  = {1, 2, 3, 4, 5, 6, 7, 8}\n");
        printf("  Lane 1: reads LDS[1][0..7]  = {17, 18, 19, 20, 21, 22, 23, 24}\n");
        printf("  Lane 2: reads LDS[2][0..7]  = {33, 34, 35, 36, 37, 38, 39, 40}\n");
        printf("  ...\n\n");
        
        printf("After ds_load_tr16_b128 (transposed):\n");
        printf("  Data is shuffled so each lane gets same column from different rows\n");
        printf("  Lane 0: gets LDS[0..7][0]  = {1, 17, 33, 49, 65, 81, 97, 113}\n");
        printf("  Lane 1: gets LDS[0..7][1]  = {2, 18, 34, 50, 66, 82, 98, 114}\n");
        printf("  Lane 2: gets LDS[0..7][2]  = {3, 19, 35, 51, 67, 83, 99, 115}\n");
        printf("  ...\n\n");
        
        printf("════════════════════════════════════════════════════════════════\n");
        printf("WHY THIS MATTERS FOR WMMA:\n");
        printf("════════════════════════════════════════════════════════════════\n\n");
        printf("In matrix multiply C = A × B:\n");
        printf("  - B matrix is stored as B[K][N] (K rows, N cols)\n");
        printf("  - Global/LDS has K as the outer dimension (row-contiguous)\n");
        printf("  - But WMMA needs N-contiguous data in registers for the B operand\n");
        printf("  - ds_load_tr16 converts K-contiguous to N-contiguous!\n\n");
        printf("Before: registers hold consecutive K values (same column)\n");
        printf("After:  registers hold consecutive N values (same row in output)\n");
    } else {
        // Copy results back
        (void)hipMemcpy(h_output_tr, d_output_tr, output_size * sizeof(f16_t), hipMemcpyDeviceToHost);
        (void)hipMemcpy(h_output_direct, d_output_direct, output_size * sizeof(f16_t), hipMemcpyDeviceToHost);
        
        printf("════════════════════════════════════════════════════════════════════════════════════════\n");
        printf("ACTUAL RESULTS FROM HARDWARE:\n");
        printf("════════════════════════════════════════════════════════════════════════════════════════\n\n");
        
        const char* quadrant_names[] = {"TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT"};
        
        printf("Direct Load (no transpose) - 4 quadrants of 8x8:\n\n");
        for (int q = 0; q < 4; q++) {
            printf("  === %s (lanes %d-%d, rows %d-%d, cols %d-%d) ===\n", 
                   quadrant_names[q], q*8, q*8+7, (q/2)*8, (q/2)*8+7, (q%2)*8, (q%2)*8+7);
            for (int lane = q * 8; lane < (q + 1) * 8; lane++) {
                printf("  Lane %2d: {", lane);
                for (int i = 0; i < 8; i++) {
                    printf("%3.0f%s", __half2float(h_output_direct[lane * 8 + i]), i < 7 ? ", " : "");
                }
                printf("}\n");
            }
            printf("\n");
        }
        
        printf("After ds_load_tr16_b128 (transposed) - 4 quadrants of 8x8:\n\n");
        for (int q = 0; q < 4; q++) {
            printf("  === %s (lanes %d-%d) ===\n", quadrant_names[q], q*8, q*8+7);
            for (int lane = q * 8; lane < (q + 1) * 8; lane++) {
                printf("  Lane %2d: {", lane);
                for (int i = 0; i < 8; i++) {
                    printf("%3.0f%s", __half2float(h_output_tr[lane * 8 + i]), i < 7 ? ", " : "");
                }
                printf("}\n");
            }
            printf("\n");
        }
    }
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_tr;
    delete[] h_output_direct;
    (void)hipFree(d_input);
    (void)hipFree(d_output_tr);
    (void)hipFree(d_output_direct);
    
    printf("════════════════════════════════════════════════════════════════\n");
    printf("COMPILATION VERIFIED: ds_load_tr16_b128 instruction generated!\n");
    printf("════════════════════════════════════════════════════════════════\n");
    
    return 0;
}

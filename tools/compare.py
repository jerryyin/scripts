import argparse
import numpy as np
import os

def read_binary_file(file_path, dtype):
    """Reads a binary file and interprets it as an array with the specified dtype."""
    with open(file_path, "rb") as f:
        array_data = np.frombuffer(f.read(), dtype=dtype)
    return array_data

def bfloat16_to_float32(bf16_array):
    """Converts a BFloat16 numpy array to float32."""
    # View the array as uint16 to access the raw bits
    uint16_array = bf16_array.view(np.uint16)
    # Convert the 16-bit values to 32-bit by shifting bits
    uint32_array = uint16_array.astype(np.uint32) << 16
    # View as float32
    return uint32_array.view(np.float32)

def compare_arrays(file1, file2, dtype, threshold):
    """Compare two binary files containing array data.

    Parameters:
        file1 (str): Path to the first binary file.
        file2 (str): Path to the second binary file.
        dtype (str): Data type of the array in the binary files. Supports "bf16", "f32", "f16", "i32".
        threshold (float): The absolute threshold for comparison.

    Returns:
        bool: True if all differences are within the threshold, False otherwise.
    """
    file_size = os.path.getsize(file1)
    dtype_map = {
        "bf16": np.uint16,
        "f32": np.float32,
        "f16": np.float16,
        "i32": np.int32
    }

    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported types are bf16, f32, f16, i32.")

    element_size = 2 if dtype == "bf16" else np.dtype(dtype_map[dtype]).itemsize
    element_count = file_size // element_size

    data1 = read_binary_file(file1, dtype_map[dtype])
    data2 = read_binary_file(file2, dtype_map[dtype])

    if dtype == "bf16":
        data1 = bfloat16_to_float32(data1)
        data2 = bfloat16_to_float32(data2)

    if dtype == "f16":
        data1 = data1.astype(np.float32)
        data2 = data2.astype(np.float32)

    # Calculate differences
    diff_values = np.abs(data1 - data2)
    error_mask = diff_values > threshold
    error_count = np.sum(error_mask)

    # Print first few differences
    if error_count > 0:
        error_indices = np.where(error_mask)[0]
        num_to_show = min(10, error_count)
        for idx in error_indices[:num_to_show]:
            print(f"Difference exceeds threshold at index {idx}: {data1[idx]} vs {data2[idx]}")

    if error_count > 0:
        proportion = error_count / element_count
        print(f"Total differences: {error_count} out of {element_count} elements ({proportion:.2%})")

        # Enhanced statistics
        print(f"")
        print(f"Difference statistics:")
        diff_nonzero = diff_values[error_mask]
        print(f"  Mean absolute diff:      {np.mean(diff_nonzero):.6f}")
        print(f"  Max absolute diff:       {np.max(diff_nonzero):.6f}")
        print(f"  Min absolute diff:       {np.min(diff_nonzero):.6f}")
        print(f"  Std dev of diff:         {np.std(diff_nonzero):.6f}")

        # Percentiles for better understanding of distribution
        print(f"  Median absolute diff:    {np.median(diff_nonzero):.6f}")
        print(f"  95th percentile diff:    {np.percentile(diff_nonzero, 95):.6f}")
        print(f"  99th percentile diff:    {np.percentile(diff_nonzero, 99):.6f}")

        # Relative error if values are non-zero
        abs_data1 = np.abs(data1[error_mask])
        valid_for_rel_error = abs_data1 > 1e-10
        if np.any(valid_for_rel_error):
            rel_errors = diff_nonzero[valid_for_rel_error] / abs_data1[valid_for_rel_error]
            print(f"")
            print(f"Relative error statistics (where |value1| > 1e-10):")
            print(f"  Mean relative error:     {np.mean(rel_errors):.6f} ({np.mean(rel_errors)*100:.4f}%)")
            print(f"  Max relative error:      {np.max(rel_errors):.6f} ({np.max(rel_errors)*100:.4f}%)")
            print(f"  Median relative error:   {np.median(rel_errors):.6f} ({np.median(rel_errors)*100:.4f}%)")

        return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Compare two binary files element-wise.")
    parser.add_argument("file1", type=str, help="Path to the first binary file.")
    parser.add_argument("file2", type=str, help="Path to the second binary file.")
    parser.add_argument("--dtype", type=str, default="bf16", help="Data type of the binary files.")
    parser.add_argument("--threshold", type=float, default=0.01, help="Absolute comparison threshold.")

    args = parser.parse_args()

    result = compare_arrays(args.file1, args.file2, args.dtype, args.threshold)
    if result:
        print("All differences are within the threshold.")
    else:
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()

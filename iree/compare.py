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

    error_count = 0
    for i in range(element_count):
        if np.abs(data1[i] - data2[i]) > threshold:
            if error_count <= 10:
                print(f"Difference exceeds threshold at index {i}: {data1[i]} vs {data2[i]}")
            error_count += 1

    if error_count > 0:
        proportion = error_count / element_count
        print(f"Total differences: {error_count} out of {element_count} elements ({proportion:.2%})")
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

if __name__ == "__main__":
    main()

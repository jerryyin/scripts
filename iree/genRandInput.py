## Usage:
## Gen rand output:
##    python genRandInput.py 2x235x363x224xbf16.bin --shape 2 235 363 224 --dtype bf16
## Gen readable output from bin input:
##    python genRandInput.py input.bin --shape 1 2 3 --dtype bf16 --dump

import numpy as np
import argparse
import sys

# Mapping for numpy dtype and struct format character
DTYPE_MAP = {
    'f32': (np.float32, 'f'),
    'f16': (np.float16, 'e'),
    'i8': (np.int8, 'b'),
    'bf16': (np.float32, 'f')  # bf16 in numpy is not supported directly, workaround by using float32.
}

def parse_shape(shape_str):
    """Parse a shape string with 'x' separator into a list of integers."""
    return list(map(int, shape_str.split('x')))

def convert_to_bf16(data: np.ndarray) -> np.ndarray:
    """Convert a float32 numpy array to bf16."""
    if data.dtype != np.float32:
        raise ValueError("Expected float32 input for bf16 conversion.")
    int_data = data.view(np.uint32)
    bf16_data = ((int_data >> 16) & 0xFFFF)  # Keep only the most significant bits corresponding to bf16
    return bf16_data.astype(np.uint16)  # Represent data in bf16 bit format

def bf16_to_float32(bf16_data: np.ndarray) -> np.ndarray:
    """Convert bf16 stored data back to float32."""
    if bf16_data.dtype != np.uint16:
        raise ValueError("Expected uint16 input for fp32 conversion.")
    float32_data = (bf16_data.astype(np.uint32) << 16).view(np.float32)
    return float32_data

def bin_to_readable(bin_file, shape_str, dtype_str):
   shape = parse_shape(shape_str)

    # Read from binary file and write to a readable text format
   with open(bin_file, "rb") as f:
       dtype, fmt_char = DTYPE_MAP[dtype_str]
       bytearr = f.read()

       if dtype_str == 'bf16':
           bf16_data = np.frombuffer(bytearr, dtype=np.uint16)
           data = bf16_to_float32(bf16_data)
       else:
           data = np.frombuffer(bytearr, dtype=dtype)

       tensor = data.reshape(shape)
       print(np.array2string(tensor, separator=', ', precision=6))

def generate_random_tensor_new(shape_str, dtype_str, bin_file):
    shape = parse_shape(shape_str)

    # Generate a random tensor with the specified shape
    rng = np.random.default_rng(19)
    dtype, fmt_char = DTYPE_MAP[dtype_str]
    random_sequence = rng.random(shape).astype(dtype)

    # Determine default binary output file name if not specified
    bin_file = bin_file or f"{shape_str}x{dtype_str}.bin"

    with open(bin_file, "wb") as f:
        if dtype_str == 'bf16':
            bf16_data = convert_to_bf16(random_sequence)
            f.write(bf16_data.tobytes())
        else:
            data = random_sequence.astype(dtype)
            f.write(data.tobytes())

    print(f"Binary tensor saved to {bin_file}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate a random tensor or convert a binary tensor file to a readable text format.")
    parser.add_argument("bin", nargs='?', default=None, type=str,
                        help="The binary data file name. Defaults to '<shape>x<dtype>.bin'.")
    parser.add_argument("--shape", required=True, type=str,
                        help="The shape of the tensor for generation, separated by 'x' (e.g., '1x3x224x224').")
    # If provided without an argument, it defaults to True.
    # If not provided at all, it defaults to None.
    parser.add_argument('--dump', nargs='?', const=True, default=None,
                        help="Dump option. Specify an optional file name.")
    parser.add_argument("--dtype", type=str, choices=DTYPE_MAP.keys(), default='fp32',
                        help="Specify the data type for tensor generation (e.g., fp32, fp16, int8, bf16)")
    args = parser.parse_args()

    if args.dtype == 'bf16' and sys.version_info < (3, 9):
        print("Warning: Using float32 as a workaround for bf16, requires Python 3.9 or later for proper compatibility.")

    if args.dump is True:
        bin_to_readable(args.bin, args.shape, args.dtype)
    else:
        generate_random_tensor_new(args.shape, args.dtype, args.bin)

if __name__ == "__main__":
    main()

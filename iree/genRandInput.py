# Usage:
# Gen rand output:
#    python genRandInput.py input.bin --shape 1 3 244 244 --dtype fp32
#    python genRandInput.py 2x235x363x224xbf16.bin --shape 2 235 363 224 --dtype bf16
# Gen readable output from bin input:
#    python3 ~/scripts/iree/genRandInput.py readable.txt --input input.bin --shape 1 2 3 --dtype bf16

import numpy as np
import argparse
import sys

# Mapping for numpy dtype and struct format character
DTYPE_MAP = {
    'fp32': (np.float32, 'f'),
    'fp16': (np.float16, 'e'),
    'int8': (np.int8, 'b'),
    'bf16': (np.float32, 'f')  # bf16 in numpy is not supported directly, workaround by using float32.
}

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate a random tensor or convert a binary tensor file to a readable text format.")
parser.add_argument("output", type=str, help="The output file name (e.g., input.bin, readable.txt)")
parser.add_argument("--shape", type=int, nargs="+", required=True,
                    help="The shape of the tensor for generation (e.g., 1 3 224 224). Required for generating a new file.")
parser.add_argument("--input", type=str, help="Specify a binary input file to convert to a readable text format.")
parser.add_argument("--dtype", type=str, choices=DTYPE_MAP.keys(), default='fp32',
                    help="Specify the data type for tensor generation (e.g., fp32, fp16, int8, bf16)")
args = parser.parse_args()

# Validate bf16 workaround
if args.dtype == 'bf16' and sys.version_info < (3, 9):
    print("Warning: Using float32 as a workaround for bf16, requires Python 3.9 or later for proper compatibility.")

def convert_to_bf16(data: np.ndarray) -> np.ndarray:
    """Convert a float32 numpy array to bf16."""
    if data.dtype != np.float32:
        raise ValueError("Expected float32 input for bf16 conversion.")
    # Treat data as int bits
    int_data = data.view(np.int32)
    # Shift right to keep most significant bits corresponding to bf16
    bf16_data = ((int_data >> 16) & 0xFFFF)
    return bf16_data.astype(np.uint16)  # Represent data in bf16 bit format

def bf16_to_float32(bf16_data: np.ndarray) -> np.ndarray:
    """Convert bf16 stored data back to float32."""
    # Create an int version of bf16 data interpreted as float32
    float32_data = (bf16_data.astype(np.int32) << 16).view(np.float32)
    return float32_data

if args.input:
    # Read from binary file and write to a readable text format
    with open(args.input, "rb") as f:
        num_elements = np.prod(args.shape)

        dtype, fmt_char = DTYPE_MAP[args.dtype]
        bytearr = f.read()

        if args.dtype == 'bf16':
            bf16_data = np.frombuffer(bytearr, dtype=np.uint16)
            data = bf16_to_float32(bf16_data)
        else:
            data = np.frombuffer(bytearr, dtype=dtype)

        # Reshape the data to the specified shape
        tensor = data.reshape(args.shape)

        # Save the tensor in a readable format, preserving the shape
        with open(args.output, "w") as txt_file:
            txt_file.write(np.array2string(tensor, separator=', ', precision=6))
        print(f"Readable tensor saved to {args.output}")
else:
    # Generate a random tensor with the specified shape
    rng = np.random.default_rng(19)
    dtype, fmt_char = DTYPE_MAP[args.dtype]
    a = rng.random(args.shape).astype(dtype)

    with open(args.output, "wb") as f:
        if args.dtype == 'bf16':
            bf16_data = convert_to_bf16(a)
            f.write(bf16_data.tobytes())
        else:
            a = a.astype(dtype)
            f.write(a.tobytes())

    print(f"Binary tensor saved to {args.output}")

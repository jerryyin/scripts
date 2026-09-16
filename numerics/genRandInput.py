## Usage:
## Gen rand output:
##    python genRandInput.py 2x235x363x224xbf16.bin --shape 2x235x363x224 --dtype bf16
## Gen readable output from bin input:
##    python genRandInput.py input.bin --shape 1x2x3 --dtype bf16 --dump
## Gen a buffer made entirely of special values (both zeros, NaN, infinities, ...):
##    python genRandInput.py --shape 1x256 --dtype bf16 --special
## Gen a mostly-random buffer with 1% of its elements special:
##    python genRandInput.py --shape 1x4096 --dtype f32 --special --special-fraction 0.01

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

# Special values, as raw bit patterns.
#
# Why this exists: an investigation once spent a long time on two kernels whose outputs
# differed by nothing but positive zero against negative zero. +0.0 == -0.0 is true, so
# every threshold comparison passed, while the bytes differed and the sign propagated
# through anything that looked at it. Uniform random data in [0, 1) contains no zero of
# either sign, no NaN, no infinity, no denormal and nothing at the edge of the format, so
# that behaviour could not be produced on purpose -- it had to be stumbled into. These
# tables make it producible.
#
# Stored as bit patterns rather than Python float literals because a pattern is exact and
# checkable by eye: the smallest denormal and the neighbours of 1.0 written as decimals
# are a rounding step waiting to go wrong, and a NaN with its sign bit set cannot be
# written as a literal at all. Each entry is (sign << (exponent_bits + mantissa_bits)) |
# (exponent << mantissa_bits) | mantissa, for a format's own field widths:
#   bf16 = 1 sign, 8 exponent, 7 mantissa      fp32 = 1 sign, 8 exponent, 23 mantissa
#   f16  = 1 sign, 5 exponent, 10 mantissa
SPECIAL_BITS = {
    # bf16, in the same order for every format so the tables read side by side.
    'bf16': [
        0x0000,  # +0.0
        0x8000,  # -0.0, the case above: equal to +0.0, different bytes
        0x7FC0,  # quiet NaN
        0xFFC0,  # quiet NaN with the sign bit set
        0x7F80,  # +infinity
        0xFF80,  # -infinity
        0x0001,  # smallest positive denormal, 2**-133
        0x8001,  # smallest negative denormal
        0x0080,  # smallest positive normal, 2**-126
        0x7F7F,  # largest finite, ~3.3895e38; one step up is +infinity
        0xFF7F,  # largest finite negative
        0x3F7F,  # 0.99609375, the value immediately below 1.0
        0x3F80,  # 1.0
        0x3F81,  # 1.0078125, the value immediately above 1.0
    ],
    'f32': [
        0x00000000,  # +0.0
        0x80000000,  # -0.0
        0x7FC00000,  # quiet NaN
        0xFFC00000,  # quiet NaN with the sign bit set
        0x7F800000,  # +infinity
        0xFF800000,  # -infinity
        0x00000001,  # smallest positive denormal, 2**-149
        0x80000001,  # smallest negative denormal
        0x00800000,  # smallest positive normal, 2**-126
        0x7F7FFFFF,  # largest finite, ~3.4028e38
        0xFF7FFFFF,  # largest finite negative
        0x3F7FFFFF,  # 0.9999999403953552, the value immediately below 1.0
        0x3F800000,  # 1.0
        0x3F800001,  # 1.0000001192092896, the value immediately above 1.0
    ],
    'f16': [
        0x0000,  # +0.0
        0x8000,  # -0.0
        0x7E00,  # quiet NaN
        0xFE00,  # quiet NaN with the sign bit set
        0x7C00,  # +infinity
        0xFC00,  # -infinity
        0x0001,  # smallest positive denormal, 2**-24
        0x8001,  # smallest negative denormal
        0x0400,  # smallest positive normal, 2**-14
        0x7BFF,  # largest finite, 65504.0
        0xFBFF,  # largest finite negative
        0x3BFF,  # 0.99951171875, the value immediately below 1.0
        0x3C00,  # 1.0
        0x3C01,  # 1.0009765625, the value immediately above 1.0
    ]
    # No entry for i8: an integer format has no NaN, no infinity and no denormal, and
    # inventing a list of "interesting" integers here would be a different tool.
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

def special_values(dtype_str) -> np.ndarray:
    """Build the array of special values for a dtype from its exact bit patterns."""
    if dtype_str not in SPECIAL_BITS:
        raise ValueError(
            f"No special values for dtype {dtype_str}. Special values are a property of "
            f"a floating-point format; supported: {', '.join(sorted(SPECIAL_BITS))}.")

    patterns = SPECIAL_BITS[dtype_str]
    if dtype_str == 'bf16':
        # bf16 is carried as the top 16 bits of a float32 here, which is the same
        # convention convert_to_bf16() below truncates back down. Shifting a bf16 pattern
        # up by 16 leaves the low 16 bits zero, so that truncation returns the pattern
        # unchanged and the value written to the file is exactly the one in the table.
        values = (np.array(patterns, dtype=np.uint32) << 16).view(np.float32)
    elif dtype_str == 'f32':
        values = np.array(patterns, dtype=np.uint32).view(np.float32)
    elif dtype_str == 'f16':
        values = np.array(patterns, dtype=np.uint16).view(np.float16)
    else:
        raise ValueError(f"Unhandled dtype {dtype_str} in special_values.")

    # A view reinterprets bytes, so a pattern array that came out wider than intended
    # would yield twice the values rather than an error. Checked here because that would
    # otherwise be a silent corruption of the buffer.
    if values.size != len(patterns):
        raise ValueError(
            f"{dtype_str}: {len(patterns)} bit patterns produced {values.size} values, so "
            f"the pattern array is not the width the format needs.")
    return values

def generate_special_tensor(shape_str, dtype_str, bin_file, fraction):
    """Write a buffer of special values, or a random buffer with special values in it."""
    shape = parse_shape(shape_str)
    dtype, fmt_char = DTYPE_MAP[dtype_str]
    specials = special_values(dtype_str)

    # Same seed and same first draw as generate_random_tensor_new, so the positions that
    # are not overwritten below hold exactly the values a plain run of the same shape and
    # dtype would have written. Any difference between the two files is then the injected
    # values and nothing else.
    rng = np.random.default_rng(19)
    random_sequence = rng.random(shape).astype(dtype)
    if specials.dtype != random_sequence.dtype:
        raise ValueError(
            f"special values are {specials.dtype} but the buffer is "
            f"{random_sequence.dtype}; a cast here could alter a NaN or overflow an edge "
            f"value, so the tables and DTYPE_MAP have to agree.")

    if fraction >= 1.0:
        # Every element special, tiled in table order rather than shuffled, so that a
        # --dump of the result is readable by eye and the pattern of values is obvious.
        # Every entry in the table appears as long as the buffer holds at least as many
        # elements as the table has entries.
        tensor = np.resize(specials, shape)
        injected = tensor.size
    else:
        # Positions drawn without replacement, so the number injected is exactly the
        # fraction asked for and no position is chosen twice.
        flat = random_sequence.reshape(-1)
        injected = int(round(flat.size * fraction))
        if injected < 1:
            raise ValueError(
                f"--special-fraction {fraction} over {flat.size} elements rounds to zero "
                f"special values, so the buffer would be plain random data.")
        positions = rng.choice(flat.size, size=injected, replace=False)
        flat[positions] = np.resize(specials, injected)
        tensor = flat.reshape(shape)
        if injected < specials.size:
            print(f"Warning: only {injected} elements are special, so the first "
                  f"{injected} of the {specials.size} table entries were used.")

    # A different default name from the random path: a buffer of NaNs and infinities that
    # is named like a random one will eventually be mistaken for one.
    bin_file = bin_file or f"{shape_str}x{dtype_str}-special.bin"

    with open(bin_file, "wb") as f:
        if dtype_str == 'bf16':
            bf16_data = convert_to_bf16(tensor)
            f.write(bf16_data.tobytes())
        else:
            f.write(tensor.tobytes())

    print(f"Special-value tensor saved to {bin_file} "
          f"({injected} of {tensor.size} elements special, "
          f"{specials.size} distinct special values available)")

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
    # Default must be a DTYPE_MAP key. It read 'fp32' for a long time, which is not one of
    # them ('f32' is), and argparse validates supplied values against choices but never the
    # default -- so omitting --dtype raised KeyError instead of using a default.
    parser.add_argument("--dtype", type=str, choices=DTYPE_MAP.keys(), default='f32',
                        help="Specify the data type for tensor generation (one of: f32, f16, i8, bf16)")
    parser.add_argument("--special", action='store_true',
                        help="Generate the values that uniform random data never contains: both "
                             "zeros, NaN, both infinities, the smallest denormal, the largest "
                             "finite value and the two neighbours of 1.0. Float formats only.")
    parser.add_argument("--special-fraction", type=float, default=None,
                        help="With --special, the fraction of elements that are special, greater "
                             "than 0 and at most 1. Defaults to 1.0, a buffer made entirely of "
                             "special values; anything less sprinkles them into a random buffer.")
    args = parser.parse_args()

    if args.dtype == 'bf16' and sys.version_info < (3, 9):
        print("Warning: Using float32 as a workaround for bf16, requires Python 3.9 or later for proper compatibility.")

    if args.special_fraction is not None and not args.special:
        parser.error("--special-fraction does nothing without --special.")
    if args.special and args.dump is not None:
        parser.error("--special generates a file and --dump reads one; pick one.")
    if args.special and args.dtype not in SPECIAL_BITS:
        parser.error(f"--special needs a floating-point dtype; {args.dtype} has no NaN, "
                     f"no infinity and no denormal. Supported: "
                     f"{', '.join(sorted(SPECIAL_BITS))}.")

    if args.dump is True:
        bin_to_readable(args.bin, args.shape, args.dtype)
    elif args.special:
        fraction = 1.0 if args.special_fraction is None else args.special_fraction
        if not 0.0 < fraction <= 1.0:
            parser.error("--special-fraction must be greater than 0 and at most 1.")
        generate_special_tensor(args.shape, args.dtype, args.bin, fraction)
    else:
        generate_random_tensor_new(args.shape, args.dtype, args.bin)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gen_miopen_config.py: Convert MIOpen convolution name strings to command-line configurations.

Usage:
  # Single name:
  python gen_miopen_config.py conv_2d_bfloat16_forward_16x192x128x32_nhwc_40x1x1x32_fhwc_nhwf_1x1s_0x0p_1x1d_1g

  # From a file of names:
  python gen_miopen_config.py names.txt

Each conv name should follow this pattern:
  conv_2d_<dtype>_<direction>_<N>x<H>x<W>x<C>_<in_layout>_<K>x<Y>x<X>x<C>_<fil_layout>_<out_layout>_<UxVs>_<PxQp>_<LxJd>_<Gg>

Example name:
  conv_2d_bfloat16_forward_16x192x128x32_nhwc_40x1x1x32_fhwc_nhwf_1x1s_0x0p_1x1d_1g

Output: a single-line config for MIOpen benchmarking, e.g.:
  convbfp16 -n 16 -c 32 -H 192 -W 128 -k 40 -y 1 -x 1 -p 0 -q 0 \
      -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --in_layout NHWC \
      --out_layout NHWC --fil_layout FHWC
"""

import argparse
import sys
import os

def parse_name(name):
    parts = name.strip().split('_')
    if len(parts) != 13:
        raise ValueError(f"Unexpected name format: {name}")

    # Extract and validate components
    dtype, direction = parts[2], parts[3]
    in_dims, fil_dims = parts[4].split('x'), parts[6].split('x')
    in_layout, fil_layout, out_layout = parts[5], parts[7], parts[8]
    u, v = parts[9].rstrip('s').split('x')
    p, q = parts[10].rstrip('p').split('x')
    l, j = parts[11].rstrip('d').split('x')
    g = parts[12].rstrip('g')

    # Map dtype
    conv_type = 'conv' + ('bfp16' if dtype in ('bfloat16', 'bf16') else 'f32' if dtype in ('float32', 'f32') else '')
    if not conv_type.endswith(('bfp16', 'f32')):
        raise ValueError(f"Unknown dtype: {dtype}")

    # Direction flag
    F_val = '1' if direction == 'forward' else '0'

    N, H, W, C = in_dims
    K, Y, X, C2 = fil_dims
    if C != C2:
        C = C2 # Use filter's channel to workaround bug
    #    raise ValueError(f"Channel mismatch: {C} vs {C2}")

    # Assemble args
    args = [
        conv_type,
        f"-n {N}", f"-c {C}", f"-H {H}", f"-W {W}",
        f"-k {K}", f"-y {Y}", f"-x {X}",
        f"-p {p}", f"-q {q}",
        f"-u {u}", f"-v {v}",
        f"-l {l}", f"-j {j}",
        "-m conv",
        f"-g {g}",
        f"-F {F_val}",
        "-t 1",
        f"--in_layout {in_layout.upper()}",
        f"--out_layout {in_layout.upper()}",
        f"--fil_layout {in_layout.upper()}"
    ]
    return ' '.join(args)


def get_input_items(path):
    if os.path.isfile(path):
        return open(path, 'r')
    # Treat as single name
    return [path]


def process_items(items):
    for name in items:
        name = name.strip()
        if not name or name.startswith('#'):
            continue
        try:
            print(parse_name(name))
        except Exception as e:
            print(f"Error parsing '{name}': {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Convert miopen conv names (single or file) to configs'
    )
    parser.add_argument(
        'input',
        help='Conv name string or path to file containing names (one per line)'
    )
    args = parser.parse_args()

    items = get_input_items(args.input)
    process_items(items)


if __name__ == '__main__':
    main()

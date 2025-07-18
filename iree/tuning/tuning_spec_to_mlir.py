#!/usr/bin/env python3
"""
Convert an IREE tuning spec (transform named sequences) to an MLIR file with individual functions.

This script reads an MLIR file containing IREE transform tuning specifications (with named_sequence for convolution patterns),
and generates an MLIR file where each convolution pattern is converted into a standalone func.func.
The function body includes the linalg.generic loop from the tuning spec, preserving the indexing maps and computations.

Usage:
    python3 tuning_spec_to_mlir.py input_tuning_specs.mlir

Each func.func is named based on the transform.named_sequence name (up to the '$' character), to ensure uniqueness.
"""

import sys
import re

def read_mlir_file(path):
    with open(path, 'r') as f:
        return f.readlines()

def is_match_sequence_start(line):
    return 'transform.named_sequence' in line and '@match_conv' in line

def sanitize_function_name(seq_name):
    return seq_name.lstrip('@').split('$')[0]

def extract_named_sequence_name(line):
    match = re.search(r'@([^(\s]+)', line)
    if not match:
        return None
    return sanitize_function_name(match.group(1))

def find_bb0_args(lines, start_index):
    for i in range(start_index, len(lines)):
        match = re.match(r'\s*\^bb0\((.*?)\):', lines[i])
        if match:
            args = []
            for part in match.group(1).split(','):
                name_type = part.strip().split(':')
                if len(name_type) == 2:
                    args.append((name_type[0].strip(), name_type[1].strip()))
            return args, i
    return [], start_index

def extract_linalg_generic_block(lines, start_index):
    for i in range(start_index, len(lines)):
        if 'linalg.generic' in lines[i]:
            start = i
            end = None
            for j in range(start + 1, len(lines)):
                if '}' in lines[j] and '-> tensor<' in lines[j]:
                    end = j
                    break
            if end is not None:
                return lines[start:end+1], end
    return [], start_index

def indent_linalg_block(block_lines):
    indented = []
    for i, line in enumerate(block_lines):
        stripped = line.lstrip()
        if i == 0:
            indented.append("    " + stripped)
        elif stripped.startswith('^bb0'):
            indented.append("        " + stripped)
        elif stripped.startswith('}'):
            indented.append("    " + stripped)
        else:
            indented.append("        " + stripped)
    return indented

def get_return_type(line):
    match = re.search(r'->\s*(tensor<[^>]+>)', line)
    return match.group(1) if match else "tensor<unknown>"

def extract_functions(lines):
    """
    Parse the input tuning spec file and extract transform.named_sequence blocks for convolutions,
    converting each to an MLIR func.func with the linalg.generic body.
    """
    i = 0
    results = []
    while i < len(lines):
        if is_match_sequence_start(lines[i]):
            func_name = extract_named_sequence_name(lines[i])
            args, bb0_index = find_bb0_args(lines, i + 1)
            linalg_block, end_index = extract_linalg_generic_block(lines, bb0_index + 1)
            if not linalg_block:
                i += 1
                continue
            indented_body = indent_linalg_block(linalg_block)
            return_type = get_return_type(linalg_block[-1])
            results.append((func_name, args, return_type, indented_body))
            i = end_index
        else:
            i += 1
    return results

def emit_functions_to_stdout(functions):
    for idx, (name, args, ret_type, body) in enumerate(functions):
        args_text = ", ".join(f"{arg}: {typ}" for arg, typ in args)
        print(f"func.func @{name}({args_text}) -> {ret_type}")
        print("{")
        for line in body:
            print(line.rstrip())
        result_var = body[0].strip().split('=')[0].strip() if '=' in body[0] else "%0"
        print(f"    return {result_var} : {ret_type}")
        print("}")
        if idx < len(functions) - 1:
            print("\n// -----\n")

def emit_run_lines(input_file):
    # Emit the run line for heuristic lowering
    print('// RUN: iree-opt --iree-gpu-test-target=gfx942 '
          '--pass-pipeline="builtin.module(iree-codegen-llvmgpu-configuration-pipeline,'
          'func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op))" '
          '%s --split-input-file')
    # Emit the run line for tuning spec lowering
    print('// RUN: iree-opt --iree-gpu-test-target=gfx942 '
          '--pass-pipeline="builtin.module(iree-codegen-llvmgpu-configuration-pipeline,'
          'func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op))" '
          '%s --split-input-file '
          f'--iree-codegen-transform-dialect-library="{input_file}"')
    print()

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 convert_tuning_to_mlir.py <input.mlir>", file=sys.stderr)
        sys.exit(1)
    input_file = sys.argv[1]
    lines = read_mlir_file(input_file)
    functions = extract_functions(lines)
    if not functions:
        print("No matching convolution sequences found.", file=sys.stderr)
        sys.exit(1)
    emit_run_lines(input_file)
    emit_functions_to_stdout(functions)

if __name__ == "__main__":
    main()

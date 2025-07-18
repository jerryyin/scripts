import re
import sys
import math
from collections import defaultdict

def parse_conv_function_name(name):
    """
    Parses a function name like:
    match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x5x5x48_fhwc_nhwf_1x1s_8x8p_4x4d_1g
    and prints interpreted shapes and GEMM dimensions.
    """
    try:
        # Split and extract the parts
        parts = name.split('_')
        input_shape = list(map(int, parts[5].split('x')))  # e.g. 16x96x64x48
        filter_shape = list(map(int, parts[7].split('x')))  # e.g. 48x5x5x48
        layout = parts[6]  # e.g. nhwc
        filter_layout = parts[8]  # e.g. fhwc

        # Print input and filter
        print(f"Input shape: {input_shape}")
        print(f"Filter shape: {filter_shape}")

        # Assume standard NHWC and FHWC format for now
        if layout != 'nhwc' or filter_layout != 'fhwc':
            print(f"Unexpected layouts: input={layout}, filter={filter_layout}")
            return

        N, H, W, C = input_shape
        F, KH, KW, IC = filter_shape  # IC should match C

        gemmM = N * H * W
        gemmN = F
        gemmK = KH * KW * C

        print(f" - gemmM = nhw dimension = {N} x {H} x {W} = {gemmM}")
        print(f" - gemmN = f dimension = {F} = {gemmN}")
        print(f" - gemmK = hwc in fhwc dimension = {KH} x {KW} x {C} = {gemmK}")

    except Exception as e:
        print(f"Failed to parse function name '{name}': {e}")

def extract_trip_count(body):
    """Extracts the total trip count from the outermost scf.forall loop."""
    forall_match = re.search(r'scf\.forall\s*\((.*?)\)\s*=\s*\((.*?)\)\s*to\s*\((.*?)\)\s*step\s*\((.*?)\)', body)
    if not forall_match:
        return None
    lowers = list(map(int, forall_match.group(2).split(',')))
    uppers = list(map(int, forall_match.group(3).split(',')))
    steps = list(map(int, forall_match.group(4).split(',')))
    trip_counts = [math.ceil((u - l) / s) for l, u, s in zip(lowers, uppers, steps)]
    return math.prod(trip_counts)

def extract_lowering_config(body):
    """Extracts the contents of the lowering_config attribute."""
    match = re.search(r'lowering_config\s*=\s*#iree_gpu\.lowering_config<\{(.*?)\}>', body, re.DOTALL)
    return match.group(1).strip() if match else None

def parse_functions(mlir_text):
    """Parses all functions in the MLIR file and returns a dict mapping function names to their bodies."""
    funcs = defaultdict(list)
    func_pattern = re.compile(r'func\.func\s+@([\w$.-]+)\s*\(.*?\)\s*->\s*.*?\{', re.DOTALL)
    start_indices = [m.start() for m in func_pattern.finditer(mlir_text)]
    for start in start_indices:
        name_match = func_pattern.match(mlir_text, start)
        if not name_match:
            continue
        name = name_match.group(1)
        brace_count = 0
        body_count = 2 # First body for function attribute, second for actual function body
        body_start = mlir_text.find('{', start)
        for j in range(body_start, len(mlir_text)):
            if mlir_text[j] == '{':
                brace_count += 1
            elif mlir_text[j] == '}':
                brace_count -= 1
            else:
                continue

            if brace_count != 0:
                continue

            body_count -= 1
            if body_count != 0:
                continue
            funcs[name].append(mlir_text[start:j+1])
            break
    return funcs

def compare_functions(funcs):
    """Compares function pairs and prints trip counts and lowering config differences."""
    errors = []
    for name, bodies in funcs.items():
        if len(bodies) != 2:
            errors.append(f"Function @{name} appeared {len(bodies)} times (expected 2)")
            continue

        trip1 = extract_trip_count(bodies[0])
        trip2 = extract_trip_count(bodies[1])
        cfg1 = extract_lowering_config(bodies[0])
        cfg2 = extract_lowering_config(bodies[1])

        print(f"Function: @{name}")
        parse_conv_function_name(name)
        print(f"  Trip count: {trip1}  vs  {trip2}")
        print(f"  Lowering config:")
        print(f"    First : {cfg1}")
        print(f"    Second: {cfg2}")
        print()

    if errors:
        print("Errors:")
        for err in errors:
            print(f"  - {err}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python tuning_analysis.py <input_file.mlir>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        mlir_text = f.read()

    funcs = parse_functions(mlir_text)
    compare_functions(funcs)

if __name__ == '__main__':
    main()

import re
import sys
import math
from collections import defaultdict

def report(msg: str, indent: int = 0):
    """
    Centralized reporting function for consistent print formatting.
    """
    prefix = '  ' * indent
    print(f"{prefix}{msg}")


def parse_conv_function_name(name):
    """
    Parses a convolution function name and returns shapes and GEMM dims.
    """
    pattern = r'(\d+(?:x\d+)+)_(\w+?)_(\d+(?:x\d+)+)_(\w+?)_nhwf'
    match = re.search(pattern, name)
    if not match:
        return None

    input_str, input_layout, filter_str, filter_layout = match.groups()
    input_shape = list(map(int, input_str.split('x')))
    filter_shape = list(map(int, filter_str.split('x')))

    if input_layout != 'nhwc' or filter_layout != 'fhwc':
        raise ValueError(
            f"Unexpected layout: input={input_layout}, filter={filter_layout}"
        )
    if len(input_shape) != 4 or len(filter_shape) != 4:
        raise ValueError(
            f"Unexpected dimensions: input={input_shape}, filter={filter_shape}"
        )

    N, H, W, C = input_shape
    F, KH, KW, IC = filter_shape

    gemmM = N * H * W
    gemmN = F
    gemmK = KH * KW * C

    return {
        'input_shape': input_shape,
        'filter_shape': filter_shape,
        'gemmM': gemmM,
        'gemmN': gemmN,
        'gemmK': gemmK,
        'N': N, 'H': H, 'W': W, 'C': C,
        'F': F, 'KH': KH, 'KW': KW
    }


def extract_trip_count(body):
    """Extracts the total trip count from the outermost scf.forall loop."""
    forall_match = re.search(
        r'scf\.forall\s*\((.*?)\)\s*=\s*\((.*?)\)\s*to\s*\((.*?)\)\s*step\s*\((.*?)\)',
        body
    )
    if not forall_match:
        return None
    lowers = list(map(int, forall_match.group(2).split(',')))
    uppers = list(map(int, forall_match.group(3).split(',')))
    steps = list(map(int, forall_match.group(4).split(',')))
    trip_counts = [math.ceil((u - l) / s) for l, u, s in zip(lowers, uppers, steps)]
    return math.prod(trip_counts)


def extract_lowering_config(body):
    """Extracts the contents of the lowering_config attribute."""
    match = re.search(
        r'lowering_config\s*=\s*#iree_gpu\.lowering_config<\{(.*?)\}>',
        body,
        re.DOTALL
    )
    return match.group(1).strip() if match else None


def parse_functions(mlir_text):
    """Parses all functions in the MLIR file into a dict mapping name to bodies."""
    funcs = defaultdict(list)
    func_pattern = re.compile(r'func\.func\s+@([\w$.-]+)\s*\(.*?\)\s*->\s*.*?\{', re.DOTALL)
    starts = [m.start() for m in func_pattern.finditer(mlir_text)]
    for start in starts:
        name_match = func_pattern.match(mlir_text, start)
        if not name_match:
            continue
        name = name_match.group(1)
        brace_count = 0
        body_count = 2
        body_start = mlir_text.find('{', start)
        for idx in range(body_start, len(mlir_text)):
            char = mlir_text[idx]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            else:
                continue

            if brace_count == 0:
                body_count -= 1
            if body_count != 0:
                continue
            funcs[name].append(mlir_text[start:idx+1])
            break
    return funcs


def print_conv_info(name, info, indent=1):
    """Reports parsed convolution shapes and GEMM dims."""
    report(f"Input shape: {info['input_shape']}", indent)
    report(f"Filter shape: {info['filter_shape']}", indent)
    report(
        f"- gemmM = nhw = {info['N']} x {info['H']} x {info['W']} = {info['gemmM']}",
        indent
    )
    report(f"- gemmN = f = {info['F']} = {info['gemmN']}", indent)
    report(
        f"- gemmK = khw = {info['KH']} x {info['KW']} x {info['C']} = {info['gemmK']}",
        indent
    )


def print_comparison(name, trip1, trip2, gemm_info, cfg1, cfg2):
    """Reports comparison of two function variants."""
    report(f"Function: @{name}")
    report(f"Trip count: {trip1} vs {trip2}", 1)

    if gemm_info:
        print_conv_info(name, gemm_info)
        workgroup_size = gemm_info['gemmM'] * gemm_info['gemmN'] / trip2
        report(
            f"Workgroup work: {workgroup_size:.2f} (gemmM*gemmN/trip2)",
            1
        )
        gemm_size_by_512 = gemm_info['gemmM'] * gemm_info['gemmN'] / (512**2)
        report(
            f"GEMM size by 512: {gemm_size_by_512:.2f}",
            1
        )

    report("Lowering config:", 1)
    report(f"First : {cfg1}", 2)
    report(f"Second: {cfg2}", 2)
    report("", 0)


def compare_functions(funcs):
    errors = []
    print(','.join([
        'name',
        'tripcount_heuristic',
        'tripcount_tuned',
        'input_shape',
        'filter_shape',
        'gemmM',
        'gemmN',
        'gemmK'
    ]))

    for name, bodies in funcs.items():
        if len(bodies) != 2:
            errors.append(f"@{name} appeared {len(bodies)} times (expected 2)")
            continue

        trip1 = extract_trip_count(bodies[0])
        trip2 = extract_trip_count(bodies[1])
        cfg1 = extract_lowering_config(bodies[0])
        cfg2 = extract_lowering_config(bodies[1])
        gemm_info = None
        try:
            gemm_info = parse_conv_function_name(name)
        except ValueError as e:
            errors.append(str(e))

        # text output
        # print_comparison(name, trip1, trip2, gemm_info, cfg1, cfg2)

        # CSV output
        in_shape = 'x'.join(map(str, gemm_info['input_shape'])) if gemm_info else ''
        f_shape = 'x'.join(map(str, gemm_info['filter_shape'])) if gemm_info else ''
        gm = gemm_info['gemmM'] if gemm_info else ''
        gn = gemm_info['gemmN'] if gemm_info else ''
        gk = gemm_info['gemmK'] if gemm_info else ''
        print(','.join(map(str, [
            name,
            trip1,
            trip2,
            in_shape,
            f_shape,
            gm,
            gn,
            gk
        ])))

    if errors:
        report("Errors:")
        for err in errors:
            report(f"- {err}", 1)


def main():
    if len(sys.argv) != 2:
        report("Usage: python tuning_analysis.py <input_file.mlir>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        mlir_text = f.read()

    funcs = parse_functions(mlir_text)
    compare_functions(funcs)


if __name__ == '__main__':
    main()

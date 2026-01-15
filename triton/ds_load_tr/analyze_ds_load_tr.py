#!/usr/bin/env python3
"""
Analyze ds_load_tr instruction generation through the Triton compilation pipeline.

This script handles multiple input types:
1. Python files (.py) - Compiles with Triton and dumps IR
2. TTIR files (.ttir) - Lowers through full TritonGPU pipeline
3. TTGIR files (.ttgir) - Lowers to LLVM IR
4. MLIR files (.mlir) - Treated as TTGIR

Usage:
    python3 analyze_ds_load_tr.py <input_file>
    python3 analyze_ds_load_tr.py <input_file> --verbose
    python3 analyze_ds_load_tr.py <input_file> --arch gfx1250
    
    # For Python kernels:
    python3 analyze_ds_load_tr.py my_kernel.py
    
    # For IR files:
    python3 analyze_ds_load_tr.py kernel.ttgir
"""

import subprocess
import sys
import os
import re
import argparse
import tempfile
import shutil
from pathlib import Path

# Tool paths
TRITON_OPT = "/root/triton-mi450/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt"
LLC = "/opt/rocm/llvm/bin/llc"

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def run_command(cmd, capture_output=True):
    """Run a command and return stdout, stderr, and exit code."""
    result = subprocess.run(cmd, capture_output=capture_output, text=True)
    return result.stdout, result.stderr, result.returncode

def print_section(title, color=Colors.HEADER):
    """Print a section header."""
    print(f"\n{color}{'═' * 80}{Colors.ENDC}")
    print(f"{color}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{color}{'═' * 80}{Colors.ENDC}\n")

def print_subsection(title):
    """Print a subsection header."""
    print(f"\n{Colors.CYAN}{'─' * 60}{Colors.ENDC}")
    print(f"{Colors.CYAN}{title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")

def extract_layout_definitions(ir_text):
    """Extract all layout definitions from TTGIR header (lines starting with #)."""
    layout_defs = {}
    lines = ir_text.split('\n')
    
    # Skip these prefixes - they're debug info, not layouts
    skip_prefixes = ['#loc', '#di_']
    
    for line in lines:
        line = line.strip()
        # Layout definitions look like: #shared = #ttg.swizzled_shared<{...}>
        # or #mma = #ttg.amd_wmma<{...}>
        if line.startswith('#') and '=' in line:
            # Extract the layout name (e.g., #shared, #mma)
            match = re.match(r'^(#\w+)\s*=\s*(.+)$', line)
            if match:
                layout_name = match.group(1)
                layout_def = match.group(2)
                
                # Skip debug/location info
                if any(layout_name.startswith(prefix) for prefix in skip_prefixes):
                    continue
                    
                layout_defs[layout_name] = layout_def
    
    return layout_defs


def analyze_local_loads(ir_text):
    """Extract and analyze local_load operations from TTGIR."""
    local_loads = []
    lines = ir_text.split('\n')
    
    # First extract all layout definitions
    layout_defs = extract_layout_definitions(ir_text)
    
    for i, line in enumerate(lines):
        if 'local_load' in line.lower() or 'ttg.local_load' in line:
            # Find all layout references in this line (e.g., #shared, #shared1, #mma)
            layout_refs = re.findall(r'#\w+', line)
            # Filter to only those that have definitions
            layouts_used = {}
            for ref in set(layout_refs):
                if ref in layout_defs:
                    layouts_used[ref] = layout_defs[ref]
            
            local_loads.append({
                'line_num': i + 1,
                'line': line.strip(),
                'layouts': layouts_used,
            })
    
    return local_loads, layout_defs

def analyze_ds_load_tr_in_llir(ir_text):
    """Count and extract ds_load_tr instructions from LLVM IR."""
    # Pattern to match the LLVM intrinsic call form
    pattern = r'llvm\.amdgcn\.ds\.load\.tr(\d+)\.b(\d+)'
    
    results = {}
    lines = ir_text.split('\n')
    
    for line in lines:
        matches = re.findall(pattern, line, re.IGNORECASE)
        for match in matches:
            key = f"ds_load_tr{match[0]}_b{match[1]}"
            results[key] = results.get(key, 0) + 1
    
    return results

def analyze_ds_load_tr_in_asm(asm_text):
    """Count ds_load_tr instructions in assembly."""
    pattern = r'ds_load_tr\d+_b\d+|ds_read_tr\d+_b\d+'
    
    results = {}
    for line in asm_text.split('\n'):
        matches = re.findall(pattern, line, re.IGNORECASE)
        for match in matches:
            results[match] = results.get(match, 0) + 1
    
    return results

def compile_python_kernel(py_file, arch, kernel_index=None, compile_only=False):
    """Compile a Python kernel and return the dump directory.
    
    Args:
        py_file: Path to Python file containing Triton kernel
        arch: Target architecture (e.g., gfx1250)
        kernel_index: If multiple kernels/configs exist, which one to analyze (0-based).
                      If None and multiple exist, lists them and prompts user to choose.
        compile_only: If True, use wrapper to compile without running kernel
    
    Returns:
        (dump_dir, ttgir_text, llir_text, amdgcn_text)
    """
    # Convert to absolute path to avoid relative path issues with subprocess cwd
    py_file_abs = os.path.abspath(py_file)
    
    dump_dir = tempfile.mkdtemp(prefix='triton_ds_load_tr_')
    
    env = os.environ.copy()
    env['TRITON_KERNEL_DUMP'] = '1'
    env['TRITON_DUMP_DIR'] = dump_dir
    env['TRITON_ALWAYS_COMPILE'] = '1'  # Force recompilation to ensure dump is created
    
    if compile_only:
        # Use the wrapper script to compile without running
        wrapper_script = Path(__file__).parent / 'compile_only_wrapper.py'
        if not wrapper_script.exists():
            print(f"{Colors.RED}Warning: compile_only_wrapper.py not found, falling back to normal execution{Colors.ENDC}")
            cmd = ['python3', py_file_abs]
        else:
            cmd = ['python3', str(wrapper_script), py_file_abs]
            print(f"{Colors.CYAN}Using compile-only mode (no kernel execution){Colors.ENDC}")
    else:
        cmd = ['python3', py_file_abs]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=os.path.dirname(py_file_abs))
    
    if result.returncode != 0:
        print(f"{Colors.RED}Error compiling Python kernel:{Colors.ENDC}")
        print(result.stderr)
        return None, None, None, None
    
    # Find the kernel dump - handle multiple kernels from autotuning
    kernel_dirs = sorted(Path(dump_dir).iterdir())
    if not kernel_dirs:
        print(f"{Colors.YELLOW}No kernel dumps found in {dump_dir}{Colors.ENDC}")
        return None, None, None, None
    
    # If there are multiple kernel directories, list them
    if len(kernel_dirs) > 1:
        print(f"\n{Colors.YELLOW}Found {len(kernel_dirs)} kernel configurations (likely from autotuning):{Colors.ENDC}")
        for i, kdir in enumerate(kernel_dirs):
            # Try to extract config info from filename
            print(f"  [{i}] {kdir.name}")
        
        if kernel_index is not None:
            if 0 <= kernel_index < len(kernel_dirs):
                print(f"\n{Colors.CYAN}Using kernel index {kernel_index} (from --kernel-index){Colors.ENDC}")
            else:
                print(f"{Colors.RED}Invalid kernel index {kernel_index}. Valid range: 0-{len(kernel_dirs)-1}{Colors.ENDC}")
                return None, None, None, None
        else:
            # Default to first kernel but inform user
            kernel_index = 0
            print(f"\n{Colors.CYAN}Analyzing first kernel (index 0). Use --kernel-index N to select a different one.{Colors.ENDC}")
            print(f"{Colors.CYAN}Use --analyze-all to analyze all kernels.{Colors.ENDC}")
    else:
        kernel_index = 0
    
    kernel_dir = kernel_dirs[kernel_index]
    print(f"Selected kernel: {kernel_dir.name}")
    
    ttgir_file = list(kernel_dir.glob('*.ttgir'))
    llir_file = list(kernel_dir.glob('*.llir'))
    amdgcn_file = list(kernel_dir.glob('*.amdgcn'))
    
    ttgir = ttgir_file[0].read_text() if ttgir_file else None
    llir = llir_file[0].read_text() if llir_file else None
    amdgcn = amdgcn_file[0].read_text() if amdgcn_file else None
    
    return dump_dir, ttgir, llir, amdgcn

def lower_ttgir_to_llvm(ttgir_file, arch):
    """Lower TTGIR to LLVM IR using triton-opt."""
    cmd = [
        TRITON_OPT, ttgir_file,
        f'--convert-triton-amdgpu-to-llvm=arch={arch}',
        '--convert-builtin-func-to-llvm'
    ]
    
    stdout, stderr, rc = run_command(cmd)
    if rc != 0:
        return None, stderr
    return stdout, None


def analyze_all_kernels(dump_dir, args):
    """Analyze all kernel configurations from autotuning."""
    kernel_dirs = sorted(Path(dump_dir).iterdir())
    
    print_section(f"ANALYZING ALL {len(kernel_dirs)} KERNEL CONFIGURATIONS", Colors.CYAN)
    
    results = []
    for i, kernel_dir in enumerate(kernel_dirs):
        ttgir_files = list(kernel_dir.glob('*.ttgir'))
        llir_files = list(kernel_dir.glob('*.llir'))
        amdgcn_files = list(kernel_dir.glob('*.amdgcn'))
        
        ttgir = ttgir_files[0].read_text() if ttgir_files else None
        llir = llir_files[0].read_text() if llir_files else None
        amdgcn = amdgcn_files[0].read_text() if amdgcn_files else None
        
        local_loads, _ = analyze_local_loads(ttgir) if ttgir else ([], {})
        ds_load_llir = analyze_ds_load_tr_in_llir(llir) if llir else {}
        ds_load_asm = analyze_ds_load_tr_in_asm(amdgcn) if amdgcn else {}
        
        total_llir = sum(ds_load_llir.values())
        total_asm = sum(ds_load_asm.values())
        
        results.append({
            'index': i,
            'name': kernel_dir.name,
            'local_loads': len(local_loads),
            'ds_load_llir': total_llir,
            'ds_load_asm': total_asm,
            'details_llir': ds_load_llir,
            'details_asm': ds_load_asm,
        })
    
    # Print summary table
    print(f"\n{'Idx':<4} {'Kernel Config':<60} {'LocalLoads':<12} {'LLIR':<8} {'ASM':<8}")
    print("-" * 92)
    for r in results:
        name = r['name'][:58] + '..' if len(r['name']) > 60 else r['name']
        status = Colors.GREEN + "✓" + Colors.ENDC if r['ds_load_asm'] > 0 else Colors.RED + "✗" + Colors.ENDC
        print(f"{r['index']:<4} {name:<60} {r['local_loads']:<12} {r['ds_load_llir']:<8} {r['ds_load_asm']:<8} {status}")
    
    # Summary
    with_ds_load = sum(1 for r in results if r['ds_load_asm'] > 0)
    print(f"\n{Colors.CYAN}Summary: {with_ds_load}/{len(results)} configs use ds_load_tr{Colors.ENDC}")
    
    if with_ds_load < len(results):
        print(f"{Colors.YELLOW}Use --kernel-index N to investigate specific configs without ds_load_tr{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description='Analyze ds_load_tr instruction generation')
    parser.add_argument('input_file', help='Input file (.py, .ttir, .ttgir, or .mlir)')
    parser.add_argument('--arch', default='gfx1250', help='Target architecture (default: gfx1250)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--keep-dump', action='store_true', help='Keep temporary dump directory')
    parser.add_argument('--kernel-index', type=int, default=None, 
                        help='Which kernel config to analyze when autotuning creates multiple (0-based)')
    parser.add_argument('--analyze-all', action='store_true',
                        help='Analyze all kernel configs from autotuning (summary mode)')
    parser.add_argument('--compile-only', '-c', action='store_true',
                        help='Compile kernel without running it (use for slow FFM environments)')
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"{Colors.RED}Error: File not found: {args.input_file}{Colors.ENDC}")
        sys.exit(1)

    input_path = Path(args.input_file)
    input_ext = input_path.suffix.lower()
    
    print_section("DS_LOAD_TR ANALYSIS PIPELINE")
    print(f"Input file: {args.input_file}")
    print(f"Input type: {input_ext}")
    print(f"Target arch: {args.arch}")

    dump_dir = None
    ttgir_text = None
    llir_text = None
    amdgcn_text = None

    # =========================================================================
    # STAGE 0: Handle different input types
    # =========================================================================
    if input_ext == '.py':
        print_section("STAGE 0: COMPILING PYTHON KERNEL", Colors.BLUE)
        print(f"Compiling {args.input_file} with TRITON_KERNEL_DUMP=1...")
        
        if args.analyze_all:
            # Analyze all kernels mode - first compile to find how many kernels
            dump_dir, _, _, _ = compile_python_kernel(args.input_file, args.arch, kernel_index=0,
                                                       compile_only=args.compile_only)
            if dump_dir:
                analyze_all_kernels(dump_dir, args)
                if not args.keep_dump:
                    shutil.rmtree(dump_dir, ignore_errors=True)
                return
        
        dump_dir, ttgir_text, llir_text, amdgcn_text = compile_python_kernel(
            args.input_file, args.arch, kernel_index=args.kernel_index,
            compile_only=args.compile_only)
        
        if dump_dir is None:
            print(f"{Colors.RED}Failed to compile Python kernel{Colors.ENDC}")
            sys.exit(1)
        
        print(f"{Colors.GREEN}Compilation successful!{Colors.ENDC}")
        print(f"Dump directory: {dump_dir}")
        
    elif input_ext in ['.ttgir', '.mlir']:
        # Already TTGIR, just read and lower
        with open(args.input_file, 'r') as f:
            ttgir_text = f.read()
        
        print_section("STAGE 0: LOWERING TTGIR TO LLVM IR", Colors.BLUE)
        llir_text, err = lower_ttgir_to_llvm(args.input_file, args.arch)
        
        if llir_text is None:
            print(f"{Colors.RED}Error lowering TTGIR:{Colors.ENDC}")
            print(err)
            sys.exit(1)
            
    elif input_ext == '.ttir':
        print(f"{Colors.YELLOW}TTIR lowering not yet supported. Use a .ttgir or .py file.{Colors.ENDC}")
        sys.exit(1)
    else:
        # Try treating as TTGIR
        with open(args.input_file, 'r') as f:
            ttgir_text = f.read()
        
        llir_text, err = lower_ttgir_to_llvm(args.input_file, args.arch)
        if llir_text is None:
            print(f"{Colors.RED}Error processing file:{Colors.ENDC}")
            print(err)
            sys.exit(1)

    # =========================================================================
    # STAGE 1: Analyze TTGIR for local_load operations
    # =========================================================================
    print_section("STAGE 1: TTGIR ANALYSIS (local_load)", Colors.BLUE)
    
    local_loads = []
    if ttgir_text:
        local_loads, all_layouts = analyze_local_loads(ttgir_text)
        print(f"Found {Colors.GREEN}{len(local_loads)}{Colors.ENDC} local_load operations:")
        
        for i, ll in enumerate(local_loads, 1):
            line = ll['line']
            # Don't truncate - show full line
            print(f"\n  {Colors.YELLOW}[{i}]{Colors.ENDC} {line}")
            
            # Extract key info
            if '#ttg.dot_op' in ll['line']:
                match = re.search(r'opIdx = (\d)', ll['line'])
                if match:
                    op_idx = match.group(1)
                    print(f"      → Operand: {'A (opIdx=0)' if op_idx == '0' else 'B (opIdx=1)'}")
            
            # Display all layout definitions used in this local_load
            if ll.get('layouts'):
                print(f"      {Colors.CYAN}Layout definitions:{Colors.ENDC}")
                for layout_name, layout_def in sorted(ll['layouts'].items()):
                    print(f"        {Colors.GREEN}{layout_name}{Colors.ENDC} = {layout_def}")
    else:
        print(f"{Colors.YELLOW}No TTGIR available{Colors.ENDC}")

    # =========================================================================
    # STAGE 2: Analyze LLVM IR for ds_load_tr
    # =========================================================================
    print_section("STAGE 2: LLVM IR ANALYSIS (ds_load_tr)", Colors.BLUE)
    
    if llir_text:
        ds_load_tr_llir = analyze_ds_load_tr_in_llir(llir_text)
        
        if ds_load_tr_llir:
            total = sum(ds_load_tr_llir.values())
            print(f"{Colors.GREEN}DS_LOAD_TR instructions in LLVM IR:{Colors.ENDC}")
            for instr, count in sorted(ds_load_tr_llir.items()):
                print(f"  • {instr}: {Colors.BOLD}{count}{Colors.ENDC}")
            print(f"\n  {Colors.BOLD}Total: {total} instructions{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}No ds_load_tr instructions in LLVM IR{Colors.ENDC}")
            
            # Check what loads are used instead
            load_count = llir_text.lower().count('llvm.load')
            print(f"  → Using regular loads instead: {load_count} llvm.load operations")
        
        if args.verbose:
            print_subsection("LLVM IR (ds_load_tr lines)")
            for line in llir_text.split('\n'):
                if 'ds.load.tr' in line.lower():
                    print(f"  {line.strip()}")
    else:
        print(f"{Colors.YELLOW}No LLVM IR available{Colors.ENDC}")
        ds_load_tr_llir = {}

    # =========================================================================
    # STAGE 3: Analyze Assembly for ds_load_tr
    # =========================================================================
    print_section("STAGE 3: ASSEMBLY ANALYSIS (ds_load_tr in ISA)", Colors.BLUE)
    
    if amdgcn_text:
        ds_load_tr_asm = analyze_ds_load_tr_in_asm(amdgcn_text)
        
        if ds_load_tr_asm:
            total = sum(ds_load_tr_asm.values())
            print(f"{Colors.GREEN}DS_LOAD_TR instructions in assembly:{Colors.ENDC}")
            for instr, count in sorted(ds_load_tr_asm.items()):
                print(f"  • {instr}: {Colors.BOLD}{count}{Colors.ENDC}")
            print(f"\n  {Colors.BOLD}Total: {total} instructions{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}No ds_load_tr instructions in assembly{Colors.ENDC}")
        
        if args.verbose:
            print_subsection("Assembly (ds_load_tr lines)")
            for line in amdgcn_text.split('\n'):
                if 'ds_load_tr' in line.lower() or 'ds_read_tr' in line.lower():
                    print(f"  {line.strip()}")
    else:
        print(f"{Colors.YELLOW}No assembly available (compile .py file to get assembly){Colors.ENDC}")
        ds_load_tr_asm = {}

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("SUMMARY", Colors.GREEN)
    
    print(f"Input: {args.input_file}")
    print(f"  • local_load operations: {len(local_loads) if ttgir_text else 'N/A'}")
    
    if ds_load_tr_llir:
        total_llir = sum(ds_load_tr_llir.values())
        print(f"  • ds_load_tr in LLVM IR: {Colors.GREEN}{total_llir}{Colors.ENDC}")
        for instr, count in sorted(ds_load_tr_llir.items()):
            print(f"    - {instr}: {count}")
    else:
        print(f"  • ds_load_tr in LLVM IR: {Colors.RED}0{Colors.ENDC}")
    
    if amdgcn_text and ds_load_tr_asm:
        total_asm = sum(ds_load_tr_asm.values())
        print(f"  • ds_load_tr in assembly: {Colors.GREEN}{total_asm}{Colors.ENDC}")
        for instr, count in sorted(ds_load_tr_asm.items()):
            print(f"    - {instr}: {count}")
    
    # Final verdict
    if ds_load_tr_llir:
        print(f"\n{Colors.GREEN}✓ ds_load_tr optimization is being applied!{Colors.ENDC}")
    else:
        print(f"\n{Colors.YELLOW}✗ ds_load_tr optimization NOT applied{Colors.ENDC}")
        print("  Possible reasons:")
        print("    1. Shared memory layout already matches register layout (no transpose needed)")
        print("    2. Padding interval too small (< 8 elements for bf16/f16)")
        print("    3. Target layout not compatible with ds_load_tr")
        print("    4. Architecture doesn't support ds_load_tr")

    # Cleanup
    if dump_dir and not args.keep_dump:
        shutil.rmtree(dump_dir, ignore_errors=True)
    elif dump_dir:
        print(f"\n{Colors.CYAN}Dump directory kept at: {dump_dir}{Colors.ENDC}")

if __name__ == '__main__':
    main()

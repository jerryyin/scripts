#!/usr/bin/env python3
"""
Wrapper to compile any Triton kernel without running it.

Usage:
    python compile_only_wrapper.py <kernel_file.py> [options]

This script:
1. Parses the kernel file's AST to extract only kernel functions
2. For JIT kernels: finds invocations to extract constexpr values
3. Creates a minimal module with just the kernels
4. Compiles them using .warmup() (no execution)

This avoids running module-level code that might execute the kernels.
"""

import argparse
import ast
import os
import sys
import tempfile
import importlib.util
import inspect
import re

# Ensure triton is available
try:
    import triton
    import triton.language as tl
    import torch
    from triton.runtime.jit import MockTensor, JITFunction
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure triton and torch are installed")
    sys.exit(1)


class KernelExtractor(ast.NodeVisitor):
    """Extract @triton.jit and @triton.autotune decorated functions from AST."""
    
    def __init__(self, source_lines):
        self.source_lines = source_lines
        self.kernels = []
        self.imports = []
        self.helper_funcs = []
        
    def visit_Import(self, node):
        self.imports.append(ast.unparse(node))
        
    def visit_ImportFrom(self, node):
        self.imports.append(ast.unparse(node))
        
    def visit_FunctionDef(self, node):
        is_jit = False
        is_autotune = False
        
        for decorator in node.decorator_list:
            decorator_str = ast.unparse(decorator)
            if 'triton.jit' in decorator_str or '@jit' in decorator_str:
                is_jit = True
            if 'triton.autotune' in decorator_str or '@autotune' in decorator_str:
                is_autotune = True
                
        if is_jit or is_autotune:
            start_line = node.lineno - 1
            end_line = node.end_lineno
            
            for d in node.decorator_list:
                start_line = min(start_line, d.lineno - 1)
            
            code = '\n'.join(self.source_lines[start_line:end_line])
            
            self.kernels.append({
                'name': node.name,
                'is_autotuned': is_autotune,
                'code': code,
                'start_line': start_line,
                'end_line': end_line,
            })
            
            if not is_autotune:
                self.helper_funcs.append(node.name)


class KernelInvocationFinder(ast.NodeVisitor):
    """Find kernel invocations like kernel[grid](args) or kernel.warmup(args)."""
    
    def __init__(self, kernel_name):
        self.kernel_name = kernel_name
        self.invocations = []
        
    # These are ALWAYS runtime/compile options passed to warmup(), never kernel constexprs
    # Note: num_stages CAN be a kernel constexpr, but num_warps/num_ctas are always compile options
    SKIP_KWARGS = {'grid', 'enable_fp_fusion', 'extern_libs', 'stream', 
                   'warmup', 'device', 'device_type', 'num_warps', 'num_ctas'}
    
    def _extract_kwargs(self, node):
        """Extract keyword arguments from a Call node."""
        kwargs = {}
        for kw in node.keywords:
            if kw.arg and kw.arg not in self.SKIP_KWARGS:
                try:
                    value_str = ast.unparse(kw.value)
                    try:
                        # Try literal eval for simple values (numbers, strings, etc.)
                        value = ast.literal_eval(value_str)
                        kwargs[kw.arg] = value
                    except (ValueError, SyntaxError):
                        # Not a literal - it's a variable reference
                        # Store as None to indicate we need to find the value elsewhere
                        kwargs[kw.arg] = None
                except:
                    pass
        return kwargs
        
    def visit_Call(self, node):
        invocation = None
        
        # Pattern 1: kernel[grid](args) - Subscript call
        if isinstance(node.func, ast.Subscript):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == self.kernel_name:
                    invocation = {
                        'lineno': node.lineno,
                        'type': 'grid_call',
                        'positional_args': [ast.unparse(arg) for arg in node.args],
                        'keyword_args': self._extract_kwargs(node),
                    }
        
        # Pattern 2: kernel.warmup(args) - Method call
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr == 'warmup':
                # Check if it's our kernel
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == self.kernel_name:
                        invocation = {
                            'lineno': node.lineno,
                            'type': 'warmup_call',
                            'positional_args': [ast.unparse(arg) for arg in node.args],
                            'keyword_args': self._extract_kwargs(node),
                        }
        
        # Pattern 3: wrapper_function(kernel, args) where kernel name is passed
        # This is harder to detect, skip for now
        
        if invocation:
            self.invocations.append(invocation)
        
        self.generic_visit(node)


class VariableValueFinder(ast.NodeVisitor):
    """Try to find literal values assigned to variables."""
    
    def __init__(self):
        self.assignments = {}  # var_name -> value (if literal)
        
    def visit_Assign(self, node):
        # Simple assignment: x = 1024
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            try:
                value_str = ast.unparse(node.value)
                value = ast.literal_eval(value_str)
                self.assignments[var_name] = value
            except:
                pass
        self.generic_visit(node)
    
    def visit_For(self, node):
        # For loop: for BLOCK_SIZE in [512, 1024, 2048]:
        # Take the first value
        if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.List):
            var_name = node.target.id
            if node.iter.elts:
                try:
                    value = ast.literal_eval(ast.unparse(node.iter.elts[0]))
                    self.assignments[var_name] = value
                except:
                    pass
        self.generic_visit(node)


def find_kernel_invocations(source, kernel_name):
    """Find all invocations of a kernel and extract constexpr values."""
    try:
        tree = ast.parse(source)
        
        # First, find variable assignments
        var_finder = VariableValueFinder()
        var_finder.visit(tree)
        
        # Then find kernel invocations
        finder = KernelInvocationFinder(kernel_name)
        finder.visit(tree)
        
        # Resolve variable references in invocations
        for inv in finder.invocations:
            for key, value in list(inv['keyword_args'].items()):
                if value is None:
                    # Try to find the variable value
                    if key in var_finder.assignments:
                        inv['keyword_args'][key] = var_finder.assignments[key]
                    elif key.upper() in var_finder.assignments:
                        inv['keyword_args'][key] = var_finder.assignments[key.upper()]
                    else:
                        # Can't resolve - remove it
                        del inv['keyword_args'][key]
        
        return finder.invocations
    except SyntaxError:
        return []


def warmup_autotuned_kernel(kernel, kernel_name):
    """Compile an autotuned kernel for each config."""
    
    print(f"  Kernel: {kernel_name} (autotuned)")
    
    configs = getattr(kernel, 'configs', [])
    print(f"  Found {len(configs)} autotune configurations")
    
    for i, config in enumerate(configs):
        print(f"    [{i}] Config: {dict(config.kwargs)}")
    
    try:
        underlying_fn = kernel.fn.fn if hasattr(kernel.fn, 'fn') else kernel.fn
        sig = inspect.signature(underlying_fn)
        params = list(sig.parameters.values())
        
        warmup_args = []
        warmup_kwargs = {}
        
        for p in params:
            param_name = p.name
            annot_str = str(p.annotation).lower()
            is_constexpr = 'constexpr' in annot_str
            
            if is_constexpr:
                # For autotuned kernels, skip constexprs that are in configs
                if not any(param_name in c.kwargs for c in configs):
                    # This constexpr isn't in configs - try common defaults
                    if param_name.upper() == 'ACTIVATION':
                        warmup_kwargs[param_name] = ''
                    elif isinstance(p.default, (int, float, str, bool)):
                        warmup_kwargs[param_name] = p.default
                    else:
                        warmup_kwargs[param_name] = 1
            elif 'ptr' in param_name.lower():
                warmup_args.append(MockTensor(torch.float16))
            elif 'stride' in param_name.lower():
                warmup_args.append(128)
            else:
                warmup_args.append(128)
        
        print(f"    Warmup args: {len(warmup_args)} positional, kwargs: {list(warmup_kwargs.keys())}")
        
        results = kernel.warmup(*warmup_args, grid=(1,), **warmup_kwargs)
        
        if isinstance(results, list):
            print(f"    ✓ Compiled {len(results)} configurations")
        else:
            print(f"    ✓ Compiled")
            
        return results
            
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def warmup_jit_kernel(kernel, kernel_name, constexpr_args):
    """Compile a regular @triton.jit kernel using extracted constexpr values."""
    
    print(f"  Kernel: {kernel_name} (jit)")
    print(f"    Using constexprs: {constexpr_args}")
    
    try:
        sig = inspect.signature(kernel.fn)
        params = list(sig.parameters.values())
        
        warmup_args = []
        warmup_kwargs = {}
        missing_constexprs = []
        
        for p in params:
            param_name = p.name
            is_constexpr = 'constexpr' in str(p.annotation).lower()
            
            if is_constexpr:
                if param_name in constexpr_args:
                    warmup_kwargs[param_name] = constexpr_args[param_name]
                elif p.default != inspect.Parameter.empty:
                    warmup_kwargs[param_name] = p.default
                else:
                    missing_constexprs.append(param_name)
            elif 'ptr' in param_name.lower():
                warmup_args.append(MockTensor(torch.float16))
            elif 'stride' in param_name.lower():
                warmup_args.append(128)
            else:
                warmup_args.append(128)
        
        if missing_constexprs:
            print(f"    ✗ Missing constexpr values: {missing_constexprs}")
            print(f"      Could not find kernel invocation with these values.")
            return None
        
        result = kernel.warmup(*warmup_args, grid=(1,), **warmup_kwargs)
        print(f"    ✓ Compiled")
        return result
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Compile Triton kernels without running them')
    parser.add_argument('kernel_file', help='Python file containing Triton kernel(s)')
    parser.add_argument('--kernel-name', '-n', help='Specific kernel to compile (default: all)')
    parser.add_argument('--invoke-index', '-i', type=int, default=0,
                        help='Which kernel invocation to use for constexpr values (default: 0, first)')
    parser.add_argument('--constexpr', '-c', action='append', default=[],
                        metavar='NAME=VALUE',
                        help='Manually specify constexpr values (can be used multiple times)')
    parser.add_argument('--extract-only', action='store_true', 
                        help='Only extract kernel code, don\'t compile')
    parser.add_argument('--show-invocations', action='store_true',
                        help='Show all found kernel invocations')
    args = parser.parse_args()
    
    # Parse manual constexpr overrides
    manual_constexprs = {}
    for spec in args.constexpr:
        if '=' in spec:
            name, value = spec.split('=', 1)
            try:
                manual_constexprs[name] = ast.literal_eval(value)
            except:
                manual_constexprs[name] = value
    args.manual_constexprs = manual_constexprs

    if not os.path.exists(args.kernel_file):
        print(f"Error: File not found: {args.kernel_file}")
        sys.exit(1)

    print(f"Loading: {args.kernel_file}")
    
    with open(args.kernel_file, 'r') as f:
        source = f.read()
        source_lines = source.split('\n')
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Syntax error in {args.kernel_file}: {e}")
        sys.exit(1)
    
    extractor = KernelExtractor(source_lines)
    extractor.visit(tree)
    
    if not extractor.kernels:
        print("No @triton.jit or @triton.autotune functions found")
        sys.exit(1)
    
    print(f"Found {len(extractor.kernels)} kernel(s):")
    for k in extractor.kernels:
        kind = "autotuned" if k['is_autotuned'] else "jit"
        print(f"  • {k['name']} ({kind})")
    
    # For JIT kernels, find invocations to extract constexpr values
    kernel_invocations = {}
    for k in extractor.kernels:
        if not k['is_autotuned']:
            invocations = find_kernel_invocations(source, k['name'])
            kernel_invocations[k['name']] = invocations
            
            if invocations:
                print(f"\n  Found {len(invocations)} invocation(s) of {k['name']}:")
                for i, inv in enumerate(invocations):
                    marker = " ← using" if i == args.invoke_index else ""
                    print(f"    [{i}] line {inv['lineno']}: {inv['keyword_args']}{marker}")
                
                if len(invocations) > 1 and args.invoke_index == 0:
                    print(f"    (Use --invoke-index N to select a different invocation)")
            else:
                print(f"\n  ⚠ No invocations found for {k['name']} - cannot extract constexpr values")
    
    if args.show_invocations:
        return
    
    if args.extract_only:
        print("\n--- Extracted kernel code ---")
        for k in extractor.kernels:
            print(f"\n# {k['name']}\n{k['code']}")
        return
    
    if args.kernel_name:
        extractor.kernels = [k for k in extractor.kernels if k['name'] == args.kernel_name]
        if not extractor.kernels:
            print(f"Kernel '{args.kernel_name}' not found")
            sys.exit(1)
    
    # Create temporary module
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import triton\n")
        tmp.write("import triton.language as tl\n")
        tmp.write("import torch\n\n")
        
        # Write helper functions first
        for k in extractor.kernels:
            if not k['is_autotuned']:
                tmp.write(k['code'] + "\n\n")
        
        # Check if we need autotune config functions
        if any(k['is_autotuned'] for k in extractor.kernels):
            if 'get_hip_autotune_config' in source or 'get_autotune_config' in source:
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name in [
                        'get_hip_autotune_config', 'get_cuda_autotune_config', 
                        'get_autotune_config', 'is_cuda'
                    ]:
                        start = node.lineno - 1
                        end = node.end_lineno
                        tmp.write('\n'.join(source_lines[start:end]) + "\n\n")
        
        # Write main kernels
        for k in extractor.kernels:
            if k['is_autotuned']:
                tmp.write(k['code'] + "\n\n")
        
        tmp_path = tmp.name
    
    print(f"\nCreated temporary module: {tmp_path}")
    
    try:
        spec = importlib.util.spec_from_file_location("extracted_kernels", tmp_path)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, os.path.dirname(os.path.abspath(args.kernel_file)))
        spec.loader.exec_module(module)
        
        print("\nCompiling kernels...")
        
        for k in extractor.kernels:
            if not hasattr(module, k['name']):
                print(f"  Warning: {k['name']} not found in module after import")
                continue
                
            kernel = getattr(module, k['name'])
            
            if k['is_autotuned']:
                warmup_autotuned_kernel(kernel, k['name'])
            else:
                # Skip helper functions
                if k['name'] in extractor.helper_funcs and len(extractor.kernels) > 1:
                    print(f"  Skipping helper function: {k['name']}")
                    continue
                
                # Get constexpr values from invocation
                invocations = kernel_invocations.get(k['name'], [])
                if not invocations:
                    print(f"  ✗ Cannot compile {k['name']}: no invocations found to extract constexpr values")
                    print(f"    The kernel must be invoked somewhere in the file.")
                    continue
                
                if args.invoke_index >= len(invocations):
                    print(f"  ✗ Invalid --invoke-index {args.invoke_index}, only {len(invocations)} invocations found")
                    continue
                
                constexpr_args = invocations[args.invoke_index]['keyword_args'].copy()
                # Apply manual overrides
                constexpr_args.update(args.manual_constexprs)
                warmup_jit_kernel(kernel, k['name'], constexpr_args)
        
        print("\nCompilation complete!")
        print("IR artifacts should be in TRITON_DUMP_DIR if TRITON_KERNEL_DUMP=1 was set")
        
    except Exception as e:
        print(f"Error during compilation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    main()

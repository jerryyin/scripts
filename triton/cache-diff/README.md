# triton/cache-diff — did a source change change the compiled kernels?

`triton_cache_diff.py BEFORE_DIR AFTER_DIR [--ext hsaco,amdgcn] [--json OUT]` compares the compiled kernels in two Triton/Gluon cache directories (`TRITON_CACHE_DIR`), for example a cache filled before and after a refactor that should not change code generation.

It groups the files by kernel name and extension and compares, per name, the multiset of SHA-256 digests on each side. `.amdgcn` is also compared after dropping debug-location and comment lines, and `.hsaco` by its `.text` and `.rodata` sections only, so a moved source line is reported as `same-code` rather than `differs`. One line per kernel and extension: `identical`, `same-code`, `differs`, `only-before` or `only-after`. Exit status 0 when nothing differs or is one-sided.

Stdlib only. Tests (no GPU): `python3 -m pytest triton/cache-diff/test_triton_cache_diff.py`.

First written to show that porting the Gluon Petit fusions onto a newer TokenSpeed left every default-off kernel unchanged.

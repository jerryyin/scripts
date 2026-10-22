# Standalone tools only

Each top-level Python file must serve one named use case and work when copied out of this repository. Document its purpose, inputs, invocation, and external prerequisites in `README.md`. Review additions for one coherent purpose; a filename or passing test alone does not establish that.

- Do not import, load, or invoke another tool or repository module. Do not use dynamic imports or alter Python's import path to bypass this boundary. Caller-supplied scripts and commands are inputs, not dependencies of the tool.
- Do not add a shared library, package initializer, umbrella command, tool registry, common configuration, or shared runtime state. Keep tool-specific state at caller-specified locations or in temporary storage.
- Keep the directory flat: named `.py` utilities, this file, `README.md`, and `tests/`. Tests may load the utility they verify; runtime tools must not load tests.
- Prefer an existing subject directory when one owns the work, such as `bench/`, `profiling/`, or `containers/`. If utilities need to cooperate or share implementation, move that family to a dedicated subject directory instead of coupling tools here.
- Add focused tests under `tests/`, including a test that exercises the tool copied outside the checkout. Keep prerequisites per tool in the README; do not make unrelated tools require each other's packages.

Run `python -B -m unittest discover -s tools/tests -v`. The structure test requires documentation and a dedicated test file for each tool, and rejects nested runtime directories, repository/sibling imports, dynamic import helpers, import-path access, and literal sibling-file dependencies. GitHub Actions runs the portable checks on pushes and pull requests touching tools. Review must still catch indirect or dynamically constructed dependencies and judge whether the use case is dedicated. Run Windows/WSL integration checks on a host with both environments before changing that utility.

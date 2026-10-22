() {
  local root=${IREE_DIR:-${DEVENV_WORKSPACE_ROOT:-$HOME}/iree}
  [[ -d $root ]] || return 0
  local directory
  for directory in "$root/build/model/tracy" "$root/third_party/tracy/csvexport/build" \
      "$root/third_party/llvm-project/build/bin" "$root/build/dbg/tools"; do
    if [[ -d $directory && ":$PATH:" != *":$directory:"* ]]; then
      export PATH="$directory${PATH:+:$PATH}"
    fi
  done
  for directory in "$root/build/model/runtime/bindings/python" "$root/build/model/compiler/bindings/python"; do
    if [[ -d $directory && ":$PYTHONPATH:" != *":$directory:"* ]]; then
      export PYTHONPATH="$directory${PYTHONPATH:+:$PYTHONPATH}"
    fi
  done
  if [[ ":$GLIBC_TUNABLES:" != *:glibc.rtld.optional_static_tls=* ]]; then
    export GLIBC_TUNABLES="${GLIBC_TUNABLES:+$GLIBC_TUNABLES:}glibc.rtld.optional_static_tls=4096"
  fi
}

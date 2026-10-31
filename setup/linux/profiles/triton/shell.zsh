() {
  local root=${TRITON_DIR:-${DEVENV_WORKSPACE_ROOT:-$HOME}/triton-mi450}
  if [[ -d $root/python && ":$PYTHONPATH:" != *":$root/python:"* ]]; then
    export PYTHONPATH="$root/python${PYTHONPATH:+:$PYTHONPATH}"
  fi
  if [[ -z $TRITON_MI450_LLVM_DOWNLOAD_GITHUB_TOKEN \
      && -r $HOME/vault-work/gh_token_amdeng.txt ]]; then
    local token="$(<"$HOME/vault-work/gh_token_amdeng.txt")"
    export TRITON_MI450_LLVM_DOWNLOAD_GITHUB_TOKEN="${token//[[:space:]]/}"
  fi
  [[ -z $_TRITON_MODEL_PKG ]] || return 0
  local directory=${${(%):-%x}:A:h}/../../../../am
  directory=${directory:A}
  local model=${AM_FFM_DIR:-/am-ffm}
  if [[ -z $AM_FFM_DIR && ! -d $model ]]; then
    model=/ffm
  fi
  local environment
  # Preserve the interactive FFM default when a package contains both backends.
  if [[ -f $model/ffmlite_env.sh ]]; then
    environment=$model/ffmlite_env.sh
  elif [[ -f $model/am_env.sh ]]; then
    environment=$model/am_env.sh
  else
    return 0
  fi
  source "$environment" || return
  if [[ -d /opt/rocm/lib && ":$LD_LIBRARY_PATH:" != *:/opt/rocm/lib:* ]]; then
    export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  if [[ ":$PYTHONPATH:" != *":$directory:"* ]]; then
    export PYTHONPATH="$directory${PYTHONPATH:+:$PYTHONPATH}"
  fi
  if [[ ",$PYTEST_PLUGINS," != *,ffm_teardown,* ]]; then
    export PYTEST_PLUGINS="${PYTEST_PLUGINS:+$PYTEST_PLUGINS,}ffm_teardown"
  fi
  export _TRITON_MODEL_PKG=$model
}

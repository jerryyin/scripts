# Load only the selected project's shell environment.
() {
  local directory=${${(%):-%x}:A:h}
  local selection=${XDG_CONFIG_HOME:-$HOME/.config}/scripts/setup-profile
  local profile=${SETUP_PROFILE:-}
  if [[ -z $profile && -r $selection ]]; then
    profile="$(<$selection)"
  fi
  case ${profile:-base} in
    base) ;;
    iree|triton) source "$directory/profiles/$profile/shell.zsh" || return ;;
    *) print -u2 "Unknown shell setup profile: $profile"; return 1 ;;
  esac
  return 0
}

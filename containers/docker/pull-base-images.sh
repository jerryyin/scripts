#!/bin/bash
# Pull the FROM images selected by this checkout's Compose services.
set -euo pipefail
if (( $# == 0 )); then set -- triton triton-mi450; fi
script_dir=$(dirname "$(readlink -f "$0")")
compose_file="${COMPOSE_FILE:-$script_dir/docker-compose.yml}"
if [[ ! -f "$compose_file" ]]; then
    echo "Compose file not found: $compose_file" >&2
    exit 1
fi
docker compose version >/dev/null
command -v python3 >/dev/null

profile_flags=()
for service in "$@"; do profile_flags+=(--profile "$service"); done
# Rendering several profiles requires bypassing their shared container-name check.
config=$(docker compose -f "$compose_file" "${profile_flags[@]}" config --no-consistency --format json)
for service in "$@"; do
    image=$(printf '%s' "$config" | python3 -c '
import json, sys
model = json.load(sys.stdin)
print(model.get("services", {}).get(sys.argv[1], {}).get("build", {}).get("args", {}).get("BASE_IMAGE", ""))
' "$service")
    if [[ -z "$image" ]]; then
        echo "Could not resolve BASE_IMAGE for $service" >&2
        exit 1
    fi
    echo "Pulling $image (base image for $service)..."
    docker pull "$image"
done

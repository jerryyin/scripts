#!/bin/bash

set -e

DOCKERFILE=${1:-Dockerfile}

if [ ! -f "$DOCKERFILE" ]; then
    echo "Dockerfile not found: $DOCKERFILE"
    exit 1
fi

function handle_from() {
    echo "# FROM $1 (Base image, not applicable in bash context)"
}

function handle_run() {
    eval "$@"
}

function handle_env() {
    if [[ "$1" == *"="* ]]; then
        export "$1"
    else
        export "$1=$2"
    fi
}

function handle_workdir() {
    cd "$1"
}

function handle_copy() {
    cp -R "$1" "$2"
}

function handle_add() {
    if [[ "$1" == http://* || "$1" == https://* ]]; then
        wget -O "$2" "$1"
    else
        cp -R "$1" "$2"
    fi
}

function handle_expose() {
    echo "# EXPOSE $1 (Port exposure, not applicable in bash context)"
}

function handle_cmd() {
    echo "# CMD $@ (Command to run at the end)"
    eval "$@"
}

function handle_entrypoint() {
    echo "# ENTRYPOINT $@ (Entry point, will be executed at the end)"
    ENTRYPOINT_CMD="$@"
}

while read -r line || [ -n "$line" ]; do
    instruction=$(echo "$line" | awk '{print $1}')
    args=$(echo "$line" | cut -d' ' -f2-)
    
    case $instruction in
        FROM)
            handle_from "$args"
            ;;
        RUN)
            handle_run "$args"
            ;;
        ENV)
            handle_env $args
            ;;
        WORKDIR)
            handle_workdir "$args"
            ;;
        COPY)
            handle_copy $args
            ;;
        ADD)
            handle_add $args
            ;;
        EXPOSE)
            handle_expose "$args"
            ;;
        CMD)
            handle_cmd $args
            ;;
        ENTRYPOINT)
            handle_entrypoint $args
            ;;
        *)
            echo "# Unknown or unsupported instruction: $line"
            ;;
    esac
done < "$DOCKERFILE"

# Execute ENTRYPOINT if set
if [ ! -z "$ENTRYPOINT_CMD" ]; then
    eval "$ENTRYPOINT_CMD"
fi

# Base image
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

# Reset ENTRYPOINT from any base image so docker-compose command works correctly.
ENTRYPOINT []

USER root
WORKDIR /root

RUN printf 'Acquire::http::proxy "%s";\nAcquire::https::proxy "%s";\n' "${HTTP_PROXY:-}" "${HTTPS_PROXY:-}" > /etc/apt/apt.conf

WORKDIR /root

ARG SERVICE_NAME
ARG SETUP_PROFILE
ARG SCRIPTS_REPOSITORY=https://github.com/jerryyin/scripts.git
ARG SCRIPTS_REF
# SERVICE_NAME remains supported for existing docker build commands. Profiles
# belong to scripts/setup; Docker selects the build phase (no credential activation).
# SCRIPTS_REF optionally selects a branch/tag for reproducible builds.
RUN apt-get update && apt-get -y install git ca-certificates sudo wget curl gpg lsb-release && \
    if [ -n "$SCRIPTS_REF" ]; then \
        git clone --depth 1 --branch "$SCRIPTS_REF" "$SCRIPTS_REPOSITORY" /root/scripts; \
    else \
        git clone --depth 1 "$SCRIPTS_REPOSITORY" /root/scripts; \
    fi && \
    git -C /root/scripts remote set-url origin git@github.com:jerryyin/scripts.git && \
    bash /root/scripts/setup/linux/setup.sh "${SETUP_PROFILE:-${SERVICE_NAME:-base}}" --phase build

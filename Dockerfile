# KISS Sorcar in a container: the kiss-web daemon and its webapp, nothing
# else.  scripts/docker-startup.sh (the entrypoint) clones the repository,
# runs its install.sh and starts kiss-web in the foreground on port 8787.
FROM debian:bookworm-slim

# What install.sh needs on a bare Debian: bash, git and curl for the clone
# and the uv/cloudflared downloads, perl for its new-session re-exec and
# update lock, ca-certificates for every https download, sudo so the
# Playwright step can install Chromium's system libraries, build tools for
# the few Python packages uv has to compile.  python3 runs
# src/kiss/scripts/remote_config.py before the app's venv exists.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash git curl ca-certificates perl sudo procps \
    python3 build-essential libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# uv builds the app's Python environment (install.sh would download it
# itself; baking it in keeps the first start shorter).  Releases before
# 0.11.15 carry two file-write/deletion advisories.
ENV UV_VERSION=0.12.19
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
        x86_64)  TARGET="x86_64-unknown-linux-gnu" ;; \
        aarch64) TARGET="aarch64-unknown-linux-gnu" ;; \
        *) echo "Unsupported arch: $ARCH" && exit 1 ;; \
    esac && \
    curl -fsSL "https://releases.astral.sh/github/uv/releases/download/${UV_VERSION}/uv-${TARGET}.tar.gz" \
    | tar xz -C /usr/local/bin --strip-components=1

# The daemon runs as an ordinary user; passwordless sudo is for
# `playwright install-deps` only.
RUN useradd --create-home --shell /bin/bash sorcar \
    && echo "sorcar ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/sorcar

COPY --chmod=755 scripts/docker-startup.sh /usr/local/bin/docker-startup.sh

USER sorcar
WORKDIR /home/sorcar
RUN git config --global init.defaultBranch main \
    && git config --global user.email "sorcar@kiss-sorcar" \
    && git config --global user.name "KISS Sorcar"

# kiss-web: HTTPS and WebSocket on one port.
EXPOSE 8787

ENTRYPOINT ["/usr/local/bin/docker-startup.sh"]

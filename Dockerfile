FROM codercom/code-server:latest

USER root

# System dependencies
#
# python3 + python-is-python3 are required by
# src/kiss/agents/vscode/copy-kiss.sh (invoked from the extension's
# `vscode:prepublish` script) — without them `npm run package` silently
# fails and the container falls back to a stale committed VSIX.
RUN apt-get update && apt-get install -y \
    git curl wget build-essential libssl-dev \
    ca-certificates gnupg sudo \
    python3 python3-venv python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager)
ENV UV_VERSION=0.11.2
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
        x86_64)  TARGET="x86_64-unknown-linux-gnu" ;; \
        aarch64) TARGET="aarch64-unknown-linux-gnu" ;; \
        *) echo "Unsupported arch: $ARCH" && exit 1 ;; \
    esac && \
    curl -fsSL "https://releases.astral.sh/github/uv/releases/download/${UV_VERSION}/uv-${TARGET}.tar.gz" \
    | tar xz -C /usr/local/bin --strip-components=1

# Passwordless sudo for coder (needed for playwright install-deps)
RUN echo "coder ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/coder

# Repo directory owned by coder
RUN mkdir -p /home/kiss && chown coder:coder /home/kiss

# Startup script
COPY --chmod=755 scripts/docker-startup.sh /usr/local/bin/docker-startup.sh

USER coder

RUN git config --global init.defaultBranch main \
    && git config --global user.email "coder@kiss-sorcar" \
    && git config --global user.name "KISS Sorcar"

WORKDIR /home/kiss
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/docker-startup.sh"]
# --disable-workspace-trust skips the "Do you trust the authors of the files
#   in this folder?" dialog that otherwise blocks the extension from
#   activating on first launch.
# --enable-proposed-api ksenxx.kiss-sorcar allows the extension to use the
#   `contribSourceControlInputBoxMenu` proposed API declared in
#   src/kiss/agents/vscode/package.json — future-proof against code-server
#   tightening proposed-API gating.
CMD ["--bind-addr", "0.0.0.0:8080", "--auth", "none", \
     "--disable-workspace-trust", \
     "--enable-proposed-api", "ksenxx.kiss-sorcar", \
     "/home/kiss"]

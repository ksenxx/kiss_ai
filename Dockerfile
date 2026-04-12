FROM codercom/code-server:latest

USER root

# Install system dependencies (Playwright deps added after venv setup below)
RUN apt-get update && apt-get install -y \
    git curl wget build-essential libssl-dev \
    ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install uv (detect architecture)
ENV UV_VERSION=0.11.2
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
        x86_64)  TARGET="x86_64-unknown-linux-gnu" ;; \
        aarch64) TARGET="aarch64-unknown-linux-gnu" ;; \
        *) echo "Unsupported arch: $ARCH" && exit 1 ;; \
    esac && \
    curl -fsSL "https://releases.astral.sh/github/uv/releases/download/${UV_VERSION}/uv-${TARGET}.tar.gz" \
    | tar xz -C /usr/local/bin --strip-components=1

# Copy the KISS project
COPY --chown=coder:coder . /home/coder/kiss

USER coder

# Configure git defaults (tests assume "main" as default branch)
RUN git config --global init.defaultBranch main \
    && git config --global user.email "coder@kiss-sorcar" \
    && git config --global user.name "KISS Sorcar"

# Set up Python environment
WORKDIR /home/coder/kiss
RUN uv venv --python 3.13 && uv sync

# Install Playwright Chromium system dependencies (needs root for apt-get)
USER root
RUN /home/coder/kiss/.venv/bin/playwright install-deps chromium \
    && rm -rf /var/lib/apt/lists/*
USER coder

# Install Playwright Chromium browser binary
RUN uv run playwright install chromium

# Install the VSIX extension into code-server
RUN code-server --install-extension /home/coder/kiss/src/kiss/agents/vscode/kiss-sorcar.vsix

# Create a demo workspace
RUN mkdir -p /home/coder/workspace

# Set environment
ENV KISS_PROJECT_PATH=/home/coder/kiss

WORKDIR /home/coder/workspace

EXPOSE 8080

# Override base image ENTRYPOINT (which bakes in "--bind-addr 0.0.0.0:8080 .")
# so our CMD controls all code-server arguments cleanly.
ENTRYPOINT ["/usr/bin/entrypoint.sh"]
CMD ["--bind-addr", "0.0.0.0:8080", "--auth", "none", "/home/coder/workspace"]

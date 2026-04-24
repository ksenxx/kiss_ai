FROM codercom/code-server:latest

USER root

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential libssl-dev \
    ca-certificates gnupg sudo \
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
CMD ["--bind-addr", "0.0.0.0:8080", "--auth", "none", "/home/kiss"]

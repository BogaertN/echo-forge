# EchoForge Container Image
# =========================
# Multi-stage Dockerfile for building optimized EchoForge containers
# Supports development, testing, and production environments

# Build arguments
ARG PYTHON_VERSION=3.11
ARG BUILD_ENV=production
ARG DEBIAN_VERSION=bullseye-slim

# =============================================================================
# Base Image Stage
# =============================================================================
FROM python:${PYTHON_VERSION}-${DEBIAN_VERSION} as base

# Metadata
LABEL maintainer="EchoForge Team"
LABEL description="Privacy-First Multi-Agent LLM Debate Platform"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/yourusername/echoforge"
LABEL org.opencontainers.image.description="EchoForge - Multi-Agent Debate Platform"
LABEL org.opencontainers.image.licenses="MIT"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential system packages
    build-essential \
    curl \
    git \
    wget \
    unzip \
    ca-certificates \
    gnupg \
    lsb-release \
    # Database support
    sqlite3 \
    # Network tools
    netcat-openbsd \
    iputils-ping \
    # Audio processing (for voice features)
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    # Text processing
    libxml2-dev \
    libxslt1-dev \
    # Cryptography dependencies
    libffi-dev \
    libssl-dev \
    # Graph visualization
    graphviz \
    libgraphviz-dev \
    # System monitoring
    htop \
    procps \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Create application user for security
RUN groupadd -r echoforge && useradd -r -g echoforge -s /bin/bash -m echoforge

# Create application directories
RUN mkdir -p /app/{data,logs,temp,frontend,agents,tests} \
    && chown -R echoforge:echoforge /app

# Set working directory
WORKDIR /app

# =============================================================================
# Dependencies Stage
# =============================================================================
FROM base as dependencies

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements files
COPY requirements.txt requirements-dev.txt* ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install development dependencies if available
RUN if [ -f requirements-dev.txt ]; then \
        pip install --no-cache-dir -r requirements-dev.txt; \
    fi

# Download NLTK data
RUN python -c "\
import nltk; \
nltk.download('punkt', quiet=True); \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('averaged_perceptron_tagger', quiet=True); \
nltk.download('vader_lexicon', quiet=True); \
"

# =============================================================================
# Development Stage
# =============================================================================
FROM dependencies as development

# Install additional development tools
RUN pip install --no-cache-dir \
    debugpy \
    ipython \
    jupyter \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-asyncio \
    pytest-cov

# Copy source code
COPY --chown=echoforge:echoforge . .

# Create development configuration
RUN cp .env.example .env 2>/dev/null || true

# Switch to application user
USER echoforge

# Expose ports
EXPOSE 8000 5678

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Development command with auto-reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# =============================================================================
# Testing Stage
# =============================================================================
FROM dependencies as testing

# Install testing dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-mock \
    faker \
    factory-boy \
    httpx \
    coverage

# Copy source code and tests
COPY --chown=echoforge:echoforge . .

# Switch to application user
USER echoforge

# Set test environment
ENV ECHOFORGE_ENV=testing \
    LOG_LEVEL=DEBUG \
    DB_PATH=/app/data/test.db

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=.", "--cov-report=html", "--cov-report=term"]

# =============================================================================
# Production Build Stage
# =============================================================================
FROM dependencies as build

# Copy source code
COPY --chown=echoforge:echoforge . .

# Remove development files
RUN rm -rf \
    tests/ \
    docs/ \
    .git/ \
    .github/ \
    .pytest_cache/ \
    __pycache__/ \
    *.pyc \
    *.pyo \
    .env.example \
    docker-compose*.yml \
    Dockerfile* \
    README.md \
    .gitignore

# Compile Python files for optimization
RUN python -m compileall -b . && \
    find . -name "*.py" -delete && \
    find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Production Stage
# =============================================================================
FROM python:${PYTHON_VERSION}-${DEBIAN_VERSION} as production

# Metadata for production
LABEL environment="production"

# Essential environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    ECHOFORGE_ENV=production \
    LOG_LEVEL=INFO

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sqlite3 \
    ffmpeg \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r echoforge && useradd -r -g echoforge -s /bin/bash -m echoforge

# Create application directories with proper permissions
RUN mkdir -p /app/{data,logs,temp} \
    && chown -R echoforge:echoforge /app

# Set working directory
WORKDIR /app

# Copy Python dependencies from build stage
COPY --from=build /usr/local/lib/python*/site-packages/ /usr/local/lib/python*/site-packages/
COPY --from=build /usr/local/bin/ /usr/local/bin/

# Copy application code from build stage
COPY --from=build --chown=echoforge:echoforge /app/ /app/

# Create production configuration
RUN echo "ECHOFORGE_ENV=production" > .env && \
    echo "DEBUG=false" >> .env && \
    echo "LOG_LEVEL=INFO" >> .env && \
    chown echoforge:echoforge .env

# Switch to application user
USER echoforge

# Create necessary runtime directories
RUN mkdir -p data/backups logs temp frontend/static/uploads

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["gunicorn", "main:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info"]

# =============================================================================
# Multi-Architecture Support
# =============================================================================
FROM production as production-arm64
# ARM64-specific optimizations if needed

FROM production as production-amd64
# AMD64-specific optimizations if needed

# =============================================================================
# Backup Utility Image
# =============================================================================
FROM alpine:latest as backup

# Install backup utilities
RUN apk add --no-cache \
    sqlite \
    tar \
    gzip \
    curl \
    bash \
    dcron \
    && rm -rf /var/cache/apk/*

# Create backup user
RUN addgroup -g 1000 backup && \
    adduser -u 1000 -G backup -s /bin/bash -D backup

# Create backup directories
RUN mkdir -p /app/{data,backups,scripts} && \
    chown -R backup:backup /app

# Copy backup scripts
COPY scripts/backup.sh /app/scripts/
COPY scripts/cleanup.sh /app/scripts/

# Make scripts executable
RUN chmod +x /app/scripts/*.sh && \
    chown backup:backup /app/scripts/*.sh

# Switch to backup user
USER backup

WORKDIR /app

# Default backup command
CMD ["/app/scripts/backup.sh"]

# =============================================================================
# Build Instructions and Usage
# =============================================================================

# Build commands:
# ---------------

# Development build:
# docker build --target development -t echoforge:dev .

# Testing build:
# docker build --target testing -t echoforge:test .

# Production build:
# docker build --target production -t echoforge:prod .

# Multi-arch production build:
# docker buildx build --platform linux/amd64,linux/arm64 --target production -t echoforge:latest .

# Backup utility build:
# docker build --target backup -t echoforge:backup .

# Build with custom Python version:
# docker build --build-arg PYTHON_VERSION=3.12 -t echoforge:python3.12 .

# Build for specific environment:
# docker build --build-arg BUILD_ENV=staging -t echoforge:staging .

# Run commands:
# -------------

# Development:
# docker run -it -p 8000:8000 -v $(pwd):/app echoforge:dev

# Production:
# docker run -d -p 8000:8000 --name echoforge echoforge:prod

# Testing:
# docker run --rm echoforge:test

# With environment variables:
# docker run -d -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 echoforge:prod

# With volume mounts:
# docker run -d -p 8000:8000 -v echoforge_data:/app/data -v echoforge_logs:/app/logs echoforge:prod

# Debug mode:
# docker run -it -p 8000:8000 -p 5678:5678 echoforge:dev python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py

# Optimization notes:
# ------------------
# - Multi-stage build reduces final image size
# - Security: runs as non-root user
# - Layer caching: dependencies installed before code copy
# - Production optimizations: compiled bytecode, minimal runtime
# - Health checks for container orchestration
# - Proper signal handling for graceful shutdown
# - Resource constraints can be set at runtime

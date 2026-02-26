# ============================================================
# Stage 1 — Builder Stage
# Purpose:
#   - Install and compile Python dependencies.
#   - Keep build tools out of final runtime image.
#   - Reduce final image size and attack surface.
# ============================================================

# Use lightweight Python base image
# "slim" reduces unnecessary OS packages while maintaining compatibility
FROM python:3.12-slim AS builder

# Prevent Python from writing .pyc files
# Ensures cleaner containers and avoids unnecessary disk usage
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Working directory for dependency installation
WORKDIR /install

# Install system packages required ONLY for building Python wheels.
# Many ML/scientific libraries require compilation (numpy, scipy, etc.)
# These tools are intentionally excluded from runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first to leverage Docker layer caching.
# Dependencies are only rebuilt when requirements.txt changes.
COPY requirements.txt .

# Upgrade pip and install dependencies into isolated directory.
# --prefix installs packages into /install instead of system path,
# allowing selective copying into runtime image.
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt


# ============================================================
# Stage 2 — Runtime Stage
# Purpose:
#   - Provide minimal environment for running application.
#   - Exclude compilers and build tools.
#   - Improve security and reduce image size.
# ============================================================

FROM python:3.12-slim

# Runtime environment configuration:
# - Disable .pyc generation
# - Ensure logs are flushed immediately (important for containers)
# - Disable pip cache to reduce image size
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Application root directory inside container
WORKDIR /app

# Copy installed dependencies from builder stage.
# This avoids reinstalling packages and keeps runtime image minimal.
COPY --from=builder /install /usr/local

# Copy application source code.
# Separation reflects system architecture:
#   app/               -> FastAPI service layer
#   src/               -> ML pipelines and business logic
#   data_schema/       -> Data validation contracts
#   production_model/  -> Serialized trained model artifacts
COPY app ./app
COPY src ./src
COPY data_schema ./data_schema
COPY production_model ./production_model

# Create a non-root user for security best practices.
# Running containers as root increases risk if compromised.
RUN useradd -m appuser

# Assign ownership of application files to runtime user.
# Ensures application has necessary permissions without root access.
RUN chown -R appuser:appuser /app

# Switch execution to non-root user.
USER appuser

# Expose application port.
# FastAPI + Uvicorn conventionally runs on port 8000.
EXPOSE 8000

# Container startup command.
# Launch ASGI server serving FastAPI application.
# app.main:app refers to:
#   - module: app/main.py
#   - object: FastAPI instance named "app"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

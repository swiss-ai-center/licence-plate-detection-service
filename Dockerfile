# Base image
FROM python:3.11-slim

# Install all required packages to run the model
RUN apt update && apt install --yes git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*


# Work directory
WORKDIR /app

# Environment variables
ENV UV_PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV ENVIRONMENT=${ENVIRONMENT}
ENV LOG_LEVEL=${LOG_LEVEL}
ENV ENGINE_URL=${ENGINE_URL}
ENV MAX_TASKS=${MAX_TASKS}
ENV ENGINE_ANNOUNCE_RETRIES=${ENGINE_ANNOUNCE_RETRIES}
ENV ENGINE_ANNOUNCE_RETRY_DELAY=${ENGINE_ANNOUNCE_RETRY_DELAY}

# Copy requirements file
COPY pyproject.toml uv.lock ./
# Install dependencies
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-dev

# Copy sources
COPY src src


# Exposed ports
EXPOSE 80

# Switch to src directory
WORKDIR "/app/src"
ENTRYPOINT ["uv", "run"]
# Command to run on start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

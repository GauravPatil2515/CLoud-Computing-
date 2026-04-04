# Use an official Python runtime as a parent image.
# Using 'slim' builds smaller distributions with minimal background processes
# which is perfect for serverless Cloud Run.
FROM python:3.11-slim

# Allow container logs to appear directly in Google Cloud Logging.
# This disables output buffering, making logs reliable.
ENV PYTHONUNBUFFERED=1

# Application working directory inside the container
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Copy application files (ignoring files specified in .dockerignore)
COPY . ./

# Install required python packages
# We use --no-cache-dir to prevent disk bloat
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Cloud Run defaults the PORT to 8080 or provides it via env variable.
# Run Uvicorn directly to boot the Web API.
#   --workers: For ML loads with 1 or 2 vCPUs, use 1 worker to save memory.
#   --timeout-keep-alive: Longer timeouts allow instances to respond correctly.
CMD exec uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --workers 1 \
    --timeout-keep-alive 75

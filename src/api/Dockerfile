FROM python:3.11-slim

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Force immediate log output
ENV PYTHONUNBUFFERED=1
ENV IS_DOCKER=true

# Working directory inside the container
WORKDIR /app
ENV PYTHONPATH=/app

# Install minimal system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY src/api/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy API source code
COPY src/api/*.py ./

COPY src/__init__.py /app/src/__init__.py
COPY src/common_utils.py /app/src/common_utils.py
COPY src/custom_logger.py /app/src/custom_logger.py
COPY src/config.yaml /app/src/config.yaml
COPY src/data_processing/schema_manager.py /app/src/data_processing/schema_manager.py
COPY src/data_processing/schema.yaml /app/src/data_processing/schema.yaml
COPY src/config.py /app/src/config.py

# Copy model artifacts (read-only on API side)
COPY models /models

# Expose internal port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

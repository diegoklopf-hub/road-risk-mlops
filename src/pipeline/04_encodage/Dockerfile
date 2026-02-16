FROM python:3.11-slim

WORKDIR /app

# install deps
COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

# copy full project src
COPY src ./src

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

ENTRYPOINT ["python", "src/pipeline/04_encodage/main.py"]

FROM python:3.10-slim

# Empêche Python de créer des fichiers .pyc
ENV PYTHONDONTWRITEBYTECODE=1
# Force l'affichage immédiat des logs
ENV PYTHONUNBUFFERED=1
ENV IS_DOCKER=true

# Dossier de travail dans le conteneur
WORKDIR /app
ENV PYTHONPATH=/app

# Installation des dépendances système minimales
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Copie des dépendances Python
COPY src/api/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copie du code de l'API
COPY src/api/*.py ./

COPY src/__init__.py /app/src/__init__.py
COPY src/common_utils.py /app/src/common_utils.py
COPY src/custom_logger.py /app/src/custom_logger.py
COPY src/config.yaml /app/src/config.yaml

# Copie du modèle (lecture seule côté API)
COPY model /model

# Exposition du port interne
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

PROJECT_NAME=sep25-bmle-mlops-accidents
SHELL := /bin/bash

all: init start-project

init:
	@echo "--- Initialisation du projet ---"
	
	@echo "Création des répertoires de base..."
	@mkdir -p data models logs metrics
	
	@echo "Configuration des variables d'environnement..."
	@export PYTHONPATH="$$(pwd):$${PYTHONPATH}" && \
	if [ ! -f .env ]; then \
		echo "  => Fichier .env absent. Création du template..."; \
		read -p "OPENWEATHER_API_KEY: " api_key; \
		echo "OPENWEATHER_API_KEY=\"$$api_key\"" > .env; \
		echo "HOST_PROJECT_ROOT=$$(pwd)" >> .env; \
		echo "AIRFLOW__CORE__LOAD_EXAMPLES=False" >> .env; \
		echo "AIRFLOW__SCHEDULER__MIN_FILE_PROCESS_INTERVAL=30" >> .env; \
		echo "AIRFLOW__SCHEDULER__PARSING_PROCESSES=2" >> .env; \
		echo "AIRFLOW_GID=0" >> .env; \
	else \
		echo "  => Fichier .env déjà présent."; \
	fi
	
	@echo "Vérification du système (Linux/Docker)..."
	@if [ "$$(uname -s)" = "Linux" ] && [ ! -d /mnt/c ]; then \
		echo "  => Configuration des permissions Airflow (UID/GID)..."; \
		echo "AIRFLOW_UID=$$(id -u)" >> .env; \
		sudo chown -R 50000:50000 logs; \
		sudo chmod -R 775 logs; \
	fi
	
	@if grep -qi microsoft /proc/version; then \
		echo "  => WSL détecté : Ajustement des droits sur le socket Docker..."; \
		sudo chmod 666 /var/run/docker.sock; \
	fi

	@echo "Vérification des certificats SSL pour Nginx..."
	@mkdir -p ./deployments/nginx/certs
	@if [ ! -f "./deployments/nginx/certs/nginx.crt" ]; then \
		echo "  => Certificats manquants. Génération de certificats auto-signés..."; \
		if command -v openssl >/dev/null 2>&1; then \
			openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
				-keyout ./deployments/nginx/certs/nginx.key \
				-out ./deployments/nginx/certs/nginx.crt \
				-subj "/C=FR/ST=France/L=Paris/O=MLOps/OU=IT/CN=localhost"; \
			chmod 644 ./deployments/nginx/certs/nginx.crt; \
			chmod 600 ./deployments/nginx/certs/nginx.key; \
			echo "  => Certificats générés avec succès."; \
		else \
			echo " => Erreur : 'openssl' n'est pas installé. Impossible de générer les certificats."; \
			exit 1; \
		fi \
	else \
		echo "  => Certificats déjà présents."; \
	fi
	
	@echo "Configuration de la base de données utilisateurs..."
	@{ [ ! -f "./data/secrets/users_db.json" ] && \
		{ echo "  => Aucun utilisateur trouvé. Création de l'admin..."; \
		  read -p "Username: " username; \
		  read -s -p "Password: " password; \
		  echo; \
		  python -m src.generate_userdb "$$username" "$$password"; }; } || \
		echo "  => users_db.json existe déjà. Étape ignorée."
	
	@echo "Construction des images docker ..."
	./build_images.sh

	@echo "--- Initialisation terminée avec succès ! ---"
	

start-project:
	docker compose -p $(PROJECT_NAME) up -d --build

stop-project:
	docker compose -p $(PROJECT_NAME) down

reset-project:
	-docker compose down --remove-orphans
	-docker rm -f nginx_revproxy
	docker compose up --build -d

logs:
	docker compose -p $(PROJECT_NAME) logs -f

status:
	docker compose -p $(PROJECT_NAME) ps

unit-test:
	pytest -v -s tests/api_test_unitaires.py 

unit-test-debug:
	pytest -v tests/api_test_unitaires.py 

int-test:
	pytest -v -s tests/api_test_integration.py 

int-test-debug:
	pytest -v tests/api_test_integration.py 

#make pipeline            # démarre à l'étape 1
#make pipeline start=7    # démarre à l'étape 7
pipeline:
	python run_pipeline_mlflow.py  --start $(or $(start),1)

user-init:
	@read -p "Username: " username; \
	read -s -p "Password: " password; \
	echo; \
	python src/generate_userdb.py "$$username" "$$password"



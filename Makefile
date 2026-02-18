PROJECT_NAME=sep25-bmle-mlops-accidents

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
	python src/generate_userdb.py

#INIT WSL : sudo chmod 666 /var/run/docker.sock


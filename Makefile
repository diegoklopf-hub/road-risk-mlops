PROJECT_NAME=accidents-mlops

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

pipeline:
	python src/pipeline/00_pipeline.py

user-init:
	python src/generate_userdb.py


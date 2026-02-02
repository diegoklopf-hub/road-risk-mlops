PROJECT_NAME=accidents-mlops

start-project:
	docker compose -p $(PROJECT_NAME) up -d --build

stop-project:
	docker compose -p $(PROJECT_NAME) down
PROJECT_NAME=accidents-mlops

logs:
	docker compose -p $(PROJECT_NAME) logs -f

status:
	docker compose -p $(PROJECT_NAME) ps


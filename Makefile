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


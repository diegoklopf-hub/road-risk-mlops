import json
import sys
from passlib.context import CryptContext
from pathlib import Path
from src.custom_logger import logger

def initialize_user_db():
    if len(sys.argv) < 3:
        logger.error("Erreur : Nom d'utilisateur ou mot de passe manquant.")
        sys.exit(1)

    username = sys.argv[1]
    password = sys.argv[2]

    pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

    # Plain-text passwords for initialization (dev/test)
    users = {
        username: pwd_context.hash(password)
    }

    # Save to a JSON file
    secrets_path = Path("./data/secrets")
    secrets_path.mkdir(parents=True, exist_ok=True)
    
    file_path = secrets_path / "users_db.json"
    with open(file_path, "w") as f:
        json.dump(users, f, indent=4)

    logger.info(f"Initialisation de {file_path} terminée pour l'utilisateur : {username}")

if __name__ == "__main__":
    initialize_user_db()
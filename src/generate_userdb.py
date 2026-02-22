import json
import sys
from passlib.context import CryptContext
from pathlib import Path
from src.custom_logger import logger

def initialize_user_db():
    if len(sys.argv) < 3:
        logger.error("Error: Missing username or password.")
        sys.exit(1)

    username = sys.argv[1]
    password = sys.argv[2]

    pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

    # Load existing users if the file exists
    secrets_path = Path("./data/secrets")
    secrets_path.mkdir(parents=True, exist_ok=True)
    file_path = secrets_path / "users_db.json"

    users = {}
    if file_path.exists():
        with open(file_path, "r") as f:
            users = json.load(f)

    # Add or update the user
    users[username] = pwd_context.hash(password)

    # Save everything
    with open(file_path, "w") as f:
        json.dump(users, f, indent=4)

    logger.info(f"User {username} added/updated in {file_path}")

if __name__ == "__main__":
    initialize_user_db()

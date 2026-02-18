import json
from passlib.context import CryptContext
from pathlib import Path
from src.custom_logger import logger

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# Plain-text passwords for initialization (dev/test)
users = {
    "admin": pwd_context.hash("XXXXX")
}

# Save to a JSON file
Path("./data/secrets").mkdir(exist_ok=True)
with open("./data/secrets/users_db.json", "w") as f:
    json.dump(users, f, indent=4)

logger.info("users_db.json initialization completed!")

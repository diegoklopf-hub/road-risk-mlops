import json
from passlib.context import CryptContext
from pathlib import Path

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# mots de passe en clair pour initialisation (dev/test)
users = {
    "admin": pwd_context.hash("XXXXX")
}

# Sauvegarde dans un fichier JSON
Path("./data/secrets").mkdir(exist_ok=True)
with open("./data/secrets/users_db.json", "w") as f:
    json.dump(users, f, indent=4)

print("Initialisation users_db.json fait!")

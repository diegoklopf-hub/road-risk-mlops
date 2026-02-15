from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import json
from pathlib import Path
from passlib.context import CryptContext
from src.custom_logger import logger

security = HTTPBasic()
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# Chargement des users
def load_users_db():
    USERS_FILE = Path("/run/secrets/users_db")
    if USERS_FILE.exists():
        users_db = json.loads(USERS_FILE.read_text()) 
        logger.info(f"Authentication: users_db file loaded!")
    else:
        users_db = {}
        logger.info(f"Authentication: fail to load creds users_db file!")
    return users_db

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    users_db = load_users_db()
    hashed = users_db.get(credentials.username)
    if not hashed or not pwd_context.verify(credentials.password, hashed):
        logger.info(f"Authentifcation failed for user: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    logger.info(f"Authentifcation successed for user: {credentials.username}")

    return credentials.username


"""Auth configuration from environment."""
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")  # Fernet key
DB_PATH = os.getenv("DB_PATH", "rag_users.db")

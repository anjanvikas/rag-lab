"""SQLite database layer with Fernet encryption for per-user API key storage."""
import sqlite3
import os
from cryptography.fernet import Fernet
from auth.config import DB_PATH, ENCRYPTION_KEY


def _get_fernet():
    key = ENCRYPTION_KEY
    if not key:
        # Generate a key and warn — for dev only
        key = Fernet.generate_key().decode()
        print(f"WARNING: No ENCRYPTION_KEY set. Generated ephemeral key. Set ENCRYPTION_KEY={key}")
    return Fernet(key.encode() if isinstance(key, str) else key)


def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            picture TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            user_id TEXT PRIMARY KEY,
            encrypted_key BLOB NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


def upsert_user(user_id: str, email: str, name: str, picture: str) -> dict:
    """Insert or update a user record."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO users (id, email, name, picture)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            email=excluded.email,
            name=excluded.name,
            picture=excluded.picture
    """, (user_id, email, name, picture))
    conn.commit()
    conn.close()
    return {"id": user_id, "email": email, "name": name, "picture": picture}


def get_user(user_id: str) -> dict | None:
    """Fetch a user record by ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def store_api_key(user_id: str, api_key: str):
    """Encrypt and store API key for a user."""
    f = _get_fernet()
    encrypted = f.encrypt(api_key.encode())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO api_keys (user_id, encrypted_key)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            encrypted_key=excluded.encrypted_key,
            updated_at=CURRENT_TIMESTAMP
    """, (user_id, encrypted))
    conn.commit()
    conn.close()


def get_api_key(user_id: str) -> str | None:
    """Retrieve and decrypt API key for a user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT encrypted_key FROM api_keys WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    f = _get_fernet()
    return f.decrypt(row[0]).decode()


def delete_api_key(user_id: str):
    """Remove stored API key for a user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM api_keys WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def has_api_key(user_id: str) -> bool:
    """Check if a user has a stored API key."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM api_keys WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row is not None

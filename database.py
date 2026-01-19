import sqlite3
import os
from contextlib import contextmanager

# Use persistent disk path for Render deployment
if os.getenv('RENDER'):
    # On Render, use the mounted disk path
    os.makedirs('/opt/render/project/src/data', exist_ok=True)
    DATABASE_NAME = '/opt/render/project/src/data/users.db'
else:
    DATABASE_NAME = 'users.db'

def init_db():
    """Initialize the SQLite database and create users table"""
    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                username TEXT NOT NULL,
                hashed_password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    print("Database initialized successfully! ✅")

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row  # Access columns by name
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

# Database operations
def create_user(email: str, username: str, hashed_password: str):
    """Insert a new user into the database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (email, username, hashed_password) VALUES (?, ?, ?)",
            (email, username, hashed_password)
        )
        return cursor.lastrowid

def get_user_by_email(email: str):
    """Retrieve user by email"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

def get_all_users():
    """Get all users (for testing/admin purposes)"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, username, created_at FROM users")
        return [dict(row) for row in cursor.fetchall()]

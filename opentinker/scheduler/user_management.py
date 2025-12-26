#!/usr/bin/env python3
"""
User Management for Job Scheduler

Provides user authentication, registration, and API key management
with SQLite persistence.
"""

import hashlib
import logging
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class User:
    """User information"""

    user_id: str
    username: str
    api_key: str
    created_at: str
    is_admin: bool = False


class UserManager:
    """
    Manages user authentication and registration with SQLite persistence.
    """

    def __init__(self, db_path: str = "scheduler_users.db"):
        """
        Initialize user manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
        logger.info(f"UserManager initialized with database: {db_path}")

    def _init_database(self):
        """Initialize SQLite database with users table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                api_key TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()

        logger.info("Database initialized")

    def generate_api_key(self) -> str:
        """
        Generate a secure API key.

        Returns:
            API key in format: otk_<random_hex>
        """
        # Generate 32 bytes of random data, convert to hex
        random_part = secrets.token_hex(16)
        return f"otk_{random_part}"

    def generate_user_id(self, username: str) -> str:
        """
        Generate user ID from username.

        Args:
            username: Username

        Returns:
            User ID in format: user_<hash>
        """
        # Use first 8 chars of SHA256 hash for collision resistance
        hash_part = hashlib.sha256(username.encode()).hexdigest()[:8]
        return f"user_{hash_part}"

    def register_user(self, username: str, is_admin: bool = False) -> User:
        """
        Register a new user.

        Args:
            username: Desired username
            is_admin: Whether user is admin

        Returns:
            Created User object

        Raises:
            ValueError: If username already exists
        """
        # Check if username already exists
        if self.get_user_by_username(username):
            raise ValueError(f"Username '{username}' already exists")

        # Generate user_id and api_key
        user_id = self.generate_user_id(username)
        api_key = self.generate_api_key()
        created_at = datetime.now().isoformat()

        # Insert into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO users (user_id, username, api_key, created_at, is_admin)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, username, api_key, created_at, 1 if is_admin else 0),
            )
            conn.commit()
            logger.info(f"User registered: {username} (admin={is_admin})")
        except sqlite3.IntegrityError as e:
            conn.close()
            raise ValueError(f"Failed to register user: {e}")

        conn.close()

        return User(
            user_id=user_id,
            username=username,
            api_key=api_key,
            created_at=created_at,
            is_admin=is_admin,
        )

    def authenticate(self, api_key: str) -> Optional[User]:
        """
        Authenticate user by API key.

        Args:
            api_key: API key to verify

        Returns:
            User object if valid, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT user_id, username, api_key, created_at, is_admin FROM users WHERE api_key = ?",
            (api_key,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return User(
                user_id=row[0],
                username=row[1],
                api_key=row[2],
                created_at=row[3],
                is_admin=bool(row[4]),
            )
        return None

    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.

        Args:
            username: Username to look up

        Returns:
            User object if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT user_id, username, api_key, created_at, is_admin FROM users WHERE username = ?",
            (username,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return User(
                user_id=row[0],
                username=row[1],
                api_key=row[2],
                created_at=row[3],
                is_admin=bool(row[4]),
            )
        return None

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by user_id.

        Args:
            user_id: User ID to look up

        Returns:
            User object if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT user_id, username, api_key, created_at, is_admin FROM users WHERE user_id = ?",
            (user_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return User(
                user_id=row[0],
                username=row[1],
                api_key=row[2],
                created_at=row[3],
                is_admin=bool(row[4]),
            )
        return None

    def list_users(self) -> List[User]:
        """
        List all users.

        Returns:
            List of User objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT user_id, username, api_key, created_at, is_admin FROM users ORDER BY created_at"
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            User(
                user_id=row[0],
                username=row[1],
                api_key=row[2],
                created_at=row[3],
                is_admin=bool(row[4]),
            )
            for row in rows
        ]

    def delete_user(self, username: str) -> bool:
        """
        Delete a user.

        Args:
            username: Username to delete

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        if deleted:
            logger.info(f"User deleted: {username}")

        return deleted

    def create_default_admin(self, admin_username: str = "admin") -> Optional[User]:
        """
        Create default admin user if it doesn't exist.

        Args:
            admin_username: Admin username (default: "admin")

        Returns:
            Admin User object, or None if already exists
        """
        existing = self.get_user_by_username(admin_username)
        if existing:
            logger.info(f"Admin user '{admin_username}' already exists")
            return None

        admin_user = self.register_user(admin_username, is_admin=True)
        logger.info(f"Created default admin user: {admin_username}")
        logger.info(f"Admin API key: {admin_user.api_key}")

        return admin_user

"""Database connection utilities."""

import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


def get_connection_params() -> dict:
    """Get database connection parameters from environment variables."""
    return {
        "dbname": os.getenv("DATABASE_NAME"),
        "user": os.getenv("DATABASE_USER"),
        "password": os.getenv("DATABASE_PASSWORD"),
        "host": os.getenv("DATABASE_HOST"),
        "port": os.getenv("DATABASE_PORT"),
    }


@contextmanager
def get_db_connection() -> Generator:
    """
    Context manager for database connections.

    Yields:
        psycopg2.connection: Database connection object

    Example:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM table")
                results = cur.fetchall()
    """
    conn = None
    try:
        params = get_connection_params()
        conn = psycopg2.connect(**params)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()


@contextmanager
def get_db_cursor(cursor_factory=RealDictCursor) -> Generator:
    """
    Context manager for database cursor.

    Args:
        cursor_factory: Cursor factory class (default: RealDictCursor for dict results)

    Yields:
        psycopg2.cursor: Database cursor object

    Example:
        with get_db_cursor() as cur:
            cur.execute("SELECT * FROM table")
            results = cur.fetchall()  # Returns list of dicts
    """
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
        finally:
            cursor.close()

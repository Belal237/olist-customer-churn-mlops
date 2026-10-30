"""
Fixtures for tests that run against a real PostgreSQL instance.

Each test session works in a dedicated, randomly named schema that is dropped
afterwards, so the suite never touches the Olist tables of a developer database.
When PostgreSQL is unreachable, tests are skipped locally but fail in CI.
"""

import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from src import config

SEED_PATH = Path(__file__).parent / "seed_rfm.sql"


@pytest.fixture(scope="session")
def pg_engine():
    schema = f"test_rfm_{uuid.uuid4().hex[:8]}"
    admin = create_engine(config.DATABASE_URL)
    try:
        with admin.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    except OperationalError as exc:
        if os.getenv("CI"):
            raise
        pytest.skip(f"PostgreSQL not reachable at {config.POSTGRES_HOST}: {exc.orig}")

    engine = create_engine(
        config.DATABASE_URL,
        connect_args={"options": f"-csearch_path={schema}"},
    )
    with engine.begin() as conn:
        conn.exec_driver_sql(SEED_PATH.read_text(encoding="utf-8"))

    yield engine

    engine.dispose()
    with admin.begin() as conn:
        conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
    admin.dispose()

"""
pytest conftest.py
Shared fixtures for tests.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_client():
    """Session-scoped test client."""
    from app.main import app
    with TestClient(app) as client:
        yield client

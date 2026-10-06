import pytest
from web.app import create_app


@pytest.fixture
def app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "DATABASE": str(tmp_path / "test.sqlite3"),
            "SECRET_KEY": "test-only-key",
            "DATABASE_URL": None,
            "MODEL_OUTPUTS": str(tmp_path),
        }
    )

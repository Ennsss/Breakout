import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text

from web.app import create_app
from web.store import take_quota
from web.tests.test_web import register, session_token, token


def test_short_password_rejected(app):
    client = app.test_client()
    response = client.post(
        "/api/register",
        json={"name": "Test", "email": "a@example.test", "password": "short-password"},
        headers={"X-CSRF-Token": token(client)},
    )
    assert response.status_code == 400


def test_cross_origin_request_rejected_even_with_csrf(app):
    client = app.test_client()
    response = client.post(
        "/api/login",
        json={},
        headers={"X-CSRF-Token": token(client), "Origin": "https://attacker.example"},
    )
    assert response.status_code == 403


def test_idle_timeout_is_server_side(app):
    client = app.test_client()
    register(client)
    with app.extensions["account_engine"].begin() as db:
        db.execute(
            text("UPDATE sessions SET last_seen=:old"), {"old": time.time() - 1801}
        )
    assert client.get("/api/players").status_code == 401


def test_session_rotation_and_hashed_storage(app):
    client = app.test_client()
    register(client)
    with client.session_transaction() as session:
        original = session["sid"]
    client.post(
        "/api/login",
        json={"email": "scout@example.test", "password": "a-long-local-test-password"},
        headers={"X-CSRF-Token": session_token(client)},
    )
    with client.session_transaction() as session:
        assert original != session["sid"]
        current = session["sid"]
    with app.extensions["account_engine"].connect() as db:
        stored = db.execute(text("SELECT token FROM sessions")).scalars().all()
    assert stored == [hashlib.sha256(current.encode()).hexdigest()]


def test_password_change_requires_reauth_and_revokes_all_sessions(app):
    a, b = app.test_client(), app.test_client()
    register(a)
    b.post(
        "/api/login",
        json={"email": "scout@example.test", "password": "a-long-local-test-password"},
        headers={"X-CSRF-Token": token(b)},
    )
    headers = {"X-CSRF-Token": session_token(a)}
    assert (
        a.post(
            "/api/password",
            json={"current": "wrong", "password": "my-new-long-password"},
            headers=headers,
        ).status_code
        == 400
    )
    assert (
        a.post(
            "/api/password",
            json={
                "current": "a-long-local-test-password",
                "password": "my-new-long-password",
            },
            headers=headers,
        ).status_code
        == 200
    )
    assert a.get("/api/players").status_code == 401
    assert b.get("/api/players").status_code == 401
    assert (
        a.post(
            "/api/login",
            json={"email": "scout@example.test", "password": "my-new-long-password"},
            headers={"X-CSRF-Token": token(a)},
        ).status_code
        == 200
    )


def test_logout_all_revokes_other_devices(app):
    a, b = app.test_client(), app.test_client()
    register(a)
    b.post(
        "/api/login",
        json={"email": "scout@example.test", "password": "a-long-local-test-password"},
        headers={"X-CSRF-Token": token(b)},
    )
    assert (
        a.post(
            "/api/logout-all", headers={"X-CSRF-Token": session_token(a)}
        ).status_code
        == 200
    )
    assert b.get("/api/players").status_code == 401


def test_concurrent_quota_is_atomic(app):
    engine = app.extensions["account_engine"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _: take_quota(engine, "test-key", 100, 10), range(30))
        )
    assert sum(results) == 10


def test_secure_cookie_and_headers(app):
    app.config.update(
        PRODUCTION=True,
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_NAME="__Host-breakout",
    )
    response = app.test_client().get("/login", base_url="https://localhost")
    cookie = response.headers["Set-Cookie"]
    assert "__Host-breakout=" in cookie
    for flag in ("Secure", "HttpOnly", "SameSite=Lax", "Path=/"):
        assert flag in cookie
    assert "Domain=" not in cookie
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "max-age=" in response.headers["Strict-Transport-Security"]


def test_production_fails_closed_without_persistent_database(monkeypatch):
    monkeypatch.setenv("BREAKOUT_PRODUCTION", "1")
    monkeypatch.setenv("BREAKOUT_SECRET_KEY", "a" * 64)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("BREAKOUT_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="PostgreSQL"):
        create_app()


def test_account_ui_requires_login(app):
    client = app.test_client()
    assert client.get("/account").status_code == 302
    register(client)
    assert client.get("/account").status_code == 200

"""Flask app with database-backed accounts, revocable sessions and private shortlists."""

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from datetime import timedelta
from pathlib import Path

from flask import Flask, g, jsonify, redirect, render_template, request, session
from werkzeug.security import check_password_hash, generate_password_hash
from web.data import load_dataset


def create_app(config=None):
    app = Flask(__name__)
    instance = Path(
        os.environ.get(
            "BREAKOUT_INSTANCE", Path(__file__).resolve().parent / ".instance"
        )
    )
    instance.mkdir(parents=True, exist_ok=True)
    secret = os.environ.get("BREAKOUT_SECRET_KEY")
    production = os.environ.get("BREAKOUT_PRODUCTION") == "1"
    if production and (not secret or len(secret) < 32):
        raise RuntimeError(
            "Production requires BREAKOUT_SECRET_KEY with at least 32 characters."
        )
    if not secret:
        key = instance / "session.key"
        try:
            with key.open("x") as f:
                f.write(secrets.token_hex(32))
        except FileExistsError:
            pass
        secret = key.read_text()
    app.config.update(
        SECRET_KEY=secret,
        DATABASE=str(instance / "accounts.sqlite3"),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=production,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
        MAX_CONTENT_LENGTH=8192,
        MODEL_OUTPUTS=os.environ.get("BREAKOUT_MODEL_OUTPUTS"),
        ALLOW_REGISTRATION=True,
    )
    if config:
        app.config.update(config)

    def db():
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DATABASE"], timeout=10)
            g.db.row_factory = sqlite3.Row
            g.db.execute("PRAGMA foreign_keys = ON")
        return g.db

    @app.teardown_appcontext
    def close_db(_error):
        connection = g.pop("db", None)
        if connection is not None:
            connection.close()

    with app.app_context():
        db().executescript("""
          CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, password TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), expires REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS shortlist (user_id INTEGER NOT NULL REFERENCES users(id), player_id TEXT NOT NULL, PRIMARY KEY(user_id,player_id));
          CREATE TABLE IF NOT EXISTS attempts (key TEXT NOT NULL, time REAL NOT NULL);
          CREATE INDEX IF NOT EXISTS attempt_time ON attempts(key,time);
        """)
        db().commit()

    def csrf():
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(32)
        return session["csrf"]

    app.jinja_env.globals["csrf_token"] = csrf

    @app.before_request
    def guard():
        g.user = None
        if session.get("sid"):
            token = hashlib.sha256(session["sid"].encode()).hexdigest()
            row = (
                db()
                .execute(
                    "SELECT u.id,u.name,u.email FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=? AND s.expires>?",
                    (token, time.time()),
                )
                .fetchone()
            )
            if row:
                g.user = dict(row)
        if (
            request.path.startswith("/api/")
            and request.path not in ["/api/login", "/api/register"]
        ) or request.path == "/app":
            if not g.user:
                return (
                    (jsonify(error="Sign in to continue."), 401)
                    if request.path.startswith("/api/")
                    else redirect("/login")
                )
        if request.method in ("POST", "DELETE", "PUT", "PATCH"):
            expected = session.get("csrf", "")
            if not expected or not hmac.compare_digest(
                expected.encode(), request.headers.get("X-CSRF-Token", "").encode()
            ):
                return jsonify(
                    error="Your session changed. Reload the page and try again."
                ), 403

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        response.headers["Cache-Control"] = (
            "no-store"
            if request.path.startswith("/api/")
            or request.path in ["/app", "/login", "/register"]
            else "no-cache"
        )
        return response

    @app.get("/")
    def home():
        return render_template("home.html")

    @app.get("/login")
    @app.get("/register")
    def login_page():
        if g.user:
            return redirect("/app")
        return render_template("auth.html", register=request.path == "/register")

    @app.get("/app")
    def dashboard():
        return render_template("dashboard.html", user=g.user)

    def throttled(email):
        # Database-backed per-address and per-account limits work across server workers.
        keys = ["ip:" + (request.remote_addr or "unknown"), "email:" + email]
        now = time.time()
        db().execute("DELETE FROM attempts WHERE time < ?", (now - 900,))
        counts = [
            db()
            .execute(
                "SELECT count(*) FROM attempts WHERE key=? AND time>?", (key, now - 900)
            )
            .fetchone()[0]
            for key in keys
        ]
        if counts[0] >= 30 or counts[1] >= 10:
            return True
        db().executemany(
            "INSERT INTO attempts VALUES (?,?)", [(key, now) for key in keys]
        )
        db().commit()
        return False

    def sign_in(user_id):
        session.clear()
        sid = secrets.token_urlsafe(32)
        session.update(sid=sid, csrf=secrets.token_urlsafe(32))
        session.permanent = True
        db().execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))
        db().execute(
            "INSERT INTO sessions VALUES (?,?,?)",
            (hashlib.sha256(sid.encode()).hexdigest(), user_id, time.time() + 43200),
        )
        db().commit()

    @app.post("/api/register")
    @app.post("/api/login")
    def authenticate():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Enter your email and password."), 400
        email, password = payload.get("email", ""), payload.get("password", "")
        if not isinstance(email, str) or not isinstance(password, str):
            return jsonify(error="Enter valid account details."), 400
        email = email.strip().lower()
        if (
            len(email) > 254
            or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email)
            or not 1 <= len(password) <= 128
        ):
            return jsonify(
                error="Enter a valid email and a password of up to 128 characters."
            ), 400
        if throttled(email):
            return jsonify(
                error="Too many attempts. Wait 15 minutes before trying again."
            ), 429
        if request.path.endswith("register"):
            if not app.config["ALLOW_REGISTRATION"]:
                return jsonify(error="Account registration is closed."), 403
            name = payload.get("name", "")
            if (
                not isinstance(name, str)
                or not 2 <= len(name.strip()) <= 80
                or len(password) < 12
            ):
                return jsonify(
                    error="Use a name of 2-80 characters and a password of at least 12 characters."
                ), 400
            try:
                result = db().execute(
                    "INSERT INTO users(name,email,password) VALUES (?,?,?)",
                    (name.strip(), email, generate_password_hash(password)),
                )
                db().commit()
            except sqlite3.IntegrityError:
                return jsonify(
                    error="An account could not be created with these details. Try signing in."
                ), 409
            sign_in(result.lastrowid)
        else:
            user = (
                db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
            )
            if not user or not check_password_hash(user["password"], password):
                return jsonify(error="Email or password is incorrect."), 401
            sign_in(user["id"])
        return jsonify(ok=True)

    @app.post("/api/logout")
    def logout():
        db().execute(
            "DELETE FROM sessions WHERE token=?",
            (hashlib.sha256(session["sid"].encode()).hexdigest(),),
        )
        db().commit()
        session.clear()
        return jsonify(ok=True)

    @app.get("/api/players")
    def players():
        try:
            data = load_dataset(app.config["MODEL_OUTPUTS"])
        except (ValueError, OSError):
            app.logger.exception("Could not load prediction artifacts")
            return jsonify(
                error="Prediction files could not be read. Check the pipeline output schema."
            ), 503
        visible_ids = {player["id"] for player in data["players"]}
        data["saved"] = [
            r[0]
            for r in db().execute(
                "SELECT player_id FROM shortlist WHERE user_id=?", (g.user["id"],)
            )
            if r[0] in visible_ids
        ]
        return jsonify(data)

    @app.post("/api/shortlist/<player_id>")
    @app.delete("/api/shortlist/<player_id>")
    def shortlist(player_id):
        if request.method == "POST":
            try:
                available = {
                    p["id"]
                    for p in load_dataset(app.config["MODEL_OUTPUTS"])["players"]
                }
            except (ValueError, OSError):
                return jsonify(
                    error="Prediction files could not be read. Try again after checking the pipeline outputs."
                ), 503
            if player_id not in available:
                return jsonify(error="Player not found."), 404
            db().execute(
                "INSERT OR IGNORE INTO shortlist VALUES (?,?)",
                (g.user["id"], player_id),
            )
        else:
            db().execute(
                "DELETE FROM shortlist WHERE user_id=? AND player_id=?",
                (g.user["id"], player_id),
            )
        db().commit()
        return jsonify(ok=True)

    return app


app = create_app()
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5052)

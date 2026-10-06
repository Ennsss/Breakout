"""Authenticated scouting workspace with shared storage and revocable sessions."""

import hashlib
import hmac
import os
import re
import secrets
import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, InvalidHashError
from flask import Flask, g, jsonify, redirect, render_template, request, session
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from web.data import load_dataset
from web.store import make_engine, metadata, take_quota, users

PASSWORDS = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=1)
DUMMY_HASH = PASSWORDS.hash(secrets.token_urlsafe(32))


def create_app(config=None):
    app = Flask(__name__)
    production = (
        os.environ.get("BREAKOUT_PRODUCTION") == "1" or os.environ.get("VERCEL") == "1"
    )
    instance = Path(
        os.environ.get("BREAKOUT_INSTANCE", Path(__file__).parent / ".instance")
    )
    secret = os.environ.get("BREAKOUT_SECRET_KEY")
    database_url = os.environ.get("BREAKOUT_DATABASE_URL") or os.environ.get(
        "DATABASE_URL"
    )
    if production:
        if not secret or len(secret) < 32:
            raise RuntimeError(
                "Production requires a random BREAKOUT_SECRET_KEY of at least 32 characters."
            )
        if not database_url or not database_url.startswith(
            ("postgres://", "postgresql://", "postgresql+psycopg://")
        ):
            raise RuntimeError(
                "Production requires a persistent PostgreSQL DATABASE_URL."
            )
    else:
        instance.mkdir(parents=True, exist_ok=True)
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
        DATABASE=str(instance / "accounts-v2.sqlite3"),
        DATABASE_URL=database_url,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=production,
        SESSION_COOKIE_NAME="__Host-breakout" if production else "session",
        SESSION_COOKIE_PATH="/",
        SESSION_REFRESH_EACH_REQUEST=False,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
        SESSION_IDLE_SECONDS=1800,
        SESSION_ABSOLUTE_SECONDS=28800,
        MAX_CONTENT_LENGTH=8192,
        MODEL_OUTPUTS=os.environ.get("BREAKOUT_MODEL_OUTPUTS"),
        ALLOW_REGISTRATION=True,
        PRODUCTION=production,
    )
    if config:
        app.config.update(config)
    if production:
        app.config["TRUSTED_HOSTS"] = list(
            filter(
                None,
                [
                    "breakout-scouting-frederick.vercel.app",
                    os.environ.get("VERCEL_URL"),
                    os.environ.get("VERCEL_PROJECT_PRODUCTION_URL"),
                ],
            )
        )
    engine = make_engine(
        app.config["DATABASE_URL"] or "sqlite:///" + app.config["DATABASE"]
    )
    app.extensions["account_engine"] = engine
    # Production schema is initialized explicitly, not by the restricted web role.
    if not production:
        metadata.create_all(engine)

    def csrf():
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(32)
        return session["csrf"]

    def token_hash():
        return hashlib.sha256(session.get("sid", "").encode()).hexdigest()

    app.jinja_env.globals["csrf_token"] = csrf

    @app.before_request
    def guard():
        g.user = None
        if request.path.startswith("/static/"):
            return None
        if request.method in ("POST", "DELETE", "PUT", "PATCH"):
            origin = request.headers.get("Origin")
            expected_origin = urlsplit(request.host_url)
            if request.headers.get("Sec-Fetch-Site") == "cross-site" or (
                origin
                and origin != f"{expected_origin.scheme}://{expected_origin.netloc}"
            ):
                return jsonify(error="Cross-site requests are not allowed."), 403
            expected = session.get("csrf", "")
            if not expected or not hmac.compare_digest(
                expected.encode(), request.headers.get("X-CSRF-Token", "").encode()
            ):
                return jsonify(
                    error="Your session changed. Reload the page and try again."
                ), 403
        if session.get("sid"):
            now = time.time()
            with engine.begin() as db:
                row = (
                    db.execute(
                        text("""
                    SELECT u.id,u.name,u.email FROM sessions s JOIN users u ON u.id=s.user_id
                    WHERE s.token=:token AND s.expires>:now AND s.last_seen>:idle
                """),
                        {
                            "token": token_hash(),
                            "now": now,
                            "idle": now - app.config["SESSION_IDLE_SECONDS"],
                        },
                    )
                    .mappings()
                    .first()
                )
                if row:
                    g.user = dict(row)
                    db.execute(
                        text("UPDATE sessions SET last_seen=:now WHERE token=:token"),
                        {"now": now, "token": token_hash()},
                    )
                else:
                    db.execute(
                        text("DELETE FROM sessions WHERE token=:token"),
                        {"token": token_hash()},
                    )
                    session.clear()
        protected = (
            request.path.startswith("/api/")
            and request.path not in ("/api/login", "/api/register")
        ) or request.path in ("/app", "/account")
        if protected and not g.user:
            return (
                (jsonify(error="Sign in to continue."), 401)
                if request.path.startswith("/api/")
                else redirect("/login")
            )

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        response.headers["Cache-Control"] = "no-store"
        if app.config["PRODUCTION"]:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.errorhandler(SQLAlchemyError)
    def unavailable(_error):
        app.logger.error("Account database operation failed")
        return jsonify(
            error="Account service is temporarily unavailable. Please try again shortly."
        ), 503

    @app.errorhandler(413)
    def oversized(_error):
        return jsonify(error="Request is too large."), 413

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

    @app.get("/account")
    def account():
        return render_template("account.html", user=g.user)

    def throttled(email):
        ip = request.remote_addr or "unknown"
        if os.environ.get("VERCEL") == "1":
            ip = request.headers.get("X-Forwarded-For", ip).split(",")[0].strip()
        for value, limit in (("ip:" + ip, 30), ("email:" + email, 10)):
            key = hmac.new(
                app.secret_key.encode(), value.encode(), hashlib.sha256
            ).hexdigest()
            if not take_quota(engine, key, int(time.time() // 900), limit):
                return True
        return False

    def sign_in(user_id, verified_hash):
        old_token = token_hash()
        sid = secrets.token_urlsafe(32)
        now = time.time()
        with engine.begin() as db:
            current = (
                db.execute(
                    users.select().where(users.c.id == user_id).with_for_update()
                )
                .mappings()
                .one()
            )
            if not hmac.compare_digest(current["password"], verified_hash):
                return False
            db.execute(
                text(
                    "DELETE FROM sessions WHERE token=:token OR expires<:now OR last_seen<:idle"
                ),
                {
                    "token": old_token,
                    "now": now,
                    "idle": now - app.config["SESSION_IDLE_SECONDS"],
                },
            )
            db.execute(
                text(
                    "INSERT INTO sessions(token,user_id,expires,last_seen) VALUES (:token,:user_id,:expires,:now)"
                ),
                {
                    "token": hashlib.sha256(sid.encode()).hexdigest(),
                    "user_id": user_id,
                    "expires": now + app.config["SESSION_ABSOLUTE_SECONDS"],
                    "now": now,
                },
            )
        session.clear()
        session.update(sid=sid, csrf=secrets.token_urlsafe(32))
        session.permanent = True
        return True

    def verify(stored, password):
        try:
            return PASSWORDS.verify(stored, password)
        except (VerificationError, InvalidHashError):
            return False

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
            return (
                jsonify(
                    error="Too many attempts. Wait 15 minutes before trying again."
                ),
                429,
                {"Retry-After": "900"},
            )
        if request.path.endswith("register"):
            if not app.config["ALLOW_REGISTRATION"]:
                return jsonify(error="Account registration is closed."), 403
            name = payload.get("name", "")
            if (
                not isinstance(name, str)
                or not 2 <= len(name.strip()) <= 80
                or len(password) < 15
            ):
                return jsonify(
                    error="Use a name of 2-80 characters and a password of at least 15 characters."
                ), 400
            hashed = PASSWORDS.hash(password)
            try:
                with engine.begin() as db:
                    user_id = db.execute(
                        users.insert()
                        .values(name=name.strip(), email=email, password=hashed)
                        .returning(users.c.id)
                    ).scalar_one()
            except IntegrityError:
                return jsonify(
                    error="An account could not be created with these details. Try signing in."
                ), 409
            sign_in(user_id, hashed)
        else:
            with engine.connect() as db:
                user = (
                    db.execute(
                        text("SELECT * FROM users WHERE email=:email"), {"email": email}
                    )
                    .mappings()
                    .first()
                )
            valid = verify(user["password"] if user else DUMMY_HASH, password)
            if not user or not valid:
                return jsonify(error="Email or password is incorrect."), 401
            if not sign_in(user["id"], user["password"]):
                return jsonify(error="Email or password is incorrect."), 401
        return jsonify(ok=True)

    @app.post("/api/logout")
    @app.post("/api/logout-all")
    def logout():
        with engine.begin() as db:
            if request.path.endswith("logout-all"):
                db.execute(
                    text("DELETE FROM sessions WHERE user_id=:id"), {"id": g.user["id"]}
                )
            else:
                db.execute(
                    text("DELETE FROM sessions WHERE token=:token"),
                    {"token": token_hash()},
                )
        session.clear()
        return jsonify(ok=True)

    @app.post("/api/password")
    def change_password():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Enter your current and new password."), 400
        current, new = payload.get("current", ""), payload.get("password", "")
        if (
            not isinstance(current, str)
            or not isinstance(new, str)
            or not 1 <= len(current) <= 128
            or not 15 <= len(new) <= 128
        ):
            return jsonify(
                error="Use a new password between 15 and 128 characters."
            ), 400
        if throttled(g.user["email"]):
            return jsonify(
                error="Too many attempts. Wait 15 minutes before trying again."
            ), 429
        with engine.begin() as db:
            user = (
                db.execute(
                    users.select().where(users.c.id == g.user["id"]).with_for_update()
                )
                .mappings()
                .one()
            )
            if not verify(user["password"], current):
                return jsonify(error="Current password is incorrect."), 400
            db.execute(
                users.update()
                .where(users.c.id == g.user["id"])
                .values(password=PASSWORDS.hash(new))
            )
            db.execute(
                text("DELETE FROM sessions WHERE user_id=:id"), {"id": g.user["id"]}
            )
        session.clear()
        return jsonify(ok=True)

    @app.get("/api/players")
    def players():
        try:
            data = load_dataset(app.config["MODEL_OUTPUTS"])
        except (ValueError, OSError):
            return jsonify(
                error="Prediction files could not be read. Check the pipeline output schema."
            ), 503
        visible_ids = {player["id"] for player in data["players"]}
        with engine.connect() as db:
            data["saved"] = [
                r[0]
                for r in db.execute(
                    text("SELECT player_id FROM shortlist WHERE user_id=:id"),
                    {"id": g.user["id"]},
                )
                if r[0] in visible_ids
            ]
        return jsonify(data)

    @app.post("/api/shortlist/<player_id>")
    @app.delete("/api/shortlist/<player_id>")
    def shortlist(player_id):
        if len(player_id) > 128:
            return jsonify(error="Player not found."), 404
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
            query = "INSERT INTO shortlist(user_id,player_id) VALUES (:id,:player) ON CONFLICT (user_id,player_id) DO NOTHING"
        else:
            query = "DELETE FROM shortlist WHERE user_id=:id AND player_id=:player"
        with engine.begin() as db:
            db.execute(text(query), {"id": g.user["id"], "player": player_id})
        return jsonify(ok=True)

    return app


app = create_app()
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5052)

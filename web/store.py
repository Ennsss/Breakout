"""Shared account storage; PostgreSQL in production, SQLite locally."""

import certifi
from sqlalchemy import Column, Float, ForeignKey, Integer, MetaData, String, Table
from sqlalchemy import create_engine, event, text
from sqlalchemy.pool import NullPool

metadata = MetaData()
users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(80), nullable=False),
    Column("email", String(254), nullable=False, unique=True),
    Column("password", String(512), nullable=False),
)
Table(
    "sessions",
    metadata,
    Column("token", String(64), primary_key=True),
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("expires", Float, nullable=False),
    Column("last_seen", Float, nullable=False),
)
Table(
    "shortlist",
    metadata,
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("player_id", String(128), primary_key=True),
)
Table(
    "rate_limits",
    metadata,
    Column("key", String(64), primary_key=True),
    Column("window", Integer, primary_key=True),
    Column("count", Integer, nullable=False),
)


def make_engine(url):
    if url.startswith(("postgres://", "postgresql://")):
        url = "postgresql+psycopg://" + url.split("://", 1)[1]
    postgres = url.startswith("postgresql+")
    engine = create_engine(
        url,
        poolclass=NullPool,
        hide_parameters=True,
        connect_args={
            "sslmode": "verify-full",
            "sslrootcert": certifi.where(),
            "connect_timeout": 10,
        }
        if postgres
        else {"timeout": 10},
    )
    if not postgres:

        @event.listens_for(engine, "connect")
        def foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

    return engine


def take_quota(engine, key, window, limit):
    # Atomic upsert prevents concurrent serverless workers bypassing the limit.
    with engine.begin() as db:
        db.execute(
            text('DELETE FROM rate_limits WHERE "window" < :old'), {"old": window - 2}
        )
        count = db.execute(
            text("""
            INSERT INTO rate_limits (key, "window", count) VALUES (:key, :window, 1)
            ON CONFLICT (key, "window") DO UPDATE SET count = rate_limits.count + 1
            WHERE rate_limits.count < :limit RETURNING count
        """),
            {"key": key, "window": window, "limit": limit},
        ).scalar_one_or_none()
    return count is not None

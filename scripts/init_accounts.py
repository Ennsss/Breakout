"""Run once with the migration DATABASE_URL before deploying the web app."""

import os
from web.store import make_engine, metadata

engine = make_engine(os.environ["DATABASE_URL"])
metadata.create_all(engine)
engine.dispose()
print("Account schema ready.")

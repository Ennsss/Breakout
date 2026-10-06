# Breakout Web

A public homepage and authenticated scouting workspace for the existing Breakout ML project. The original Streamlit dashboard and model pipeline are unchanged.

## Run

Python 3.11+:

```sh
python -m venv .venv
# Activate .venv, then:
pip install -r requirements-web.txt
python -m web.app
```

Open http://127.0.0.1:5052. Create an account using your own email and a unique password of at least 12 characters. There are no built-in credentials. Accounts are local to this installation.

For a persistent WSGI process:

```sh
waitress-serve --listen=127.0.0.1:5052 web.app:app
```

## What Works

- Public homepage, registration, login, logout, invalid-login states.
- Protected rankings API and dashboard; per-account persistent shortlists.
- Player/club search, cohort/league/season/age filters, sorting, pagination, player reports, and CSV export.
- Model insights sourced from evaluation artifacts, with clearly attributed historical README figures when artifacts are unavailable.
- Responsive mobile navigation, native dialogs, keyboard controls, and accessible form labels.

## Data Provenance

GitHub does not include `outputs/models/` or the source datasets. Without artifacts, the UI uses **10 selected historical examples explicitly published in the existing README**. It labels them as selected case studies, never as current rankings or a representative evaluation cohort. Ages and scores are historical. The snapshot does not invent player statistics, positions, seasons, or per-player SHAP explanations.

To connect the original pipeline, add either or both files:

```text
outputs/models/predictions_test.csv
outputs/models/predictions_current.csv
```

Alternatively, set `BREAKOUT_MODEL_OUTPUTS` to an absolute directory containing those files. Required columns: `name`, `team`, `season`, `league`, `prob_calibrated`. Optional columns include `age`, `birth_year`, `position_group`, `label`, `breakout_league`, `prob_lgbm`, `prob_xgb`. Invalid/non-finite probability rows are excluded; malformed schemas return an error rather than silently displaying fallback examples. The API reads artifacts on each request.

Records represent player-seasons, not unique people. The cohort selector separates historical evaluation from scouting predictions, and the latter is selected initially when available. Scouting predictions still reflect their saved season, not today's live data. Shortlists retain their original record IDs; saved records absent from the connected dataset are not counted or shown.

Optionally include `evaluation_results.json` and `feature_importance.csv` in the same directory. Model insights reads `summary.calibrated_metrics` and the top five global feature importances, with source labels. Missing or invalid report artifacts fall back to explicitly attributed README figures. No metrics are recomputed. Individual SHAP charts and richer statistics remain in the original Streamlit dashboard when its artifacts are available.

## Authentication and Hosting

- SQLite stores scrypt password hashes, hashed session tokens, account-scoped shortlists, and login-attempt counters.
- Sessions expire after 12 hours, are revocable on logout, and use HttpOnly / SameSite=Lax cookies.
- State-changing requests require a session-bound CSRF token. Database-backed login limits apply per email and source address.
- `web/.instance/` contains local secrets and account data and is ignored by Git. Back it up appropriately; never publish it.
- Set `BREAKOUT_PRODUCTION=1` and a random `BREAKOUT_SECRET_KEY` of at least 32 characters for hosting. This enforces Secure cookies. Serve behind HTTPS.
- Set `BREAKOUT_INSTANCE` to a persistent, access-restricted disk directory. Ephemeral/serverless storage is not appropriate for this SQLite setup.
- Deploy a single host, or replace SQLite with a shared database for a multi-host deployment. Apply proxy rate limits and configure trusted proxy handling for your actual topology rather than trusting arbitrary forwarding headers.
- No email verification, password reset, team sharing, MFA, or external identity provider is implemented. This is a small research/portfolio app, not an enterprise identity service.
- Authentication protects the new Flask web app only. Do not expose the separate legacy Streamlit server publicly without its own access controls.

## Tests

```sh
python -m pytest web/tests -q
```

Tests cover registration, login failure/success, password hashing, session expiry and revocation, CSRF, rate limiting, private shortlists, stale saved records, artifact validation, optional evaluation reports, and snapshot provenance.

## Assets and AI Assistance

- Icons: Lucide 0.468.0, ISC license (vendored; license header retained).
- Stadium: [Omar Ramadan / Unsplash](https://unsplash.com/photos/soccer-game-on-a-stadium-jvBRJWFGbtg), illustrative stadium photography, not a club endorsement.
- Frontend and web integration developed with OpenAI Codex. Existing model results remain the original project's work.

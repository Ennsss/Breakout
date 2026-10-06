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

Open http://127.0.0.1:5052. Create an account using a unique passphrase of at least 15 characters. There are no built-in credentials. Local and hosted accounts are separate.

Live application: https://breakout-scouting-frederick.vercel.app

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

- Neon PostgreSQL persists hosted accounts and shortlists across Vercel deployments. SQLAlchemy uses parameterized statements and verified TLS. The runtime database role has only CRUD permissions on the four application tables and access to the user-ID sequence, not schema ownership. Owner credentials are not installed in Vercel's runtime environment.
- Passwords use Argon2id (64 MiB, three iterations, one lane), with 15-128 character passphrases. Nonexistent accounts still perform a dummy hash check; login failures do not identify whether an email exists.
- Random 256-bit session tokens are stored only as SHA-256 hashes in PostgreSQL. Signed `__Host-breakout` cookies are Secure, HttpOnly, SameSite=Lax, and scoped to `/` without a Domain attribute. Session tokens never go into localStorage.
- Server-side expiry is 30 minutes idle / eight hours absolute. Login rotates sessions; logout revokes the token. Account security supports current-password-verified password changes and signing out all devices. Password changes revoke all sessions.
- Mutations require a session-bound CSRF token and reject cross-site browser origins. CSP, HSTS, frame restrictions, no-sniff, and no-store headers are enabled. Trusted hosts are restricted in production.
- Atomic PostgreSQL counters enforce 10 authentication attempts per email and 30 per IP per fixed 15-minute window across workers. Counters store keyed hashes, not raw email/IP values. Vercel's overwritten client-IP header is trusted only on Vercel. These are application controls, not complete DDoS protection.
- `web/.instance/` remains local-only and ignored. Version 2 uses `accounts-v2.sqlite3`; the original local database is preserved, not migrated or uploaded. Never publish instance data or `.env*` files.
- No email verification, password reset, team sharing, MFA, or external identity provider is implemented. This is a small research/portfolio app, not an enterprise identity service.
- Authentication protects the new Flask web app only. Do not expose the separate legacy Streamlit server publicly without its own access controls.

## Vercel Deployment

`pyproject.toml` selects the lightweight Flask runtime, not the original ML training dependencies. The build publishes only `web/static` as CDN assets. The four prediction/evaluation files remain in the server bundle, behind the authenticated API. `.vercelignore` excludes account files, secrets, caches, tests, and the training datasets.

1. Initialize an empty PostgreSQL database with `DATABASE_URL` set to its migration connection: `python -m scripts.init_accounts`.
2. Create a runtime login with only the table/sequence permissions above. Configure its URL as `BREAKOUT_DATABASE_URL` and a random 64-byte `BREAKOUT_SECRET_KEY` in Vercel's Production environment. Do not configure the database-owner URL in the app.
3. Include the four original files in `outputs/models/` when deploying from the CLI. They remain ignored by Git, so Git-based builds alone cannot reproduce the full dataset.
4. Run `vercel deploy --prod`. Production fails closed without a strong key and persistent PostgreSQL URL. For non-Vercel hosting, set `BREAKOUT_PRODUCTION=1` and adjust the trusted-host list to your domain.

Keep the Neon resource on its free plan. The marketplace's default project connection was disconnected after provisioning to remove owner-level environment variables; the restricted runtime URL still connects to the same database. No paid upgrade was selected. Free-tier quotas and cold starts still apply.

## Tests

```sh
python -m pytest web/tests -q
```

Tests cover registration, login failure/success, Argon2id hashing, absolute/idle expiry, rotation and revocation, CSRF/origin checks, concurrent rate limits, secure cookies, password changes, private shortlists, stale records, artifact validation, optional evaluation reports, and snapshot provenance. Live smoke tests additionally exercise PostgreSQL permissions and persistent shortlists across logout/login using disposable accounts that are deleted afterward.

## Assets and AI Assistance

- Icons: Lucide 0.468.0, ISC license (vendored; license header retained).
- Stadium: [Omar Ramadan / Unsplash](https://unsplash.com/photos/soccer-game-on-a-stadium-jvBRJWFGbtg), illustrative stadium photography, not a club endorsement.
- Frontend and web integration developed with OpenAI Codex. Existing model results remain the original project's work.

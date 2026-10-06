import csv
import json
import re
import sqlite3

import pytest

from web.data import load_dataset, load_model_report


def token(client):
    html = client.get("/login").text
    return re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)


def register(client, email="scout@example.test"):
    return client.post(
        "/api/register",
        json={
            "name": "Test Scout",
            "email": email,
            "password": "a-long-local-test-password",
        },
        headers={"X-CSRF-Token": token(client)},
    )


def session_token(client):
    with client.session_transaction() as s:
        return s["csrf"]


def test_public_home_and_protected_routes(app):
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/app").status_code == 302
    assert client.get("/api/players").status_code == 401


def test_csrf_required(app):
    client = app.test_client()
    assert client.post("/api/register", json={}).status_code == 403


def test_register_login_wrong_password_and_logout_revocation(app):
    client = app.test_client()
    assert register(client).status_code == 200
    assert client.get("/app").status_code == 200
    old_cookie = client.get_cookie("session").value
    assert (
        client.post(
            "/api/logout", headers={"X-CSRF-Token": session_token(client)}
        ).status_code
        == 200
    )
    client.set_cookie("session", old_cookie)
    assert client.get("/api/players").status_code == 401
    client.delete_cookie("session")
    headers = {"X-CSRF-Token": token(client)}
    assert (
        client.post(
            "/api/login",
            json={"email": "scout@example.test", "password": "wrong"},
            headers=headers,
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/login",
            json={
                "email": "SCOUT@example.test",
                "password": "a-long-local-test-password",
            },
            headers=headers,
        ).status_code
        == 200
    )


def test_shortlists_are_private_and_idempotent(app):
    a, b = app.test_client(), app.test_client()
    register(a, "a@example.test")
    register(b, "b@example.test")
    player = a.get("/api/players").json["players"][0]["id"]
    headers = {"X-CSRF-Token": session_token(a)}
    for _ in range(2):
        assert a.post("/api/shortlist/" + player, headers=headers).status_code == 200
    assert a.get("/api/players").json["saved"] == [player]
    assert b.get("/api/players").json["saved"] == []
    assert a.post("/api/shortlist/nonexistent", headers=headers).status_code == 404
    assert a.delete("/api/shortlist/" + player, headers=headers).status_code == 200
    assert a.get("/api/players").json["saved"] == []


def test_password_hash_and_session_expiry(app):
    client = app.test_client()
    register(client)
    with sqlite3.connect(app.config["DATABASE"]) as db:
        value = db.execute("SELECT password FROM users").fetchone()[0]
        assert value.startswith("$argon2id$") and value != "a-long-local-test-password"
        db.execute("UPDATE sessions SET expires=0")
        db.commit()
    assert client.get("/api/players").status_code == 401


def test_auth_rate_limit(app):
    client = app.test_client()
    headers = {"X-CSRF-Token": token(client)}
    for _ in range(10):
        assert (
            client.post(
                "/api/login",
                json={"email": "bad@example.test", "password": "wrong"},
                headers=headers,
            ).status_code
            == 401
        )
    assert (
        client.post(
            "/api/login",
            json={"email": "bad@example.test", "password": "wrong"},
            headers=headers,
        ).status_code
        == 429
    )


@pytest.mark.parametrize(
    "body",
    [None, [], {"email": 42, "password": []}, {"email": "invalid", "password": "123"}],
)
def test_bad_auth_payloads(app, body):
    client = app.test_client()
    assert (
        client.post(
            "/api/register", json=body, headers={"X-CSRF-Token": token(client)}
        ).status_code
        == 400
    )


def test_snapshot_is_identified(tmp_path):
    data = load_dataset(tmp_path)
    assert data["snapshot"] and len(data["players"]) == 10
    assert all(p["source"] == "README case study" for p in data["players"])


def test_pipeline_data_overrides_snapshot_and_rejects_bad_scores(tmp_path):
    path = tmp_path / "predictions_current.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "team",
                "season",
                "league",
                "birth_year",
                "prob_calibrated",
            ],
        )
        writer.writeheader()
        for score in ["0.42", "nan", "-1", "2"]:
            writer.writerow(
                {
                    "name": "Test Player",
                    "team": "Test Club",
                    "season": "2024-2025",
                    "league": "championship",
                    "birth_year": "2003",
                    "prob_calibrated": score,
                }
            )
    data = load_dataset(tmp_path)
    assert not data["snapshot"] and len(data["players"]) == 1
    assert data["players"][0]["age"] == 21
    assert data["players"][0]["label"] is None


def test_malformed_artifact_does_not_silently_fallback(tmp_path):
    (tmp_path / "predictions_test.csv").write_text("wrong,columns\na,b\n")
    with pytest.raises(ValueError):
        load_dataset(tmp_path)


def test_evaluation_artifacts_are_read_without_recomputing(tmp_path):
    metrics = {
        "roc_auc": 0.72,
        "precision_at_20": 0.65,
        "average_precision": 0.4,
        "brier_score": 0.13,
    }
    (tmp_path / "evaluation_results.json").write_text(
        json.dumps({"summary": {"calibrated_metrics": metrics}})
    )
    (tmp_path / "feature_importance.csv").write_text(
        "feature,importance\nage,0.1\nminutes,0.3\ninvalid,nan\n"
    )
    report = load_model_report(tmp_path)
    assert report["metrics"] == metrics
    assert report["features"] == [
        {"name": "minutes", "importance": 0.3},
        {"name": "age", "importance": 0.1},
    ]


def test_missing_or_invalid_reports_are_explicitly_unavailable(tmp_path):
    assert load_model_report(tmp_path) == {"metrics": None, "features": None}
    (tmp_path / "evaluation_results.json").write_text("{invalid")
    (tmp_path / "feature_importance.csv").write_text("feature,importance\nage,nan\n")
    assert load_model_report(tmp_path) == {"metrics": None, "features": None}


def test_shortlist_counts_only_records_in_connected_dataset(app):
    client = app.test_client()
    register(client)
    with sqlite3.connect(app.config["DATABASE"]) as db:
        db.execute("INSERT INTO shortlist VALUES (1, 'old-dataset-record')")
        db.commit()
    assert client.get("/api/players").json["saved"] == []


def test_broken_artifact_has_actionable_api_error(app, tmp_path):
    client = app.test_client()
    register(client)
    (tmp_path / "predictions_current.csv").write_text("wrong,columns\na,b\n")
    assert client.get("/api/players").status_code == 503
    assert (
        client.post(
            "/api/shortlist/anything", headers={"X-CSRF-Token": session_token(client)}
        ).status_code
        == 503
    )

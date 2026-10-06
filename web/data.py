"""Read pipeline artifacts, or explicitly identified README case studies."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def number(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (ValueError, TypeError):
        return None


def load_model_report(directory):
    """Optional evaluation artifacts never change the prediction cohort."""
    report = {"metrics": None, "features": None}
    try:
        evaluation = json.loads((directory / "evaluation_results.json").read_text())
        metrics = evaluation["summary"]["calibrated_metrics"]
        keys = ["roc_auc", "precision_at_20", "average_precision", "brier_score"]
        parsed = {key: number(metrics.get(key)) for key in keys}
        if all(value is not None and 0 <= value <= 1 for value in parsed.values()):
            report["metrics"] = parsed
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    try:
        with (directory / "feature_importance.csv").open(encoding="utf-8-sig") as f:
            features = [
                {"name": row["feature"], "importance": number(row["importance"])}
                for row in csv.DictReader(f)
            ]
        valid = [
            f for f in features if f["importance"] is not None and f["importance"] >= 0
        ]
        report["features"] = (
            sorted(valid, key=lambda f: f["importance"], reverse=True)[:5] or None
        )
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return report


def load_dataset(directory=None):
    directory = Path(directory or ROOT / "outputs" / "models")
    rows = []
    found = False
    for name, source in [
        ("predictions_test.csv", "Historical evaluation"),
        ("predictions_current.csv", "Scouting predictions"),
    ]:
        path = directory / name
        if not path.exists():
            continue
        found = True
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"name", "team", "season", "league", "prob_calibrated"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{name} is missing required prediction columns")
            for raw in reader:
                probability = number(raw.get("prob_calibrated"))
                if probability is None or not 0 <= probability <= 1:
                    continue
                age = number(raw.get("age"))
                if age is None and number(raw.get("birth_year")) is not None:
                    age = number(str(raw.get("season", "")).split("-")[0])
                    age = age - number(raw["birth_year"]) if age is not None else None
                rows.append(
                    {
                        "name": raw["name"],
                        "team": raw["team"],
                        "league": raw["league"],
                        "season": raw["season"],
                        "position": raw.get("position_group")
                        or raw.get("position")
                        or "Unspecified",
                        "age": age,
                        "probability": probability,
                        "label": number(raw.get("label")),
                        "source": source,
                        "destination": raw.get("breakout_league") or None,
                        "lgbm": number(raw.get("prob_lgbm")),
                        "xgb": number(raw.get("prob_xgb")),
                    }
                )
    if not found:
        rows = json.loads(
            (Path(__file__).parent / "published_cases.json").read_text(encoding="utf-8")
        )
    for row in rows:
        identity = "|".join(
            str(row.get(k, "")) for k in ["source", "name", "team", "season"]
        )
        row["id"] = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return {
        "players": sorted(rows, key=lambda p: p["probability"], reverse=True),
        "source": "Pipeline artifacts" if found else "Published case studies",
        "snapshot": not found,
        "model_report": load_model_report(directory),
        "notice": "Saved pipeline predictions, not live match data."
        if found
        else "10 selected historical predictions from the project README, not the full evaluation cohort. Connect pipeline artifacts for complete rankings.",
    }

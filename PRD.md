# Hidden Gem Finder - Product Requirements Document

## Executive Summary

Build a machine learning system that identifies undervalued football players in lower European leagues who have high potential to succeed at top-5 leagues. The system will provide scouts and analysts with data-driven recommendations, complete with explanations and similar player comparisons.

---

## 1. Problem Statement

### The Challenge
- Top football clubs spend millions on scouting networks
- Most "hidden gems" are discovered too late or missed entirely
- Human scouts can only watch a fraction of available matches
- Lower league data is underutilized despite being publicly available

### The Opportunity
- Historical data shows patterns: players who break out share common statistical profiles
- ML can process thousands of players simultaneously
- Objective metrics complement subjective scouting reports
- Early identification = lower transfer fees, competitive advantage

---

## 2. Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| ROC-AUC | > 0.75 | Strong discriminative ability |
| Precision @ 20 | > 0.30 | 6+ actual breakouts in top 20 predictions |
| Recall @ 100 | > 0.50 | Catch half of all breakouts in top 100 |
| Brier Score | < 0.15 | Well-calibrated probability estimates |
| Explanation Coverage | 100% | Every prediction has SHAP explanation |

### Qualitative Success
- System would have flagged known success stories (Haaland at Salzburg, Luis Diaz at Porto)
- Explanations make sense to domain experts
- Actionable output format for scouts

---

## 3. Scope

### In Scope
| Area | Details |
|------|---------|
| **Player Types** | Outfield players only (FW, MF, DF) |
| **Source Leagues** | Eredivisie, Portuguese Liga, Belgian Pro League, Championship, Serie B, Ligue 2, Austrian Bundesliga, Scottish Premiership |
| **Target Leagues** | Premier League, La Liga, Bundesliga, Serie A, Ligue 1 |
| **Training Period** | 2015-2022 seasons |
| **Validation Period** | 2022-2023 season |
| **Prediction Period** | Current season |
| **Age Range** | 17-26 years old |

### Out of Scope
- Goalkeepers (different metrics entirely)
- Youth/academy players without senior minutes
- MLS, South American, Asian leagues (future extension)
- Real-time match analysis
- Video/computer vision features

---

## 4. Data Requirements

### 4.1 Data Sources

| Source | Data Type | Priority | Notes |
|--------|-----------|----------|-------|
| **FBref** | Advanced stats (per 90) | P0 | Primary source, most comprehensive |
| **Transfermarkt** | Market values, transfers, contracts | P0 | Required for target variable |
| **Understat** | xG, xA, shot maps | P1 | Attacking metrics depth |
| **Sofascore** | Match ratings, form | P2 | Supplementary signal |
| **Capology** | Wage data | P3 | Future enhancement |

### 4.2 Data Schema

#### PLAYERS (Static Info)
```
player_id: str (primary key, generated UUID)
name: str
date_of_birth: date
nationality: str
primary_position: str (ST, LW, CM, CB, etc.)
secondary_positions: list[str]
height_cm: int (optional)
foot: str (optional)
```

#### PLAYER_SEASONS (Per Season Stats)
```
player_id: str (foreign key)
season: str (e.g., "2023-24")
team: str
league: str
minutes_played: int
games_started: int
games_as_sub: int

# Attacking
goals: int
assists: int
xg: float
xa: float
npxg: float (non-penalty xG)
shots_per90: float
shots_on_target_pct: float

# Passing
passes_per90: float
pass_completion_pct: float
progressive_passes_per90: float
key_passes_per90: float
through_balls_per90: float

# Possession
progressive_carries_per90: float
successful_dribbles_per90: float
touches_in_box_per90: float

# Defensive
tackles_per90: float
interceptions_per90: float
blocks_per90: float
clearances_per90: float
aerials_won_pct: float

# Valuation
market_value_eur: int
wage_eur: int (if available)
contract_expires: date (if available)
```

#### TRANSFERS (Historical Record)
```
transfer_id: str (primary key)
player_id: str (foreign key)
transfer_date: date
season: str
from_team: str
from_league: str
to_team: str
to_league: str
transfer_fee_eur: int (0 for free, -1 for loan)
is_loan: bool
```

### 4.3 Target Variable Definition

A player is labeled as **"breakout"** (y=1) if within 3 seasons they:
1. Transferred to a top-5 league, AND
2. Played 900+ minutes there (not a failed move)

```python
def create_label(player_id, observation_season, transfers_df, minutes_df):
    TOP_5 = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
    MIN_MINUTES = 900
    LOOKFORWARD_YEARS = 3

    # Find transfers to top-5 within window
    future_transfers = transfers_df[
        (transfers_df.player_id == player_id) &
        (transfers_df.season > observation_season) &
        (transfers_df.season <= observation_season + LOOKFORWARD_YEARS) &
        (transfers_df.to_league.isin(TOP_5))
    ]

    if future_transfers.empty:
        return 0

    # Check meaningful playing time
    top5_minutes = minutes_df[
        (minutes_df.player_id == player_id) &
        (minutes_df.league.isin(TOP_5)) &
        (minutes_df.season > observation_season)
    ].minutes_played.sum()

    return 1 if top5_minutes >= MIN_MINUTES else 0
```

---

## 5. Feature Engineering

### 5.1 Raw Features (Direct from Data)
- Age at observation time
- Minutes played (current season)
- All per-90 statistics from schema
- Current market value
- Contract years remaining

### 5.2 Derived Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| **xG Overperformance** | goals - xG | Clinical finishing ability |
| **xA Overperformance** | assists - xA | Creative quality beyond chance creation |
| **League Difficulty Coefficient** | UEFA/Elo based multiplier | Normalize across leagues |
| **Age-Position Percentile** | Rank within age band + position | Context for raw numbers |
| **YoY Growth** | (stat_current - stat_prev) / stat_prev | Trajectory matters |
| **Minutes Growth** | minutes_current / minutes_prev | Rising importance |
| **Consistency Score** | 1 - std(match_ratings) / mean(match_ratings) | Scouts value consistency |
| **Involvement Score** | (touches + passes) / minutes * 90 | Central to team play |
| **Goal Contribution per90** | (goals + assists) / 90 | Combined attacking output |

### 5.3 League Difficulty Coefficients

Based on historical transfer success rates:

```python
LEAGUE_COEFFICIENTS = {
    "Eredivisie": 0.85,
    "Portuguese Liga": 0.82,
    "Championship": 0.78,
    "Belgian Pro League": 0.75,
    "Serie B": 0.72,
    "Ligue 2": 0.70,
    "Austrian Bundesliga": 0.68,
    "Scottish Premiership": 0.65,
}
```

### 5.4 Position-Specific Feature Sets

| Position Group | Primary Features |
|----------------|------------------|
| **Forwards (ST, LW, RW, CF)** | xG, npxG, shots, dribbles, touches in box |
| **Midfielders (CM, CAM, CDM)** | Progressive passes, key passes, xA, pass completion |
| **Defenders (CB, LB, RB)** | Tackles, interceptions, aerials, clearances |

---

## 6. Technical Architecture

### 6.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   FBref     │  │Transfermarkt│  │  Understat  │              │
│  │  Scraper    │  │  Scraper    │  │  Scraper    │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│                 ┌─────────────────┐                             │
│                 │  Raw Data Lake  │                             │
│                 │   (Parquet)     │                             │
│                 └────────┬────────┘                             │
└──────────────────────────┼──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │  Data Cleaning  │───▶│Feature Engineer │                     │
│  │   & Merging     │    │                 │                     │
│  └─────────────────┘    └────────┬────────┘                     │
│                                  ▼                              │
│                    ┌─────────────────────────┐                  │
│                    │  Training Dataset       │                  │
│                    │  (features + labels)    │                  │
│                    └────────────┬────────────┘                  │
└─────────────────────────────────┼───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   LightGBM   │  │   XGBoost    │  │   Logistic   │           │
│  │  Classifier  │  │  Classifier  │  │  (baseline)  │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           ▼                                     │
│                 ┌─────────────────┐                             │
│                 │ Ensemble Model  │                             │
│                 │ (weighted avg)  │                             │
│                 └────────┬────────┘                             │
└──────────────────────────┼──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  SHAP Explainer │  │ Player Similarity│  │ Streamlit App  │  │
│  │                 │  │ Engine (cosine)  │  │ + PDF Reports  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Directory Structure

```
hidden-gem-finder/
├── CLAUDE.md                 # AI assistant instructions
├── PRD.md                    # This document
├── README.md                 # Project overview
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project config
│
├── config/
│   ├── leagues.yaml          # League metadata, coefficients
│   ├── features.yaml         # Feature definitions
│   └── model_params.yaml     # Hyperparameters
│
├── data/
│   ├── raw/                  # Untouched scraped data
│   │   ├── fbref/
│   │   ├── transfermarkt/
│   │   └── understat/
│   ├── processed/            # Cleaned, merged data
│   │   ├── players.parquet
│   │   ├── player_seasons.parquet
│   │   └── transfers.parquet
│   └── outputs/              # Predictions, reports
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_predictions.ipynb
│
├── src/
│   ├── __init__.py
│   ├── scrapers/
│   │   ├── __init__.py
│   │   ├── base_scraper.py
│   │   ├── fbref_scraper.py
│   │   ├── transfermarkt_scraper.py
│   │   └── understat_scraper.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cleaning.py
│   │   ├── merging.py
│   │   └── labeling.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py
│   │   ├── selection.py
│   │   └── league_adjustment.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── predictor.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_analysis.py
│   │   └── similarity.py
│   └── reporting/
│       ├── __init__.py
│       ├── dashboard.py
│       └── pdf_report.py
│
├── tests/
│   ├── __init__.py
│   ├── test_scrapers.py
│   ├── test_features.py
│   ├── test_models.py
│   └── fixtures/             # Mock data for tests
│
├── notes/                    # Session notes, decisions
│   └── .gitkeep
│
└── outputs/
    ├── models/               # Saved model artifacts
    ├── reports/              # Generated PDFs
    └── predictions/          # Prediction CSVs
```

### 6.3 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Language | Python 3.11+ | ML ecosystem |
| Data Processing | Pandas, Polars | Efficient dataframes |
| ML Framework | scikit-learn, LightGBM, XGBoost | Industry standard |
| Hyperparameter Tuning | Optuna | Efficient search |
| Explainability | SHAP | Global + local explanations |
| Dashboard | Streamlit | Rapid prototyping |
| Data Storage | Parquet | Efficient, typed storage |
| Testing | pytest | Standard testing |
| Linting | ruff | Fast, comprehensive |

---

## 7. Development Phases

### Phase 1: Setup & Data Collection (Weeks 1-2)

**Goals:**
- Project infrastructure
- Working scrapers for all sources
- Raw data for 2015-2024

**Deliverables:**
- [ ] Git repo with structure
- [ ] Virtual environment + requirements.txt
- [ ] FBref scraper with tests
- [ ] Transfermarkt scraper with tests
- [ ] Understat scraper with tests
- [ ] Raw data collected and stored
- [ ] Data exploration notebook

**Success Criteria:**
- All scrapers pass tests with mock responses
- Can scrape a full season in <30 minutes per source
- Raw data validates against expected schemas

---

### Phase 2: Data Processing & Feature Engineering (Weeks 3-4)

**Goals:**
- Merge datasets reliably
- Construct target variable
- Build feature pipeline

**Deliverables:**
- [ ] Player matching/merging logic
- [ ] Target variable construction
- [ ] Feature engineering pipeline
- [ ] League adjustment implementation
- [ ] Feature selection analysis
- [ ] Final training dataset

**Success Criteria:**
- Player matching achieves >95% precision
- No temporal leakage in features
- Feature importance analysis completed
- Dataset ready for modeling

---

### Phase 3: Model Development (Weeks 5-6)

**Goals:**
- Train performant models
- Robust evaluation
- Ensemble if beneficial

**Deliverables:**
- [ ] Train/val/test splits (walk-forward)
- [ ] Baseline logistic regression
- [ ] Tuned LightGBM model
- [ ] Tuned XGBoost model
- [ ] Ensemble implementation
- [ ] Comprehensive evaluation notebook

**Success Criteria:**
- ROC-AUC > 0.75 on test set
- Precision @ 20 > 0.30
- Models validated on 2022-2023 holdout
- Case study validation (known breakouts)

---

### Phase 4: Explainability & Similarity (Weeks 7-8)

**Goals:**
- Interpretable predictions
- Player comparison system

**Deliverables:**
- [ ] SHAP integration
- [ ] Global feature importance
- [ ] Local explanations per player
- [ ] Player embedding space
- [ ] Similarity search function
- [ ] Explanation validation with domain knowledge

**Success Criteria:**
- Every prediction has explanation
- Explanations pass sanity check
- "Similar to X" comparisons are sensible

---

### Phase 5: Output & Deployment (Weeks 9-10)

**Goals:**
- Production prediction pipeline
- User-facing interface
- Documentation

**Deliverables:**
- [ ] Prediction pipeline for current season
- [ ] Streamlit dashboard
- [ ] PDF report generation
- [ ] Complete README
- [ ] Test coverage >80%
- [ ] Final predictions for current season

**Success Criteria:**
- Dashboard loads in <5 seconds
- PDF reports generate correctly
- End-to-end pipeline runs without intervention
- Documentation complete

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Player name matching fails | High | High | Fuzzy matching + multiple identifiers (DOB, team) |
| Severe class imbalance | High | Medium | SMOTE, class weights, precision-focused metrics |
| Data quality in lower leagues | Medium | High | Imputation, uncertainty flags, exclude worst leagues |
| Model overfits to specific leagues | Medium | High | League as feature, cross-validate across leagues |
| Scraping blocked/rate limited | Medium | Medium | Delays, caching, respect ToS |
| Feature leakage | Medium | Critical | Time-based splits, feature audit checklist |
| Transfermarkt values unreliable | Medium | Medium | Use as feature, not ground truth |

---

## 9. Evaluation Framework

### 9.1 Metrics

```python
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    brier_score_loss
)

def evaluate_model(y_true, y_pred_proba, k=20):
    """Comprehensive evaluation suite."""

    # Sort by predicted probability
    top_k_idx = np.argsort(y_pred_proba)[-k:]

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "precision_at_k": y_true[top_k_idx].mean(),
        "recall_at_100": y_true[np.argsort(y_pred_proba)[-100:]].sum() / y_true.sum(),
        "brier_score": brier_score_loss(y_true, y_pred_proba),
    }

    return metrics
```

### 9.2 Walk-Forward Validation

```python
def walk_forward_cv(df):
    """Time-based cross-validation."""

    folds = [
        {"train": ["2015-16", "2016-17", "2017-18", "2018-19"],
         "val": "2019-20", "test": "2020-21"},
        {"train": ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20"],
         "val": "2020-21", "test": "2021-22"},
        {"train": ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21"],
         "val": "2021-22", "test": "2022-23"},
    ]

    return folds
```

### 9.3 Case Study Validation

Must validate that model would have flagged:
- Erling Haaland (Salzburg 2019)
- Luis Diaz (Porto 2021)
- Darwin Nunez (Benfica 2021)
- Cody Gakpo (PSV 2022)
- Alexis Mac Allister (Brighton via Argentinos)
- Moises Caicedo (Brighton)

---

## 10. Future Extensions

### Short Term (Post-MVP)
- Position-specific models
- Wage-adjusted value metric
- Injury history integration
- Contract situation features

### Medium Term
- StatsBomb event data
- Video highlight embeddings
- Social/news sentiment signals
- Agent/intermediary network data

### Long Term
- Real-time weekly updates
- Betting market integration (true "undervalued")
- API for third-party integrations
- Mobile app for scouts

---

## 11. Appendix

### A. Known Successful Transfers (Validation Set)

| Player | From | To | Season | Fee |
|--------|------|-----|--------|-----|
| Erling Haaland | RB Salzburg | Dortmund | 2019-20 | €20M |
| Luis Diaz | Porto | Liverpool | 2021-22 | €45M |
| Darwin Nunez | Benfica | Liverpool | 2022-23 | €75M |
| Cody Gakpo | PSV | Liverpool | 2022-23 | €42M |
| Viktor Gyokeres | Coventry | Sporting | 2023-24 | €20M |
| Khvicha Kvaratskhelia | Dinamo Batumi | Napoli | 2022-23 | €10M |

### B. League UEFA Coefficients (2024)

| League | Coefficient | Our Adjustment |
|--------|-------------|----------------|
| Premier League | 90.5 | 1.00 (baseline) |
| La Liga | 76.6 | 0.95 |
| Bundesliga | 73.4 | 0.93 |
| Serie A | 68.7 | 0.90 |
| Ligue 1 | 56.4 | 0.85 |
| Eredivisie | 54.9 | 0.85 |
| Portuguese Liga | 51.4 | 0.82 |
| Belgian Pro League | 36.5 | 0.75 |

### C. Position Mapping

```python
POSITION_GROUPS = {
    "FW": ["ST", "CF", "LW", "RW", "SS"],
    "MF": ["CM", "CAM", "CDM", "LM", "RM", "AM", "DM"],
    "DF": ["CB", "LB", "RB", "LWB", "RWB", "SW"],
}
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-03 | Initial | Document creation |

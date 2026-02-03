# Hidden Gem Finder

ML system to identify undervalued football players in lower European leagues with high potential to succeed at top-5 leagues.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

## Project Structure

```
hidden-gem-finder/
├── config/           # League, feature, model configurations
├── data/             # Raw, processed, and output data
├── notebooks/        # Exploration and analysis notebooks
├── src/              # Source code
│   ├── scrapers/     # Data collection from FBref, Transfermarkt, Understat
│   ├── data/         # Cleaning, merging, labeling
│   ├── features/     # Feature engineering
│   ├── models/       # Training and prediction
│   ├── explainability/  # SHAP analysis, similarity search
│   └── reporting/    # Dashboard and PDF reports
├── tests/            # Test suite
└── outputs/          # Models, reports, predictions
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - AI assistant guidelines
- [PRD.md](PRD.md) - Product requirements document
- [notes/](notes/) - Session notes and decisions

## Target Metrics

| Metric | Target |
|--------|--------|
| ROC-AUC | > 0.75 |
| Precision @ 20 | > 0.30 |
| Recall @ 100 | > 0.50 |

## Development Phases

1. **Data Collection** (Weeks 1-2): Scrapers for FBref, Transfermarkt, Understat
2. **Feature Engineering** (Weeks 3-4): Player matching, derived features
3. **Model Development** (Weeks 5-6): LightGBM, XGBoost, ensemble
4. **Explainability** (Weeks 7-8): SHAP, similarity search
5. **Deployment** (Weeks 9-10): Dashboard, reports

## Source Leagues

- Eredivisie (Netherlands)
- Primeira Liga (Portugal)
- Belgian Pro League
- Championship (England)
- Serie B (Italy)
- Ligue 2 (France)
- Austrian Bundesliga
- Scottish Premiership

## License

Private project - not for distribution.

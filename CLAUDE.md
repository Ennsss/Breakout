# Hidden Gem Finder - Claude Instructions

## Project Overview
Machine learning system to identify undervalued football players in lower leagues with high potential to succeed at top-5 European leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1).

## Development Philosophy (Based on @bcherny's Principles)

### 1. Plan Mode First
- **Always start complex tasks in plan mode** - Pour energy into the plan so implementation can be 1-shot
- If something goes sideways, **immediately switch back to plan mode** - Don't keep pushing
- Use plan mode for verification steps, not just initial builds
- Have a second review pass on plans before implementation

### 2. Work in Parallel
- Structure work to enable multiple Claude sessions on different components
- Use git worktrees or branches for parallel development:
  - `scrapers` branch - Data collection
  - `features` branch - Feature engineering
  - `models` branch - ML training
  - `viz` branch - Dashboard/reporting

### 3. Success Criteria Over Instructions
- Don't tell Claude what to do - give it success criteria and let it loop
- Write tests FIRST, then implement to pass them
- Use assertions and validation checks as guardrails
- Example: "Make the scraper pass all tests in test_scrapers.py" not "Write a scraper that does X, Y, Z"

### 4. Code Quality Rules

#### DO:
- Keep solutions simple and focused
- Write the naive algorithm first, optimize while preserving correctness
- Clean up dead code immediately after refactoring
- Surface tradeoffs and inconsistencies before proceeding
- Ask clarifying questions when requirements are ambiguous
- Push back when an approach seems wrong

#### DON'T:
- Over-engineer or add unnecessary abstractions
- Implement 1000 lines when 100 will do
- Make assumptions without checking - ASK FIRST
- Add features beyond what was requested
- Leave orphaned code, comments, or imports
- Be sycophantic - honest feedback is more valuable

### 5. Error Handling
- When stuck, clearly state what's blocking progress
- Present multiple approaches with tradeoffs
- If a fix is mediocre, say so and propose the elegant solution
- After corrections, note what went wrong to avoid repeating

---

## Project-Specific Guidelines

### Data & Scraping
- **Rate limiting is mandatory** - 2-3 seconds between requests minimum
- **Cache everything** - Store raw HTML/JSON before parsing
- **Player matching is critical** - Use fuzzy matching with multiple identifiers (name + DOB + team)
- Respect robots.txt and terms of service
- Log all scraping activity for debugging

### Feature Engineering
- **Document every derived feature** with formula and rationale
- Always include league difficulty adjustments
- Use per-90 stats, not totals (normalize for playing time)
- Position-group features separately (FW, MF, DF have different key metrics)
- **Beware feature leakage** - Strict time-based splits, audit feature construction

### Model Development
- **Class imbalance is severe** (~5% positive rate) - Use appropriate techniques:
  - SMOTE or class weights
  - Focus on precision at top-K
  - Calibrate probabilities
- Walk-forward validation only - no random splits
- Ensemble models preferred (LightGBM + XGBoost)
- SHAP explanations for every prediction

### Output & Interpretability
- Every flagged player needs an explanation
- Include similar successful players for context
- Confidence scores must be calibrated
- Reports should be actionable for scouts

---

## File Structure Conventions

```
src/
├── scrapers/     # One file per data source
├── data/         # Cleaning, merging, labeling
├── features/     # Feature engineering pipelines
├── models/       # Training, evaluation, prediction
├── explainability/  # SHAP, similarity search
└── reporting/    # Dashboard, PDF generation

data/
├── raw/          # Untouched scraped data
├── processed/    # Cleaned and merged
└── outputs/      # Model predictions, reports

notebooks/        # Numbered exploration notebooks (01_, 02_, etc.)
tests/            # Mirror src/ structure
config/           # YAML configs for leagues, features, model params
```

---

## Testing Standards
- Every scraper must have tests with mock responses
- Feature engineering functions need unit tests with known inputs/outputs
- Model evaluation needs reproducibility tests
- Integration tests for full pipeline

---

## Notes Directory
Maintain `notes/` directory for:
- Session summaries after each major milestone
- Decisions made and rationale
- Bugs encountered and fixes
- Ideas for future improvements

Update after every PR or significant work session.

---

## Common Mistakes to Avoid

1. **Player name matching failures** - Always use multiple identifiers
2. **Data leakage from future** - Triple-check temporal splits
3. **Ignoring league context** - Raw stats are meaningless without adjustment
4. **Over-fitting to specific leagues** - Use league as feature, cross-validate
5. **Treating all positions equally** - Separate models or feature sets
6. **Ignoring class imbalance** - Will get 95% accuracy by predicting "no breakout"
7. **Bloated abstractions** - Keep it simple, refactor only when needed

---

## Prompting Tips for This Project

- "Grill me on this feature engineering approach before I implement it"
- "Prove to me this doesn't have data leakage"
- "What would a staff ML engineer critique about this?"
- "Knowing everything you know now, what's the elegant solution?"
- "What are the top 3 things that could go wrong with this approach?"

---

## Version History
- v0.1 - Initial project setup and planning

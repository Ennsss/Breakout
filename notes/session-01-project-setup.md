# Session 01: Project Setup

**Date:** 2026-02-03
**Focus:** Initial project setup, documentation, and structure

## What Was Done

### 1. Created Core Documentation
- **CLAUDE.md**: Project-specific AI assistant guidelines incorporating Boris Cherny's best practices
- **PRD.md**: Comprehensive product requirements document with:
  - Success criteria and metrics
  - Data requirements and schemas
  - Technical architecture
  - Development phases
  - Risk register

### 2. Project Structure
Created full directory structure:
```
hidden-gem-finder/
├── config/           # YAML configs for leagues, features, model params
├── data/             # raw/, processed/, outputs/
├── notebooks/        # Numbered exploration notebooks
├── src/              # Python source code (scrapers, data, features, models, etc.)
├── tests/            # Test suite
├── notes/            # Session notes (this directory)
└── outputs/          # Model artifacts, reports, predictions
```

### 3. Configuration Files Created
- `config/leagues.yaml`: League metadata, coefficients, position mappings
- `config/features.yaml`: Feature definitions, derived features, position-specific sets
- `config/model_params.yaml`: LightGBM, XGBoost, ensemble settings

### 4. Dependencies
- `requirements.txt`: All Python dependencies for the project

## Key Decisions Made

1. **Target Definition**: A "breakout" is transfer to top-5 league + 900+ minutes played (not just a failed move)
2. **Training Window**: 2015-2022 with 2022-23 as validation
3. **Primary Metric**: Precision @ 20 (scouts care about actionable recommendations)
4. **Ensemble Strategy**: LightGBM + XGBoost weighted average with calibration

## Development Philosophy (from @bcherny)

1. Start complex tasks in plan mode
2. Work in parallel with git worktrees/branches
3. Give success criteria, not instructions
4. Ruthlessly iterate on CLAUDE.md
5. Use subagents for exploration
6. Challenge the output: "Prove to me this works"

## Next Steps

1. **Phase 1 Start**: Build FBref scraper
   - Research FBref page structure
   - Implement base scraper class
   - Create tests with mock responses

2. **Data Collection Priority**:
   - FBref (P0) - primary stats source
   - Transfermarkt (P0) - transfers and values
   - Understat (P1) - detailed xG data

## Open Questions

- [ ] How to handle players who changed positions mid-career?
- [ ] Should we include loan moves to top-5 leagues?
- [ ] What's the minimum minutes threshold at source league?

## Learnings / Notes for CLAUDE.md Updates

- None yet - first session

---

*Next session should focus on FBref scraper implementation*

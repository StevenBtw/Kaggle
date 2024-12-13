# Insurance Premium Prediction

Solution for the [Playground Series - Season 4, Episode 12](https://www.kaggle.com/competitions/playground-series-s4e12) competition on Kaggle.

## Competition Overview

The goal of this competition is to predict insurance premiums based on various customer attributes. This is a regression problem where we need to predict a continuous target variable (Premium Amount) using features like age, health score, and other customer characteristics.

## Solution Approach

### Feature Engineering
- Domain-specific features (risk factors, age groups, etc.)
- Statistical features (rolling statistics, mean ratios)
- Interaction features
- Polynomial features
- Clustering-based features

### Models
- LightGBM
- XGBoost
- CatBoost
- TabNet
- NGBoost
- HistGradientBoosting
- Extra Trees

### Ensemble
- Model stacking with Ridge regression
- Weighted average blending

## Directory Structure

```
├── README.md               # This file
├── main.py                # Main training pipeline
├── models/                # Model implementations
├── common/                # Shared utilities
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── training.py
│   └── utils.py
├── data/                  # Data directory (not tracked)
└── output/                # Model outputs and predictions
```

## Usage

1. Download competition data:
```bash
python download_data.py
```

2. Run the training pipeline:
```bash
python -m insurance.main
```

## Performance

Current best score: 1.04463
Public Leaderboard Position: 132

## Requirements

See [setup.py](setup.py) for detailed requirements.

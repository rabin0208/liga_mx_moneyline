"""
Fit logistic regression to predict whether the closing favorite wins (fav_won).

Train on all seasons before TEST_SEASONS; evaluate on the holdout seasons.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from model_utils import FEATURE_COLS, TEST_SEASONS, print_feature_importance, verify_test_set

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FEATURED_CSV = DATA_DIR / "liga_mx_featured.csv"
TARGET_COL = "fav_won"


def load_and_prepare() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(FEATURED_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    for c in ("home_rest_days", "away_rest_days", "week_of_year", "season_index"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS].astype(float)
    y = df[TARGET_COL].astype(int)
    season = df["season"]
    return X, y, season


def season_split(
    X: pd.DataFrame, y: pd.Series, season: pd.Series, test_seasons: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_mask = ~season.isin(test_seasons)
    test_mask = season.isin(test_seasons)
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def main() -> None:
    print(f"Loading {FEATURED_CSV}...")
    X, y, season = load_and_prepare()
    print(f"  Samples: {len(X)}, features: {len(FEATURE_COLS)}")

    X_train, X_test, y_train, y_test = season_split(X, y, season, TEST_SEASONS)
    verify_test_set(y_test, TEST_SEASONS)
    print(
        f"  Train: {len(y_train)} (not in {TEST_SEASONS}), "
        f"test: {len(y_test)} ({TEST_SEASONS})"
    )
    print(f"  Train fav rate: {y_train.mean():.3f}  test fav rate: {y_test.mean():.3f}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\nFitting logistic regression (predict fav_won)...")
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Test ROC-AUC:  {auc:.4f}")
    print(f"  Test log-loss: {ll:.4f}")
    print(f"  Test Brier:    {brier:.4f}")
    print_feature_importance(
        FEATURE_COLS, model.coef_[0].tolist(), title="Coefficients (abs = importance)"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "logistic_regression.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"\nSaved model and scaler to {MODELS_DIR}/")

    coef = pd.DataFrame(
        {"feature": FEATURE_COLS, "coefficient": model.coef_[0]}
    ).sort_values("coefficient", key=abs, ascending=False)
    coef_path = TABLES_DIR / "logistic_regression_coefficients.csv"
    coef.to_csv(coef_path, index=False)
    print(f"Saved coefficients to {coef_path}")


if __name__ == "__main__":
    main()

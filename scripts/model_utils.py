"""
Shared utilities for Liga MX feature engineering and model fitting.
"""
from __future__ import annotations

LAG_WINDOW = 5

# Holdout seasons for train/test (string labels as in MEX.csv Season column).
# Train = all seasons strictly before the earliest test season (so later years
# are excluded from both train and test — no future leakage).
TEST_SEASONS: list[str] = ["2025/2026", "2026/2027"]


def train_test_season_sets(
    all_seasons: list[str] | set[str],
    test_seasons: list[str] | None = None,
) -> tuple[set[str], set[str]]:
    """Train = seasons before min(test); test = TEST_SEASONS; later years dropped."""
    test = set(test_seasons if test_seasons is not None else TEST_SEASONS)
    first_test = min(test)
    train = {str(s) for s in all_seasons if str(s) < first_test}
    return train, test


def team_lag_column_names() -> list[str]:
    """Per-game lags: win / goals / goals_allowed / goal_diff × home/away."""
    names: list[str] = []
    for side in ("home", "away"):
        for stat in ("win", "goals", "goals_allowed", "goal_diff"):
            for k in range(1, LAG_WINDOW + 1):
                names.append(f"{side}_{stat}_lag_{k}")
    return names


def h2h_lag_column_names() -> list[str]:
    """1 if current home team won that past H2H meeting; 0 otherwise / missing."""
    return [f"home_h2h_win_lag_{k}" for k in range(1, LAG_WINDOW + 1)]


def calendar_feature_names() -> list[str]:
    return ["season_index", "week_of_year", "home_rest_days", "away_rest_days"]


def feature_column_names() -> list[str]:
    return team_lag_column_names() + h2h_lag_column_names() + calendar_feature_names()


FEATURE_COLS: list[str] = feature_column_names()


def lag_vector(values: list, window: int, pad: float = 0.0) -> list[float]:
    """Lag k = k-th most recent game; `values` is chronological (oldest first)."""
    out: list[float] = []
    for k in range(1, window + 1):
        if len(values) >= k:
            out.append(float(values[-k]))
        else:
            out.append(pad)
    return out


def verify_test_set(y_test, test_seasons: list[str]) -> None:
    if len(y_test) == 0:
        raise RuntimeError(
            f"No rows for test seasons {test_seasons}. "
            "Re-run prepare_data.py and build_features.py."
        )


def print_feature_importance(
    feature_names: list[str],
    values: list[float],
    title: str = "Feature importance",
    *,
    limit: int | None = 40,
) -> None:
    pairs = list(zip(feature_names, values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  {title} (sorted by |value|):")
    if limit is not None and len(pairs) > limit:
        pairs = pairs[:limit]
        print(f"    (showing top {limit} of {len(feature_names)})")
    for name, val in pairs:
        print(f"    {name}: {val:.4f}")

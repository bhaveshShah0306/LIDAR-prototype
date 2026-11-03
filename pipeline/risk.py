"""Train the alert model (AI #2) and prove whether it beats the hand-written rule.

Supervised learning, binary classification: features at time t -> "close pass within 2 s?" (0/1).
Model: LightGBM, gradient-boosted decision trees. Fast on a CPU, handles NaN and categories.

Data split BY RIDE (never mix frames of one ride into both train and test):
    meta.json "split": "train"  -> train the model
                       "val"    -> choose the alert threshold
                       "test"   -> final score, touched ONCE at the end

Pipeline position:  features.py -> risk.py -> api.py / alert_out.py

Output: models/risk_lgbm.joblib
        results/alerts.md                     model vs rule table
        results/pr_curve.png
        data/rides/<ride>/video_risk.parquet  frame, t, left_level, right_level (for the dashboard/LEDs)

Usage:
    uv add scikit-learn lightgbm matplotlib
    uv run python pipeline/risk.py

Done when: results/alerts.md compares model and rule on the TEST ride: precision, recall,
false alarms per minute, median warning time.
"""
import json
from pathlib import Path

import pandas as pd

from features import FEATURES

ROOT = Path(__file__).resolve().parent.parent
RIDES, MODELS, RESULTS = ROOT / "data" / "rides", ROOT / "models", ROOT / "results"


def load_split(split):
    """All feature rows from rides whose meta.json has this split.

    TODO:
      1. For every ride folder in RIDES with video_features.parquet:
         read meta.json; if meta.get("split") == split, read the parquet.
      2. pd.concat them (ignore_index=True). Return an empty DataFrame if none.
      3. Make sure "zone" and "class" are category dtype AFTER concatenating
         (categories must be the same in train and test; pd.Categorical with fixed categories helps).
    """
    raise NotImplementedError("TODO: load_split")


def train(train_df):
    """Fit and return an LGBMClassifier.

    TODO:
      1. from lightgbm import LGBMClassifier
      2. model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                class_weight="balanced")   # positives are rare
      3. model.fit(train_df[FEATURES], train_df["label"]); return model

    Check yourself: what does class_weight="balanced" do, and why do rare positives need it?
    """
    raise NotImplementedError("TODO: train")


def choose_threshold(model, val_df, min_recall=0.8):
    """Lowest-false-alarm threshold that still catches at least min_recall of close passes.

    TODO:
      1. p = model.predict_proba(val_df[FEATURES])[:, 1]
      2. from sklearn.metrics import precision_recall_curve
         precision, recall, thresholds = precision_recall_curve(val_df["label"], p)
      3. Among thresholds with recall >= min_recall, pick the one with the highest precision.
      4. Return it (a float).
    """
    raise NotImplementedError("TODO: choose_threshold")


def evaluate(y_true, alert, t, ride):
    """Numbers for the README, from per-row 0/1 alerts.

    TODO:
      1. precision, recall (sklearn.metrics.precision_score / recall_score)
      2. false alarms per minute: count alerts where y_true == 0, merge alerts on consecutive
         frames of the same ride into one alarm, divide by total minutes of video
      3. warning time: for each true event, seconds between the first alert and the event
         (start simple: report precision, recall, false alarms/min; add warning time after)
      4. Return a dict.
    """
    raise NotImplementedError("TODO: evaluate")


def rule_alerts(test_df):
    """The rule's alert for each row, so both are scored the same way.

    TODO: alert = (ttc_s < 2.5) | (dist_m < 4)  — the same thresholds as rules.py's amber level.
    """
    raise NotImplementedError("TODO: rule_alerts")


if __name__ == "__main__":
    train_df, val_df, test_df = load_split("train"), load_split("val"), load_split("test")
    print(f"rows  train {len(train_df)}  val {len(val_df)}  test {len(test_df)}")

    model = train(train_df)
    threshold = choose_threshold(model, val_df)
    p_test = model.predict_proba(test_df[FEATURES])[:, 1]

    model_scores = evaluate(test_df["label"], p_test >= threshold, test_df["t"], test_df["ride"])
    rule_scores = evaluate(test_df["label"], rule_alerts(test_df), test_df["t"], test_df["ride"])
    table = pd.DataFrame({"rule": rule_scores, "model": model_scores})
    print(table.to_string())

    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    # TODO: joblib.dump({"model": model, "threshold": threshold, "features": FEATURES},
    #                   MODELS / "risk_lgbm.joblib")
    # TODO: write table.to_markdown() to RESULTS / "alerts.md" (uv add tabulate) and a PR curve PNG
    # TODO: per ride, turn the model's row alerts into frame-level left/right levels
    #       (same idea as rules.frame_levels) and save video_risk.parquet for the dashboard and LEDs
    print(json.dumps({"threshold": threshold}))

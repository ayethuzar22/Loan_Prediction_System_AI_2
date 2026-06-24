"""
=============================================================
PART 4: MODEL TRAINING
=============================================================
Trains Logistic Regression, Random Forest, and XGBoost.
Compares all three using Accuracy, Precision, Recall, F1, ROC-AUC.
Saves the best model for production use.
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb

# Import our preprocessing pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import run_preprocessing   # type: ignore

logger   = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────

def build_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,          # adjust if imbalanced
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",          # faster training
        ),
    }


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_model(
    name: str,
    model,
    X_train, y_train,
    X_test, y_test,
) -> dict:
    """Train and evaluate a single model. Returns metrics dict."""

    # ── Train ──
    if name == "XGBoost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    # ── Predict ──
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    # ── Metrics ──
    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred),         4),
        "precision": round(precision_score(y_test, y_pred),        4),
        "recall":    round(recall_score(y_test, y_pred),           4),
        "f1":        round(f1_score(y_test, y_pred),               4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob),          4),
    }

    # ── Cross-validation (5-fold AUC) ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring="roc_auc", n_jobs=-1)
    metrics["cv_auc_mean"] = round(cv_auc.mean(), 4)
    metrics["cv_auc_std"]  = round(cv_auc.std(),  4)

    # ── Print report ──
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1        : {metrics['f1']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")
    print(f"  CV-AUC    : {metrics['cv_auc_mean']} ± {metrics['cv_auc_std']}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Rejected','Approved'])}")

    return metrics


# ─────────────────────────────────────────────────────────────
# COMPARE & SELECT BEST
# ─────────────────────────────────────────────────────────────

def compare_models(results: list[dict]) -> str:
    """Return the name of the model with highest ROC-AUC."""
    df = pd.DataFrame(results).set_index("model")
    print("\n" + "=" * 55)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 55)
    print(df[["accuracy", "precision", "recall", "f1", "roc_auc"]].to_string())

    best = df["roc_auc"].idxmax()
    print(f"\n  ✓ Best model by ROC-AUC: {best} ({df.loc[best,'roc_auc']})")
    return best


# ─────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE (XGBoost / RF)
# ─────────────────────────────────────────────────────────────

def print_feature_importance(model, feature_names: list, top_n: int = 15):
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_names)
        fi = fi.sort_values(ascending=False).head(top_n)
        print(f"\nTop {top_n} Feature Importances:")
        for feat, imp in fi.items():
            bar = "█" * int(imp * 40)
            print(f"  {feat:<35} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────
# SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────

def save_best_model(
    best_name: str,
    trained_models: dict,
    selected_features: list,
    metrics: dict,
):
    """Save model, metadata, and feature list."""
    model = trained_models[best_name]

    # Primary artifact
    model_path = MODELS_DIR / "loan_model.joblib"
    joblib.dump(model, model_path)
    print(f"\n[Save] Model → {model_path}")

    # Metadata
    meta = {
        "model_name":         best_name,
        "trained_at":         datetime.now(datetime.UTC).isoformat(),
        "selected_features":  selected_features,
        "metrics":            metrics,
        "version":            "1.0.0",
    }
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Save] Metadata → {meta_path}")

    # Versioned backup
    version_dir = MODELS_DIR / "versions"
    version_dir.mkdir(exist_ok=True)
    ts = datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    versioned = version_dir / f"loan_model_{ts}.joblib"
    joblib.dump(model, versioned)
    print(f"[Save] Versioned → {versioned}")

    return model_path


# ─────────────────────────────────────────────────────────────
# MASTER TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────

def train_all(kaggle_csv_path: str) -> str:
    """Full pipeline: preprocess → train → compare → save best."""

    X_tr, X_te, y_tr, y_te, scaler, feats = run_preprocessing(
        kaggle_path=kaggle_csv_path,
        do_feature_selection=True,
    )

    models  = build_models()
    results = []
    trained = {}

    for name, model in models.items():
        metrics = evaluate_model(name, model, X_tr, y_tr, X_te, y_te)
        results.append(metrics)
        trained[name] = model

    best_name = compare_models(results)
    best_metrics = next(r for r in results if r["model"] == best_name)

    # Feature importance for tree models
    print_feature_importance(trained[best_name], feats)

    model_path = save_best_model(best_name, trained, feats, best_metrics)
    print(f"\n✓ Training complete. Best model saved to: {model_path}")
    return model_path


# ─────────────────────────────────────────────────────────────
# Run standalone
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    kaggle_csv = DATA_DIR / "myanmar_loan_mock_data.csv"
    if kaggle_csv.exists():
        train_all(str(kaggle_csv))
    else:
        print(f"Place Kaggle CSV at: {kaggle_csv}")
        print("Download from: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset")
# if __name__ == "__main__":
#     import sys
#
#     sys.path.insert(0, str(Path(__file__).parent))
#
#     kaggle_csv = DATA_DIR / "loan_approval_dataset.csv"
#
#     # 🌟 ဒီစာကြောင်းကို ထည့်ပြီး Python တကယ် ရှာနေတဲ့ ပတ်လမ်းကြောင်းကို စစ်ပါ
#     print(f"DEBUG: Python is looking for the file at: {kaggle_csv.resolve()}")
#
#     if kaggle_csv.exists():
#         train_all(str(kaggle_csv))
#     else:
#         print(f"Place Kaggle CSV at: {kaggle_csv}")
#         print("Download from: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset")
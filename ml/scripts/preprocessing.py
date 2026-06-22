"""
=============================================================
PART 3: DATA PREPROCESSING PIPELINE
=============================================================
Merges the Kaggle dataset with custom DB-derived features,
handles missing values, outliers, encoding, scaling, and
produces a clean training-ready DataFrame.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import logging

logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 1 — Load Kaggle dataset
# ─────────────────────────────────────────────────────────────

def load_kaggle(filepath: str) -> pd.DataFrame:
    """
    Loads and minimally cleans the Kaggle CSV.
    Column names are normalised to snake_case.
    """
    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
    )

    # Drop pure identifier columns
    df.drop(columns=["loan_id"], errors="ignore", inplace=True)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    print(f"[Kaggle] Loaded {len(df)} rows × {len(df.columns)} cols")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — Encode target variable
# ─────────────────────────────────────────────────────────────

def encode_target(df: pd.DataFrame, col: str = "loan_status") -> pd.DataFrame:
    """Approved=1, Rejected=0."""
    df = df.copy()
    df["target"] = (df[col].str.lower() == "approved").astype(int)
    df.drop(columns=[col], inplace=True)
    print(f"[Target] Distribution:\n{df['target'].value_counts()}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — Encode categorical features
# ─────────────────────────────────────────────────────────────

KAGGLE_CAT_COLS = ["education", "self_employed"]

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Binary encode known Kaggle categorical columns."""
    df = df.copy()

    if "education" in df.columns:
        df["education"] = (df["education"].str.lower() == "graduate").astype(int)

    if "self_employed" in df.columns:
        df["self_employed"] = (df["self_employed"].str.lower() == "yes").astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# STEP 4 — Handle missing values
# ─────────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
    - Numeric  → median imputation  (robust against outliers)
    - Categoric→ most frequent      (already encoded to int by now)
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col   = "target"
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    num_imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"Remaining nulls after imputation:\n{missing[missing>0]}")

    print(f"[Missing] After imputation: {df.isnull().sum().sum()} nulls remain")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 5 — Remove duplicates
# ─────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"[Dedup] Removed {before - len(df)} duplicates → {len(df)} rows")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 6 — Outlier treatment (IQR capping)
# ─────────────────────────────────────────────────────────────

OUTLIER_COLS = [
    "income_annum", "loan_amount", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value",
    "income_average", "requested_amount", "total_outstanding",
]

def cap_outliers(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    IQR-based winsorization: cap values at [Q1 - 1.5×IQR, Q3 + 1.5×IQR].
    This preserves all rows but limits extreme values.
    """
    df    = df.copy()
    cols  = cols or [c for c in OUTLIER_COLS if c in df.columns]

    for col in cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        clipped = df[col].clip(lower, upper)
        changed = (clipped != df[col]).sum()
        if changed:
            print(f"[Outlier] {col}: capped {changed} values")
        df[col] = clipped

    return df


# ─────────────────────────────────────────────────────────────
# STEP 7 — Merge Kaggle + custom DB features
# ─────────────────────────────────────────────────────────────

def merge_with_db_features(
    kaggle_df: pd.DataFrame,
    db_features: list[dict],
) -> pd.DataFrame:
    """
    Horizontally concatenates Kaggle rows with custom-engineered
    DB features.  When training on Kaggle data only (no real DB),
    db_features can be an empty list — defaults/zeros are used.

    In production, this is NOT needed: you build features purely
    from the DB (see feature engineering script).
    """
    if not db_features:
        print("[Merge] No DB features — using Kaggle-only dataset")
        return kaggle_df

    db_df = pd.DataFrame(db_features)
    # Both must have same index length for horizontal merge
    assert len(kaggle_df) == len(db_df), (
        "Kaggle and DB feature counts must match for row-wise merge"
    )
    merged = pd.concat([kaggle_df.reset_index(drop=True),
                        db_df.reset_index(drop=True)], axis=1)
    print(f"[Merge] Combined shape: {merged.shape}")
    return merged


# ─────────────────────────────────────────────────────────────
# STEP 8 — Feature scaling
# ─────────────────────────────────────────────────────────────

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on train, transform both train and test.
    Saves scaler for later use in inference.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
    )

    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"[Scaler] Saved to {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────────────────────
# STEP 9 — Feature selection
# ─────────────────────────────────────────────────────────────

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    threshold: str = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Uses a shallow Random Forest to select features whose importance
    is above the threshold (default = mean importance).
    Returns reduced train/test DataFrames + list of selected features.
    """
    selector_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector_model.fit(X_train, y_train)

    selector = SelectFromModel(selector_model, threshold=threshold, prefit=True)
    selected_mask   = selector.get_support()
    selected_cols   = X_train.columns[selected_mask].tolist()

    X_train_sel = X_train[selected_cols]
    X_test_sel  = X_test[selected_cols]

    print(f"[Feature Selection] {len(selected_cols)}/{len(X_train.columns)} features kept:")
    for c in selected_cols:
        print(f"   • {c}")

    # Save feature list
    feature_path = MODELS_DIR / "selected_features.joblib"
    joblib.dump(selected_cols, feature_path)
    print(f"[Feature Selection] Saved to {feature_path}")

    return X_train_sel, X_test_sel, selected_cols


# ─────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────

def run_preprocessing(
    kaggle_path: str,
    db_features: list[dict] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    do_feature_selection: bool = True,
) -> tuple:
    """
    Full end-to-end preprocessing pipeline.
    Returns: X_train, X_test, y_train, y_test, scaler, selected_features
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE START")
    print("=" * 60)

    # 1. Load
    df = load_kaggle(kaggle_path)

    # 2. Encode target
    df = encode_target(df)

    # 3. Encode categoricals
    df = encode_categoricals(df)

    # 4. Merge with DB features (optional)
    if db_features:
        df = merge_with_db_features(df, db_features)

    # 5. Handle missing
    df = handle_missing(df)

    # 6. Remove duplicates
    df = remove_duplicates(df)

    # 7. Cap outliers
    df = cap_outliers(df)

    # 8. Split
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"[Split] Train: {len(X_train)} | Test: {len(X_test)}")

    # 9. Scale
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # 10. Feature selection
    selected_features = X_train_sc.columns.tolist()
    if do_feature_selection:
        X_train_sc, X_test_sc, selected_features = select_features(
            X_train_sc, y_train, X_test_sc
        )

    print("\n[Pipeline] COMPLETE ✓")
    return X_train_sc, X_test_sc, y_train, y_test, scaler, selected_features


# ─────────────────────────────────────────────────────────────
# Run standalone
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kaggle_csv = DATA_DIR / "loan_approval_dataset.csv"
    if kaggle_csv.exists():
        X_tr, X_te, y_tr, y_te, scaler, feats = run_preprocessing(str(kaggle_csv))
        print(f"\nFinal training shape: {X_tr.shape}")
        print(f"Final test shape:     {X_te.shape}")
    else:
        print(f"Place Kaggle CSV at: {kaggle_csv}")

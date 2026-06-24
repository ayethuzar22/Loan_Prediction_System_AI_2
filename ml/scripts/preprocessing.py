# """
# =============================================================
# PART 3: DATA PREPROCESSING PIPELINE
# =============================================================
# Merges the Kaggle dataset with custom DB-derived features,
# handles missing values, outliers, encoding, scaling, and
# produces a clean training-ready DataFrame.
# """
#
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# import joblib
# import logging
#
# logger = logging.getLogger(__name__)
#
# DATA_DIR   = Path(__file__).parent.parent / "data"
# MODELS_DIR = Path(__file__).parent.parent / "models"
# MODELS_DIR.mkdir(parents=True, exist_ok=True)
#
#
# # ─────────────────────────────────────────────────────────────
# # STEP 1 — Load Kaggle dataset
# # ─────────────────────────────────────────────────────────────
#
# def load_kaggle(filepath: str) -> pd.DataFrame:
#     """
#     Loads and minimally cleans the Kaggle CSV.
#     Column names are normalised to snake_case.
#     """
#     df = pd.read_csv(filepath)
#
#     # Normalize column names
#     df.columns = (
#         df.columns.str.strip()
#                   .str.lower()
#                   .str.replace(" ", "_")
#                   .str.replace("-", "_")
#     )
#
#     # Drop pure identifier columns
#     df.drop(columns=["loan_id"], errors="ignore", inplace=True)
#
#     # Strip whitespace from string columns
#     for col in df.select_dtypes(include="object").columns:
#         df[col] = df[col].str.strip()
#
#     print(f"[Kaggle] Loaded {len(df)} rows × {len(df.columns)} cols")
#     return df
#
#
# # ─────────────────────────────────────────────────────────────
# # STEP 2 — Encode target variable
# # ─────────────────────────────────────────────────────────────
#
# def encode_target(df: pd.DataFrame, col: str = "loan_status") -> pd.DataFrame:
#     """Approved=1, Rejected=0."""
#     df = df.copy()
#     df["target"] = (df[col].str.lower() == "approved").astype(int)
#     df.drop(columns=[col], inplace=True)
#     print(f"[Target] Distribution:\n{df['target'].value_counts()}")
#     return df
#
#
# # ─────────────────────────────────────────────────────────────
# # STEP 3 — Encode categorical features
# # ─────────────────────────────────────────────────────────────
#
# KAGGLE_CAT_COLS = ["education", "self_employed"]
#
# def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
#     """Binary encode known Kaggle categorical columns."""
#     df = df.copy()
#
#     if "education" in df.columns:
#         df["education"] = (df["education"].str.lower() == "graduate").astype(int)
#
#     if "self_employed" in df.columns:
#         df["self_employed"] = (df["self_employed"].str.lower() == "yes").astype(int)
#
#     return df
#
#
# # ─────────────────────────────────────────────────────────────
# # STEP 4 — Handle missing values
# # ─────────────────────────────────────────────────────────────
#
# def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Strategy:
#     - Numeric  → median imputation  (robust against outliers)
#     - Categoric→ most frequent      (already encoded to int by now)
#     """
#     df = df.copy()
#
#     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     target_col   = "target"
#     if target_col in numeric_cols:
#         numeric_cols.remove(target_col)
#
#     num_imputer = SimpleImputer(strategy="median")
#     df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
#
#     missing = df.isnull().sum()
#     if missing.sum() > 0:
#         logger.warning(f"Remaining nulls after imputation:\n{missing[missing>0]}")
#
#     print(f"[Missing] After imputation: {df.isnull().sum().sum()} nulls remain")
#     return df
#
#
# # ─────────────────────────────────────────────────────────────
# # STEP 5 — Remove duplicates
# # ─────────────────────────────────────────────────────────────
#
# def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
#     before = len(df)
#     df = df.drop_duplicates()
#     print(f"[Dedup] Removed {before - len(df)} duplicates → {len(df)} rows")
#     return df
#
#
# # ─────────────────────────────────────────────────────────────
# # STEP 6 — Outlier treatment (IQR capping)
# # ─────────────────────────────────────────────────────────────
#
# OUTLIER_COLS = [
#     "income_annum", "loan_amount", "residential_assets_value",
#     "commercial_assets_value", "luxury_assets_value", "bank_asset_value",
#     "income_average", "requested_amount", "total_outstanding",
# ]
#
# def cap_outliers(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
#     """
#     IQR-based winsorization: cap values at [Q1 - 1.5×IQR, Q3 + 1.5×IQR].
#     This preserves all rows but limits extreme values.
#     """
#     df    = df.copy()
#     cols  = cols or [c for c in OUTLIER_COLS if c in df.columns]
#
#     for col in cols:
#         Q1  = df[col].quantile(0.25)
#         Q3  = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
#         clipped = df[col].clip(lower, upper)
#         changed = (clipped != df[col]).sum()
#         if changed:
#             print(f"[Outlier] {col}: capped {changed} values")
#         df[col] = clipped
#
#     return df
#
#
# # ─────────────────────────────────────────────────────────────
# # STEP 7 — Merge Kaggle + custom DB features
# # ─────────────────────────────────────────────────────────────
#
# def merge_with_db_features(
#     kaggle_df: pd.DataFrame,
#     db_features: list[dict],
# ) -> pd.DataFrame:
#     """
#     Horizontally concatenates Kaggle rows with custom-engineered
#     DB features.  When training on Kaggle data only (no real DB),
#     db_features can be an empty list — defaults/zeros are used.
#
#     In production, this is NOT needed: you build features purely
#     from the DB (see feature engineering script).
#     """
#     if not db_features:
#         print("[Merge] No DB features — using Kaggle-only dataset")
#         return kaggle_df
#
#     db_df = pd.DataFrame(db_features)
#     # Both must have same index length for horizontal merge
#     assert len(kaggle_df) == len(db_df), (
#         "Kaggle and DB feature counts must match for row-wise merge"
#     )
#     merged = pd.concat([kaggle_df.reset_index(drop=True),
#                         db_df.reset_index(drop=True)], axis=1)
#     print(f"[Merge] Combined shape: {merged.shape}")
#     return merged
#
#
# # ─────────────────────────────────────────────────────────────
# # STEP 8 — Feature scaling
# # ─────────────────────────────────────────────────────────────
#
# def scale_features(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
# ) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
#     """
#     Fit StandardScaler on train, transform both train and test.
#     Saves scaler for later use in inference.
#     """
#     scaler = StandardScaler()
#     X_train_scaled = pd.DataFrame(
#         scaler.fit_transform(X_train),
#         columns=X_train.columns,
#     )
#     X_test_scaled = pd.DataFrame(
#         scaler.transform(X_test),
#         columns=X_test.columns,
#     )
#
#     scaler_path = MODELS_DIR / "scaler.joblib"
#     joblib.dump(scaler, scaler_path)
#     print(f"[Scaler] Saved to {scaler_path}")
#
#     return X_train_scaled, X_test_scaled, scaler
#
#
# # ─────────────────────────────────────────────────────────────
# # STEP 9 — Feature selection
# # ─────────────────────────────────────────────────────────────
#
# def select_features(
#     X_train: pd.DataFrame,
#     y_train: pd.Series,
#     X_test: pd.DataFrame,
#     threshold: str = "mean",
# ) -> tuple[pd.DataFrame, pd.DataFrame, list]:
#     """
#     Uses a shallow Random Forest to select features whose importance
#     is above the threshold (default = mean importance).
#     Returns reduced train/test DataFrames + list of selected features.
#     """
#     selector_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#     selector_model.fit(X_train, y_train)
#
#     selector = SelectFromModel(selector_model, threshold=threshold, prefit=True)
#     selected_mask   = selector.get_support()
#     selected_cols   = X_train.columns[selected_mask].tolist()
#
#     X_train_sel = X_train[selected_cols]
#     X_test_sel  = X_test[selected_cols]
#
#     print(f"[Feature Selection] {len(selected_cols)}/{len(X_train.columns)} features kept:")
#     for c in selected_cols:
#         print(f"   • {c}")
#
#     # Save feature list
#     feature_path = MODELS_DIR / "selected_features.joblib"
#     joblib.dump(selected_cols, feature_path)
#     print(f"[Feature Selection] Saved to {feature_path}")
#
#     return X_train_sel, X_test_sel, selected_cols
#
#
# # ─────────────────────────────────────────────────────────────
# # MASTER PIPELINE
# # ─────────────────────────────────────────────────────────────
#
# def run_preprocessing(
#     kaggle_path: str,
#     db_features: list[dict] = None,
#     test_size: float = 0.2,
#     random_state: int = 42,
#     do_feature_selection: bool = True,
# ) -> tuple:
#     """
#     Full end-to-end preprocessing pipeline.
#     Returns: X_train, X_test, y_train, y_test, scaler, selected_features
#     """
#     print("\n" + "=" * 60)
#     print("PREPROCESSING PIPELINE START")
#     print("=" * 60)
#
#     # 1. Load
#     df = load_kaggle(kaggle_path)
#
#     # 2. Encode target
#     df = encode_target(df)
#
#     # 3. Encode categoricals
#     df = encode_categoricals(df)
#
#     # 4. Merge with DB features (optional)
#     if db_features:
#         df = merge_with_db_features(df, db_features)
#
#     # 5. Handle missing
#     df = handle_missing(df)
#
#     # 6. Remove duplicates
#     df = remove_duplicates(df)
#
#     # 7. Cap outliers
#     df = cap_outliers(df)
#
#     # 8. Split
#     X = df.drop(columns=["target"])
#     y = df["target"]
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, stratify=y, random_state=random_state
#     )
#     print(f"[Split] Train: {len(X_train)} | Test: {len(X_test)}")
#
#     # 9. Scale
#     X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
#
#     # 10. Feature selection
#     selected_features = X_train_sc.columns.tolist()
#     if do_feature_selection:
#         X_train_sc, X_test_sc, selected_features = select_features(
#             X_train_sc, y_train, X_test_sc
#         )
#
#     print("\n[Pipeline] COMPLETE ✓")
#     return X_train_sc, X_test_sc, y_train, y_test, scaler, selected_features
#
#
# # ─────────────────────────────────────────────────────────────
# # Run standalone
# # ─────────────────────────────────────────────────────────────
#
# if __name__ == "__main__":
#     kaggle_csv = DATA_DIR / "loan_approval_dataset.csv"
#     if kaggle_csv.exists():
#         X_tr, X_te, y_tr, y_te, scaler, feats = run_preprocessing(str(kaggle_csv))
#         print(f"\nFinal training shape: {X_tr.shape}")
#         print(f"Final test shape:     {X_te.shape}")
#     else:
#         print(f"Place Kaggle CSV at: {kaggle_csv}")


"""
=============================================================
PART 3 (UPDATED): DATA PREPROCESSING PIPELINE — Myanmar Microfinance
=============================================================
Merges the Kaggle/synthetic dataset with Myanmar-specific features,
handles missing values, outliers, encoding, scaling, and produces
a clean training-ready DataFrame.

Loan categories:
  1 → Agriculture  (has_form_7, land_size_acre, seasonal_income)
  2 → MSME         (has_business_license, years_in_business, daily_cash_flow)
  3 → Consumer     (fixed_monthly_salary, has_collateral)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
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
# CATEGORY FEATURE SCHEMAS
# Keys = column names expected in the CSV.
# Values = default fill value (used when a row belongs to a
#          different category and those columns are absent/NaN).
# ─────────────────────────────────────────────────────────────

CATEGORY_FEATURES = {
    1: {   # Agriculture
        "has_form_7":            0,
        "land_size_acre":        0.0,
        "seasonal_income":       0.0,
        "seasonal_income_ratio": 0.0,
        "crop_paddy":            0,
        "crop_fruit":            0,
        "crop_vegetable":        0,
    },
    2: {   # MSME
        "has_business_license":   0,
        "years_in_business":      0.0,
        "daily_cash_flow":        0.0,
        "monthly_cash_flow":      0.0,
        "cashflow_to_loan_ratio": 0.0,
        "biz_retail":             0,
        "biz_wholesale":          0,
        "biz_manufacturing":      0,
    },
    3: {   # Consumer
        "fixed_monthly_salary":  0.0,
        "has_collateral":        0,
        "salary_to_installment": 0.0,
        "purpose_vehicle":       0,
        "purpose_housing":       0,
        "purpose_other":         0,
    },
}

# Shared numeric columns subject to outlier capping
OUTLIER_COLS = [
    "income_average", "requested_amount", "total_outstanding",
    "land_size_acre", "seasonal_income",
    "daily_cash_flow", "monthly_cash_flow",
    "fixed_monthly_salary",
]

# Columns that must never be scaled (flags and encoded categories)
PASSTHROUGH_COLS = [
    "category_id",
    "has_form_7", "crop_paddy", "crop_fruit", "crop_vegetable",
    "has_business_license", "biz_retail", "biz_wholesale", "biz_manufacturing",
    "has_collateral", "purpose_vehicle", "purpose_housing", "purpose_other",
    "sub_cat_paddy", "sub_cat_orchard", "sub_cat_retail", "sub_cat_wholesale",
    "sub_cat_manufacturing", "sub_cat_vehicle", "sub_cat_housing",
    "education", "self_employed", "gender_male",
    "ownership_exists", "business_license_exists",
    "occ_government", "occ_employee", "occ_business",
    "occ_self_employed", "occ_farmer", "occ_freelancer",
]


# ─────────────────────────────────────────────────────────────
# STEP 1 — Load dataset (Kaggle CSV or synthetic Myanmar CSV)
# ─────────────────────────────────────────────────────────────

def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
    )
    df.drop(columns=["loan_id"], errors="ignore", inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    print(f"[Load] {len(df)} rows × {len(df.columns)} cols")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — Encode target
# ─────────────────────────────────────────────────────────────

def encode_target(df):
    # ညီမရဲ့ CSV ထဲက Target ကော်လံအမည်အမှန်ကို တိုက်ရိုက်သတ်မှတ်ခြင်း
    col = "loan_status_target"

    # အကယ်၍ ကော်လံအမည် ရှေ့/နောက်မှာ Space ပါနေရင် အလိုအလျောက် ဖျက်ပေးရန်
    df.columns = df.columns.str.strip()

    # အကယ်၍ အကြောင်းအမျိုးမျိုးကြောင့် loan_status_target နာမည် မတွေ့ခဲ့ရင်
    # အလိုအလျောက် နောက်ဆုံးကော်လံကို ယူသုံးဖို့ Back-up အနေနဲ့ ထည့်ထားပေးပါတယ်
    if col  not in df.columns:
        col = df.columns[-1]

    # ဒေတာအမျိုးအစားအလိုက် target (0 သို့မဟုတ် 1) အဖြစ် ပြောင်းလဲခြင်း
    if df[col].dtype in ['int64', 'int32', 'float64', 'int']:
        # အကယ်၍ ၁ နဲ့ ၀ ဖြစ်နေပြီးသားဆိုရင် တိုက်ရိုက်ယူသုံးမည်
        df["target"] = df[col].astype(int)
    else:
        # အကယ်၍ စာသား (Approved/Rejected) ဖြစ်နေလျှင် 0, 1 ပြောင်းမည်
        df["target"] = (df[col].astype(str).str.lower().str.strip() == "approved").astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — Encode shared categoricals
# ─────────────────────────────────────────────────────────────

def encode_shared_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "education" in df.columns:
        df["education"] = (df["education"].str.lower() == "graduate").astype(int)
    if "self_employed" in df.columns:
        df["self_employed"] = (df["self_employed"].str.lower() == "yes").astype(int)
    if "gender" in df.columns:
        df["gender_male"] = (df["gender"].str.lower() == "male").astype(int)
        df.drop(columns=["gender"], inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 4 — Fill category-specific missing columns
# ─────────────────────────────────────────────────────────────

def fill_category_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all category-specific feature columns exist in the DataFrame.
    For rows belonging to a different category, the column value should
    already be 0/0.0 (set during synthetic data generation or by the
    feature engine). Any remaining NaN is filled with the schema default.
    """
    df = df.copy()
    for cat_id, schema in CATEGORY_FEATURES.items():
        for col, default in schema.items():
            if col not in df.columns:
                df[col] = default
                logger.info(f"[CategoryFill] Added missing column '{col}' with default {default}")
            else:
                # Fill NaN only for rows of the OTHER categories
                # (rows of this category should have real values)
                mask_other = df["category_id"] != cat_id
                df.loc[mask_other & df[col].isna(), col] = default

    print(f"[CategoryFill] All category columns present.")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 5 — Derive computed columns
# ─────────────────────────────────────────────────────────────

def derive_computed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ratio/engineered columns that depend on multiple raw columns.
    Safe to call after NaN-filling.
    """
    df = df.copy()
    income = df.get("income_average", pd.Series(1.0, index=df.index)).clip(lower=1)

    if "requested_amount" in df.columns:
        df["loan_to_income_ratio"] = (df["requested_amount"] / income).round(4)
        tenure = df.get("loan_tenure_months", pd.Series(1, index=df.index)).clip(lower=1)
        df["debt_to_income_ratio"] = (df["requested_amount"] / tenure / income).round(4)

    # Agriculture: seasonal_income_ratio
    if "seasonal_income" in df.columns and "requested_amount" in df.columns:
        denom = df["requested_amount"].clip(lower=1)
        df["seasonal_income_ratio"] = (df["seasonal_income"] / denom).round(4)

    # MSME: monthly_cash_flow and cashflow_to_loan_ratio
    if "daily_cash_flow" in df.columns:
        df["monthly_cash_flow"] = df["daily_cash_flow"] * 30
        if "requested_amount" in df.columns:
            installment = (df["requested_amount"] / tenure).clip(lower=1)
            df["cashflow_to_loan_ratio"] = (df["monthly_cash_flow"] / installment).round(4)

    # Consumer: salary_to_installment
    if "fixed_monthly_salary" in df.columns and "requested_amount" in df.columns:
        installment = (df["requested_amount"] / tenure).clip(lower=1)
        df["salary_to_installment"] = (df["fixed_monthly_salary"] / installment).round(4)

    return df


# ─────────────────────────────────────────────────────────────
# STEP 6 — Handle missing values
# ─────────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != "target"
    ]
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    remaining = df.isnull().sum().sum()
    print(f"[Missing] After imputation: {remaining} nulls remain")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 7 — Remove duplicates
# ─────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"[Dedup] Removed {before - len(df)} duplicates → {len(df)} rows")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 8 — Outlier capping (IQR winsorization)
# ─────────────────────────────────────────────────────────────

def cap_outliers(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    df   = df.copy()
    cols = cols or [c for c in OUTLIER_COLS if c in df.columns]
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        clipped = df[col].clip(lower, upper)
        changed = (clipped != df[col]).sum()
        if changed:
            print(f"[Outlier] {col}: capped {changed} values")
        df[col] = clipped
    return df


# ─────────────────────────────────────────────────────────────
# STEP 9 — Feature scaling
# ─────────────────────────────────────────────────────────────

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    """
    StandardScaler applied only to continuous numeric columns.
    Binary flags and one-hot columns are passed through unchanged.
    """
    scale_cols = [
        c for c in X_train.select_dtypes(include=[np.number]).columns
        if c not in PASSTHROUGH_COLS
    ]
    passthrough = [c for c in X_train.columns if c not in scale_cols]

    scaler = StandardScaler()
    X_train_sc = X_train.copy()
    X_test_sc  = X_test.copy()
    X_train_sc[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_sc[scale_cols]  = scaler.transform(X_test[scale_cols])

    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"[Scaler] Saved → {scaler_path}  (scaled {len(scale_cols)} cols)")
    return X_train_sc, X_test_sc, scaler


# ─────────────────────────────────────────────────────────────
# STEP 10 — Feature selection
# ─────────────────────────────────────────────────────────────

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    threshold: str = "mean",
) -> tuple:
    selector_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector_model.fit(X_train, y_train)
    selector = SelectFromModel(selector_model, threshold=threshold, prefit=True)
    selected_cols = X_train.columns[selector.get_support()].tolist()

    print(f"[FeatureSelection] {len(selected_cols)}/{len(X_train.columns)} features kept:")
    for c in selected_cols:
        print(f"   • {c}")

    feat_path = MODELS_DIR / "selected_features.joblib"
    joblib.dump(selected_cols, feat_path)
    print(f"[FeatureSelection] Saved → {feat_path}")

    return X_train[selected_cols], X_test[selected_cols], selected_cols


# ─────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────

def run_preprocessing(
    dataset_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
    do_feature_selection: bool = True,
    kaggle_path: str = None,
) -> tuple:
    """
    Full end-to-end preprocessing pipeline for Myanmar microfinance data.
    Returns: X_train, X_test, y_train, y_test, scaler, selected_features
    """
    print("\n" + "=" * 60)
    print("MYANMAR MICROFINANCE PREPROCESSING PIPELINE START")
    print("=" * 60)

    final_path = dataset_path if dataset_path is not None else kaggle_path

    if final_path is None:
        raise ValueError("Either dataset_path or kaggle_path must be provided to run preprocessing.")

    df = load_dataset(final_path)
    df = encode_target(df)
    df = encode_shared_categoricals(df)
    df = fill_category_columns(df)
    df = derive_computed_columns(df)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = cap_outliers(df)

    X = df.drop(columns=["target", "loan_status_target"], errors="ignore")
    y = df["target"]

    # Drop any remaining object columns (safety net)
    obj_cols = X.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        logger.warning(f"Dropping remaining object columns: {obj_cols}")
        X.drop(columns=obj_cols, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"[Split] Train: {len(X_train)} | Test: {len(X_test)}")

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    selected_features = X_train_sc.columns.tolist()
    if do_feature_selection:
        X_train_sc, X_test_sc, selected_features = select_features(
            X_train_sc, y_train, X_test_sc
        )

    print("\n[Pipeline] COMPLETE ✓")
    return X_train_sc, X_test_sc, y_train, y_test, scaler, selected_features


if __name__ == "__main__":
    csv = DATA_DIR / "myanmar_loan_mock_data.csv"
    if csv.exists():
        X_tr, X_te, y_tr, y_te, scaler, feats = run_preprocessing(str(csv))
        print(f"\nFinal training shape : {X_tr.shape}")
        print(f"Final test shape     : {X_te.shape}")
    else:
        print(f"Place dataset CSV at: {csv}")
        print("Run generate_mock_data.py first.")
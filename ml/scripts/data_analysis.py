"""
=============================================================
PART 1: DATA UNDERSTANDING
=============================================================
Analyzes both Kaggle dataset and the Myanmar LMS database schema.

KAGGLE DATASET FEATURES (loan_approval_prediction.csv):
  loan_id               - USELESS: just an identifier
  no_of_dependents      - USEFUL: more dependents = more financial burden
  education             - USEFUL: Graduate vs Not Graduate → income stability proxy
  self_employed         - USEFUL: self-employed = more income volatility
  income_annum          - USEFUL: primary repayment capacity signal
  loan_amount           - USEFUL: the amount being requested
  loan_tenure           - USEFUL: repayment period
  cibil_score           - USEFUL: credit history score (0-900 in India, maps to risk)
  residential_assets_value  - USEFUL: collateral proxy
  commercial_assets_value   - USEFUL: collateral proxy
  luxury_assets_value       - USEFUL: collateral proxy
  bank_asset_value          - USEFUL: liquidity proxy
  loan_status               - TARGET VARIABLE (Approved / Rejected)

TARGET VARIABLE: loan_status → binary 1=Approved, 0=Rejected

DATABASE SCHEMA FEATURES:
  FROM document_table:
    income_average        - USEFUL: monthly income (core feature)
    occupation            - USEFUL: employment type
    ownership_image       - USEFUL: has property collateral
    business_license_*    - USEFUL: has business = stable income signal

  FROM history_table:
    amount_request        - USEFUL: requested amount
    month                 - USEFUL: requested tenure
    status                - USEFUL: past loan outcomes

  FROM guarantor_table:
    guarantor count       - USEFUL: more guarantors = lower risk

  FROM list_table:
    monthly_savings       - USEFUL: savings behavior
    remain_amount         - USEFUL: outstanding debt
    remain_month          - USEFUL: remaining obligation
    flag                  - USEFUL: overdue flag (repayment discipline)

FEATURES TO DROP:
  - loan_id, user_id, document_id (identifiers)
  - nrc_photo_*, household_list_photo_* (raw image paths - not ML features)
  - face_scan_photo, notification data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def load_kaggle_data(filepath: str) -> pd.DataFrame:
    """Load and do initial inspection of Kaggle dataset."""
    df = pd.read_csv(filepath)
    
    print("=" * 60)
    print("KAGGLE DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nTarget Distribution:\n{df['loan_status'].value_counts(normalize=True)}")
    print(f"\nSample:\n{df.head()}")
    
    return df


def analyze_features(df: pd.DataFrame):
    """Correlation analysis and feature importance preview."""
    # Encode target
    df = df.copy()
    df['loan_status_enc'] = (df['loan_status'].str.strip() == 'Approved').astype(int)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\n" + "=" * 60)
    print("FEATURE CORRELATION WITH TARGET")
    print("=" * 60)
    corr = df[numeric_cols].corr()['loan_status_enc'].drop('loan_status_enc')
    print(corr.sort_values(ascending=False))
    
    return corr


if __name__ == "__main__":
    # Example usage - point to your downloaded Kaggle CSV
    kaggle_path = DATA_DIR / "loan_approval_dataset.csv"
    if kaggle_path.exists():
        df = load_kaggle_data(str(kaggle_path))
        analyze_features(df)
    else:
        print(f"Place Kaggle CSV at: {kaggle_path}")
        print("Download from: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset")

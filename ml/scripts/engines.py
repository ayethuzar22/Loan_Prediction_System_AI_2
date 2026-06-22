"""
=============================================================
PARTS 5–9: PREDICTION ENGINES
=============================================================
  Part 5 – Risk Scoring Engine
  Part 6 – Loan Amount Recommendation
  Part 7 – Interest Rate Engine
  Part 8 – Rejection Reason Engine
  Part 9 – Model Export / Load / Predict
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).parent.parent / "models"


# ═════════════════════════════════════════════════════════════
# PART 5 — RISK SCORING ENGINE
# ═════════════════════════════════════════════════════════════

class RiskScoringEngine:
    """
    Converts raw approval probability + features into:
      - score      : 0–100 (higher = riskier)
      - risk_level : "LOW" | "MEDIUM" | "HIGH"

    Scoring Formula
    ───────────────
    Base risk = (1 - approval_probability) × 60     [0–60]

    Penalty factors (each adds to risk):
      +10  if loan_to_income_ratio > 4
      + 8  if debt_to_income_ratio > 0.4
      + 7  if late_payment_count > 2
      + 6  if active_loan_count > 1
      - 5  if ownership_exists
      - 5  if guarantor_count >= 2
      - 3  if business_license_exists
      - 4  if repayment_score > 80

    Risk Levels
    ───────────
      LOW    :  0–30
      MEDIUM : 31–60
      HIGH   : 61–100
    """

    THRESHOLDS = {"LOW": (0, 30), "MEDIUM": (31, 60), "HIGH": (61, 100)}

    def compute(
        self,
        approval_probability: float,
        features: dict,
    ) -> dict:
        # ── Base score from model probability ──────────────────
        base = (1.0 - approval_probability) * 60.0

        # ── Penalty / Bonus factors ────────────────────────────
        delta = 0.0

        lti = features.get("loan_to_income_ratio", 0)
        if lti > 4:
            delta += 10
        elif lti > 2:
            delta += 4

        dti = features.get("debt_to_income_ratio", 0)
        if dti > 0.4:
            delta += 8
        elif dti > 0.25:
            delta += 3

        late = features.get("late_payment_count", 0)
        if late > 2:
            delta += 7
        elif late > 0:
            delta += 3

        active = features.get("active_loan_count", 0)
        if active > 1:
            delta += 6
        elif active == 1:
            delta += 2

        if features.get("ownership_exists", 0):
            delta -= 5

        if features.get("guarantor_count", 0) >= 2:
            delta -= 5
        elif features.get("guarantor_count", 0) == 1:
            delta -= 2

        if features.get("business_license_exists", 0):
            delta -= 3

        repayment_score = features.get("repayment_score", 50)
        if repayment_score > 80:
            delta -= 4
        elif repayment_score < 40:
            delta += 5

        stability = features.get("customer_stability_score", 50)
        if stability > 75:
            delta -= 4
        elif stability < 30:
            delta += 5

        # ── Final score ────────────────────────────────────────
        raw_score = base + delta
        score = int(max(0, min(100, round(raw_score))))

        # ── Risk level ─────────────────────────────────────────
        if score <= 30:
            risk_level = "LOW"
        elif score <= 60:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        factors = self._describe_factors(features, delta)

        return {
            "score":      score,
            "risk_level": risk_level,
            "factors":    factors,
        }

    def _describe_factors(self, features: dict, delta: float) -> list:
        """Return a human-readable list of the most impactful risk factors."""
        factors = []

        if features.get("loan_to_income_ratio", 0) > 4:
            factors.append("Loan amount is very high relative to income")
        if features.get("debt_to_income_ratio", 0) > 0.4:
            factors.append("Monthly debt obligations exceed 40% of income")
        if features.get("late_payment_count", 0) > 2:
            factors.append("Multiple late payments recorded")
        if features.get("active_loan_count", 0) > 1:
            factors.append("Multiple active loans outstanding")
        if not features.get("ownership_exists", 0):
            factors.append("No property collateral provided")
        if features.get("repayment_score", 50) < 40:
            factors.append("Poor repayment track record")

        return factors or ["No significant risk factors identified"]


# ═════════════════════════════════════════════════════════════
# PART 6 — LOAN AMOUNT RECOMMENDATION
# ═════════════════════════════════════════════════════════════

class LoanAmountRecommender:
    """
    Recommends a safe loan amount based on:
      income × tenure multiplier × risk discount × history bonus

    Formula
    ───────
    base_capacity    = income_average × tenure_months × 0.35
    risk_discount    = 1 - (risk_score / 200)        [50%→0.75, 30%→0.85, 80%→0.60]
    history_bonus    = 1 + previous_approval_rate × 0.15
    repayment_bonus  = 1 + (repayment_score / 1000)
    collateral_bonus = 1.10 if ownership else 1.0

    recommended = base_capacity × risk_discount × history_bonus
                               × repayment_bonus × collateral_bonus

    Cap: min(recommended, requested_amount)   (never recommend more than asked)
    Floor: income × 2                          (always at least 2× monthly income)
    """

    def recommend(
        self,
        income_average: float,
        tenure_months: int,
        risk_score: int,
        repayment_score: float,
        previous_approval_rate: float,
        ownership_exists: int,
        requested_amount: float,
        active_loan_count: int,
        total_outstanding: float,
    ) -> float:
        income = max(income_average, 1.0)

        # Base capacity: customer can repay ~35% of income per month
        base_capacity = income * tenure_months * 0.35

        # Risk discount (higher risk → lower amount)
        risk_discount = 1.0 - (risk_score / 200.0)   # 0.50–1.00

        # History bonus
        history_bonus = 1.0 + previous_approval_rate * 0.15

        # Repayment behaviour bonus
        repayment_bonus = 1.0 + (repayment_score / 1000.0)

        # Collateral bonus
        collateral_bonus = 1.10 if ownership_exists else 1.00

        # Active debt penalty
        active_penalty = max(0, 1 - (active_loan_count * 0.10))

        recommended = (
            base_capacity
            * risk_discount
            * history_bonus
            * repayment_bonus
            * collateral_bonus
            * active_penalty
        )

        # Subtract outstanding debt
        recommended = recommended - total_outstanding

        # Apply bounds
        floor = income * 2
        recommended = max(floor, min(recommended, requested_amount))

        return round(recommended, -3)   # round to nearest 1,000 MMK


# ═════════════════════════════════════════════════════════════
# PART 7 — INTEREST RATE ENGINE
# ═════════════════════════════════════════════════════════════

# class InterestRateEngine:
#     """
#     Assigns monthly interest rate based on risk level.
#
#     Business Rules (monthly flat rate):
#       LOW    → 1.5%
#       MEDIUM → 2.0%
#       HIGH   → 2.5%
#
#     Also calculates monthly installment:
#       installment = (principal × monthly_rate) / (1 - (1 + monthly_rate)^(-tenure))
#     """
#
#     RATES = {
#         "LOW":    0.015,
#         "MEDIUM": 0.020,
#         "HIGH":   0.025,
#     }
#
#     def get_rate(self, risk_level: str) -> float:
#         return self.RATES.get(risk_level.upper(), 0.020)
#
#     def monthly_installment(
#         self,
#         principal: float,
#         risk_level: str,
#         tenure_months: int,
#     ) -> float:
#         """Reducing-balance EMI formula."""
#         r = self.get_rate(risk_level)
#         n = tenure_months
#
#         if r == 0 or n == 0:
#             return round(principal / max(n, 1), 0)
#
#         emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
#         return round(emi, 0)

class InterestRateEngine:
    """
    Sathapana Limited Myanmar Microfinance Interest Rate System.
    Fixed Annual Interest Rate: 28% (p.a.) for all loan types.
    Repayment Method: Reducing Balance (Effective Interest Rate) Method.
    """

    # Sathapana ရဲ့ တစ်နှစ်စာ ပုံသေအတိုးနှုန်း 28% ကို လစဉ်အတိုးနှုန်းအဖြစ် ပြောင်းလဲတွက်ချက်ခြင်း
    ANNUAL_RATE = 0.28
    MONTHLY_RATE = 0.28 / 12  # ~0.02333 (သို့မဟုတ်) 2.333% per month

    def get_rate(self, risk_level: str = "MEDIUM") -> float:
        """
        Sathapana စနစ်တွင် Risk Level အပေါ်မတည်ဘဲ အတိုးနှုန်းမှာ တစ်ပြေးညီ 28% p.a. ဖြစ်သည်။
        (API တည်ငြိမ်မှုရှိစေရန် function အမည်နှင့် argument ကို မူရင်းအတိုင်း ထားရှိပါသည်)
        """
        return self.MONTHLY_RATE

    def monthly_installment(
            self,
            principal: float,
            risk_level: str,
            tenure_months: int,
    ) -> float:
        """
        Reducing-balance EMI formula based on Sathapana's 28% annual rate.
        လစဉ် အရစ်ကျ ပေးသွင်းရမည့်ငွေ (အရင်း + အတိုး) ကို တွက်ချက်ခြင်း။
        """
        r = self.MONTHLY_RATE
        n = tenure_months

        # သုညဖြင့် စားမိခြင်း သို့မဟုတ် လ အရေအတွက် မရှိခြင်းများအတွက် အကာအကွယ်ပေးရန်
        if r == 0 or n == 0:
            return round(principal / max(n, 1), 0)

        # ကမ္ဘာသုံး လျှော့ကနုတ် အရစ်ကျတွက်ချက်မှု ပုံသေနည်း (Sathapana သုံး စနစ်)
        emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

        return round(emi, 0)  # လစဉ်ပေးသွင်းရမည့်ငွေကို အနီးစပ်ဆုံး ကျပ်ပြည့် ဖြတ်ပေးမည်

# ═════════════════════════════════════════════════════════════
# PART 8 — REJECTION REASON ENGINE
# ═════════════════════════════════════════════════════════════

class RejectionReasonEngine:
    """
    Generates human-readable rejection reasons from the feature vector.
    Thresholds are tuned for Myanmar microfinance context.
    """

    def generate(self, features: dict, approval_probability: float) -> list:
        reasons = []

        income = features.get("income_average", 0)
        lti    = features.get("loan_to_income_ratio", 0)
        dti    = features.get("debt_to_income_ratio", 0)
        rep    = features.get("repayment_score", 100)
        late   = features.get("late_payment_count", 0)
        active = features.get("active_loan_count", 0)
        own    = features.get("ownership_exists", 0)
        guar   = features.get("guarantor_count", 0)
        prev_approval = features.get("previous_approval_rate", 1)
        stability = features.get("customer_stability_score", 100)

        # ── Income checks ──────────────────────────────────────
        if income < 150_000:
            reasons.append("Income is too low to qualify for a loan (minimum 150,000 MMK/month)")
        elif income < 250_000 and lti > 5:
            reasons.append("Low income relative to the requested loan amount")

        # ── Loan-to-income ratio ───────────────────────────────
        if lti > 8:
            reasons.append(f"Loan amount is too high — {lti:.1f}× monthly income (maximum 8×)")
        elif lti > 5:
            reasons.append("Requested amount exceeds recommended loan-to-income limit")

        # ── Debt burden ────────────────────────────────────────
        if dti > 0.5:
            reasons.append(f"Monthly debt obligations exceed 50% of income (current: {dti*100:.0f}%)")
        elif dti > 0.4:
            reasons.append("High debt-to-income ratio — existing commitments are too high")

        # ── Repayment history ──────────────────────────────────
        if rep < 30:
            reasons.append("Very poor repayment history — multiple missed or late payments")
        elif rep < 50:
            reasons.append("Below-average repayment score")

        if late > 3:
            reasons.append(f"Too many late payments in repayment history ({late} occurrences)")

        # ── Active loans ───────────────────────────────────────
        if active > 2:
            reasons.append(f"Too many active loans ({active}) — maximum 2 concurrent loans allowed")
        elif active > 1:
            reasons.append("Existing active loan must be partially repaid before a new application")

        # ── Collateral ─────────────────────────────────────────
        if not own and guar == 0:
            reasons.append("No collateral and no guarantor provided — insufficient security")
        elif not own and lti > 4:
            reasons.append("Property collateral required for loans above 4× monthly income")

        # ── Credit history ─────────────────────────────────────
        if prev_approval < 0.3 and features.get("previous_loan_count", 0) > 2:
            reasons.append("Majority of previous loan applications were rejected")

        # ── Customer stability ─────────────────────────────────
        if stability < 25:
            reasons.append("Overall customer financial stability score is too low")

        # Fallback if no specific reason triggered
        if not reasons and approval_probability < 0.5:
            reasons.append("Application does not meet the minimum creditworthiness criteria")

        return reasons


# ═════════════════════════════════════════════════════════════
# PART 9 — MODEL EXPORT / LOAD / PREDICT
# ═════════════════════════════════════════════════════════════

class LoanPredictor:
    """
    Production inference wrapper.
    Loads model + scaler + feature list from disk.
    Runs the full prediction pipeline.
    """

    def __init__(self, models_dir: str = None):
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.model      = None
        self.scaler     = None
        self.features   = None
        self.metadata   = {}
        self._load()

    def _load(self):
        """Load all model artifacts from disk."""
        model_path   = self.models_dir / "loan_model.joblib"
        scaler_path  = self.models_dir / "scaler.joblib"
        feature_path = self.models_dir / "selected_features.joblib"
        meta_path    = self.models_dir / "model_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run training first."
            )

        self.model    = joblib.load(model_path)
        self.scaler   = joblib.load(scaler_path)   if scaler_path.exists()  else None
        self.features = joblib.load(feature_path)  if feature_path.exists() else None

        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)

        logger.info(f"Model loaded: {self.metadata.get('model_name', 'unknown')} "
                    f"v{self.metadata.get('version', '?')}")

    def predict(self, features: dict) -> dict:
        """
        Run full prediction pipeline.

        Input:  flat feature dict from LoanFeatureEngine
        Output: dict with approval, probability, raw proba vector
        """
        # Build DataFrame with correct column order
        feature_cols = self.features or list(features.keys())
        row = {col: features.get(col, 0.0) for col in feature_cols}
        X   = pd.DataFrame([row])

        # Scale if scaler is available
        if self.scaler is not None:
            X = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
            )

        # Predict
        prob_vector   = self.model.predict_proba(X)[0]
        approval_prob = float(prob_vector[1])
        approved      = bool(approval_prob >= 0.50)

        return {
            "approved":           approved,
            "approval_probability": round(approval_prob, 4),
        }


# ─────────────────────────────────────────────────────────────
# CONVENIENCE: SAVE MODEL (called after training)
# ─────────────────────────────────────────────────────────────

def save_model(model, scaler, selected_features: list, metadata: dict = None):
    """Save all model artifacts. Call this after training."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model,             MODELS_DIR / "loan_model.joblib")
    joblib.dump(scaler,            MODELS_DIR / "scaler.joblib")
    joblib.dump(selected_features, MODELS_DIR / "selected_features.joblib")

    if metadata:
        with open(MODELS_DIR / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"[ModelExport] All artifacts saved to {MODELS_DIR}")


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_features = {
        "income_average":           450_000,
        "loan_to_income_ratio":     4.4,
        "debt_to_income_ratio":     0.21,
        "requested_amount":         2_000_000,
        "loan_tenure_months":       24,
        "loan_category_id":         2,
        "guarantor_count":          2,
        "monthly_savings":          100_000,
        "remain_amount":            200_000,
        "remain_month":             6,
        "late_payment_count":       1,
        "repayment_score":          72,
        "active_loan_count":        1,
        "total_outstanding":        200_000,
        "previous_loan_count":      2,
        "previous_approval_rate":   1.0,
        "ownership_exists":         1,
        "business_license_exists":  1,
        "occupation_risk_score":    0.35,
        "occ_business":             1,
        "gender_male":              1,
        "customer_stability_score": 68,
    }

    approval_prob = 0.78   # simulated (replace with LoanPredictor.predict())

    # Part 5: Risk score
    risk_engine = RiskScoringEngine()
    risk = risk_engine.compute(approval_prob, sample_features)
    print(f"\n[Risk]     score={risk['score']}  level={risk['risk_level']}")

    # Part 6: Loan recommendation
    recommender = LoanAmountRecommender()
    rec_amount = recommender.recommend(
        income_average          = sample_features["income_average"],
        tenure_months           = sample_features["loan_tenure_months"],
        risk_score              = risk["score"],
        repayment_score         = sample_features["repayment_score"],
        previous_approval_rate  = sample_features["previous_approval_rate"],
        ownership_exists        = sample_features["ownership_exists"],
        requested_amount        = sample_features["requested_amount"],
        active_loan_count       = sample_features["active_loan_count"],
        total_outstanding       = sample_features["total_outstanding"],
    )
    print(f"[Recommend] Recommended Amount: {rec_amount:,.0f} MMK")

    # Part 7: Interest rate
    rate_engine  = InterestRateEngine()
    monthly_rate = rate_engine.get_rate(risk["risk_level"])
    installment  = rate_engine.monthly_installment(rec_amount, risk["risk_level"], 24)
    print(f"[Rate]     {monthly_rate*100:.1f}%/month  EMI={installment:,.0f} MMK")

    # Part 8: Rejection reasons (will be empty since approved)
    reason_engine = RejectionReasonEngine()
    reasons = reason_engine.generate(sample_features, approval_prob)
    print(f"[Reasons]  {reasons}")

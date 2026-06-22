"""
=============================================================
PART 2: FEATURE ENGINEERING
=============================================================
Generates meaningful ML features from the Myanmar LMS database.
All features are computed from raw DB tables and assembled
into a single feature vector per loan application.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA CONTAINERS  (mirror DB rows as plain dicts / dataclasses)
# ─────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    user_id: int
    full_name: str
    gender: str          # "Male" / "Female"
    status: str          # "pending" / "approved" / "rejected"


@dataclass
class DocumentInfo:
    income_average: float        # monthly income in MMK
    occupation: str              # e.g. "Government", "Self-Employed", "Business"
    ownership_image: Optional[str]            # path → None means no asset
    business_license_photo_front: Optional[str]
    business_license_photo_back: Optional[str]


@dataclass
class LoanApplication:
    amount_request: float        # requested amount in MMK
    month: int                   # requested tenure in months
    category_id: int             # loan category


@dataclass
class GuarantorRecord:
    guarantor_id: int


@dataclass
class RepaymentRecord:
    amount: float
    monthly_savings: float
    flag: int                    # 0=on-time, 1=late
    remain_amount: float
    remain_month: int


@dataclass
class HistoryRecord:
    status: str                  # "approved" / "rejected" / "pending"
    amount_request: float


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING ENGINE
# ─────────────────────────────────────────────────────────────

class LoanFeatureEngine:
    """
    Converts raw database records into a flat feature dictionary
    suitable for XGBoost prediction.
    """

    # Occupation risk multipliers  (lower = more stable)
    OCCUPATION_RISK = {
        "government":     0.1,
        "military":       0.1,
        "employee":       0.2,
        "teacher":        0.2,
        "nurse":          0.2,
        "doctor":         0.15,
        "engineer":       0.2,
        "business":       0.35,
        "self-employed":  0.45,
        "farmer":         0.50,
        "freelancer":     0.55,
        "unemployed":     0.95,
        "other":          0.40,
    }

    def __init__(
        self,
        user: UserProfile,
        document: DocumentInfo,
        application: LoanApplication,
        guarantors: List[GuarantorRecord],
        repayment_history: List[RepaymentRecord],
        loan_history: List[HistoryRecord],
    ):
        self.user = user
        self.doc = document
        self.app = application
        self.guarantors = guarantors
        self.repayments = repayment_history
        self.history = loan_history

    # ── Individual feature groups ─────────────────────────────

    def _income_features(self) -> Dict[str, float]:
        income = max(self.doc.income_average, 1.0)   # avoid div/0
        loan_amount = self.app.amount_request

        loan_to_income_ratio = loan_amount / income

        # Monthly installment estimate at flat 2% for ratio calculation
        estimated_monthly = loan_amount / max(self.app.month, 1)
        debt_to_income_ratio = estimated_monthly / income

        return {
            "income_average":       income,
            "loan_to_income_ratio": round(loan_to_income_ratio, 4),
            "debt_to_income_ratio": round(debt_to_income_ratio, 4),
        }

    def _loan_features(self) -> Dict[str, float]:
        return {
            "requested_amount": self.app.amount_request,
            "loan_tenure_months": self.app.month,
            "loan_category_id": self.app.category_id,
        }

    def _guarantor_features(self) -> Dict[str, int]:
        return {
            "guarantor_count": len(self.guarantors),
        }

    def _repayment_features(self) -> Dict[str, float]:
        if not self.repayments:
            return {
                "monthly_savings":       0.0,
                "remain_amount":         0.0,
                "remain_month":          0,
                "late_payment_count":    0,
                "repayment_score":       50.0,   # neutral for new customers
                "active_loan_count":     0,
                "total_outstanding":     0.0,
            }

        late_count = sum(1 for r in self.repayments if r.flag == 1)
        total_records = len(self.repayments)
        on_time_rate = (total_records - late_count) / total_records

        # Repayment score: 0-100
        # Base = on_time_rate × 80 → penalties for outstanding debt
        base_score = on_time_rate * 80
        avg_remain_amount = np.mean([r.remain_amount for r in self.repayments])
        # Penalise up to 20 points for large remaining debt relative to savings
        avg_savings = np.mean([r.monthly_savings for r in self.repayments]) or 1
        debt_penalty = min(20, (avg_remain_amount / (avg_savings * 12)) * 10)
        repayment_score = max(0, min(100, base_score - debt_penalty))

        # Active loans = records where remain_month > 0
        active_loans = sum(1 for r in self.repayments if r.remain_month > 0)

        latest = self.repayments[-1]    # most recent record

        return {
            "monthly_savings":    latest.monthly_savings,
            "remain_amount":      latest.remain_amount,
            "remain_month":       latest.remain_month,
            "late_payment_count": late_count,
            "repayment_score":    round(repayment_score, 2),
            "active_loan_count":  active_loans,
            "total_outstanding":  sum(r.remain_amount for r in self.repayments),
        }

    def _history_features(self) -> Dict[str, float]:
        if not self.history:
            return {
                "previous_loan_count":    0,
                "previous_approval_rate": 0.5,   # neutral prior
            }

        approved = sum(1 for h in self.history if h.status == "approved")
        total = len(self.history)

        return {
            "previous_loan_count":    total,
            "previous_approval_rate": round(approved / total, 4),
        }

    def _collateral_features(self) -> Dict[str, int]:
        return {
            "ownership_exists":        int(bool(self.doc.ownership_image)),
            "business_license_exists": int(
                bool(self.doc.business_license_photo_front)
                or bool(self.doc.business_license_photo_back)
            ),
        }

    def _occupation_features(self) -> Dict[str, float]:
        occ = self.doc.occupation.lower().strip()
        risk = self.OCCUPATION_RISK.get(occ, self.OCCUPATION_RISK["other"])

        # One-hot for most common occupations
        categories = [
            "government", "employee", "business",
            "self-employed", "farmer", "freelancer",
        ]
        occ_encoded = {f"occ_{c.replace('-','_')}": int(occ == c) for c in categories}
        occ_encoded["occupation_risk_score"] = risk
        return occ_encoded

    def _demographic_features(self) -> Dict[str, int]:
        return {
            "gender_male": int(self.user.gender.lower() == "male"),
        }

    def _stability_score(
        self,
        income_feats: Dict,
        repayment_feats: Dict,
        collateral_feats: Dict,
        history_feats: Dict,
    ) -> Dict[str, float]:
        """
        Composite customer stability score (0–100).

        Weights:
          repayment_score        → 35%
          previous_approval_rate → 20%
          collateral bonus       → 15%
          income sufficiency     → 20%
          guarantor bonus        → 10%
        """
        repayment_score   = repayment_feats["repayment_score"]
        approval_rate     = history_feats["previous_approval_rate"] * 100
        collateral_bonus  = (
            collateral_feats["ownership_exists"] * 10
            + collateral_feats["business_license_exists"] * 5
        )
        income            = min(100, (income_feats["income_average"] / 500_000) * 100)
        # If loan_to_income_ratio > 5 → penalise
        lti_penalty       = max(0, (income_feats["loan_to_income_ratio"] - 5) * 5)
        guarantor_bonus   = min(10, len(self.guarantors) * 5)

        stability = (
            repayment_score   * 0.35
            + approval_rate   * 0.20
            + collateral_bonus
            + income          * 0.20
            - lti_penalty
            + guarantor_bonus
        )
        stability = max(0, min(100, stability))

        return {"customer_stability_score": round(stability, 2)}

    # ── Public API ─────────────────────────────────────────────

    def build(self) -> Dict[str, Any]:
        """
        Assemble the complete feature vector.
        Returns a flat dict ready for model inference.
        """
        income_feats     = self._income_features()
        loan_feats       = self._loan_features()
        guarantor_feats  = self._guarantor_features()
        repayment_feats  = self._repayment_features()
        history_feats    = self._history_features()
        collateral_feats = self._collateral_features()
        occupation_feats = self._occupation_features()
        demographic_feats = self._demographic_features()
        stability_feats  = self._stability_score(
            income_feats, repayment_feats, collateral_feats, history_feats
        )

        features = {
            **income_feats,
            **loan_feats,
            **guarantor_feats,
            **repayment_feats,
            **history_feats,
            **collateral_feats,
            **occupation_feats,
            **demographic_feats,
            **stability_feats,
        }

        logger.debug(f"Built feature vector with {len(features)} features")
        return features

    def to_dataframe(self) -> pd.DataFrame:
        """Return features as a single-row DataFrame."""
        return pd.DataFrame([self.build()])


# ─────────────────────────────────────────────────────────────
# DEMO / UNIT TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = LoanFeatureEngine(
        user=UserProfile(1, "Ko Aung", "Male", "pending"),
        document=DocumentInfo(
            income_average=450_000,
            occupation="Business",
            ownership_image="path/to/house.jpg",
            business_license_photo_front="path/to/biz.jpg",
            business_license_photo_back=None,
        ),
        application=LoanApplication(
            amount_request=2_000_000,
            month=24,
            category_id=2,
        ),
        guarantors=[GuarantorRecord(1), GuarantorRecord(2)],
        repayment_history=[
            RepaymentRecord(500_000, 100_000, 0, 300_000, 12),
            RepaymentRecord(500_000, 120_000, 1, 200_000, 6),
            RepaymentRecord(500_000, 130_000, 0, 0, 0),
        ],
        loan_history=[
            HistoryRecord("approved", 1_000_000),
            HistoryRecord("approved", 1_500_000),
        ],
    )

    features = engine.build()
    print("\n" + "=" * 60)
    print("GENERATED FEATURE VECTOR")
    print("=" * 60)
    for k, v in features.items():
        print(f"  {k:<35} {v}")
    print(f"\nTotal features: {len(features)}")

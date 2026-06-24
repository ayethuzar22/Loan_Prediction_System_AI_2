# """
# =============================================================
# PART 2: FEATURE ENGINEERING
# =============================================================
# Generates meaningful ML features from the Myanmar LMS database.
# All features are computed from raw DB tables and assembled
# into a single feature vector per loan application.
# """
#
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass, field, asdict
# from typing import Optional, List, Dict, Any
# import logging
#
# logger = logging.getLogger(__name__)
#
#
# # ─────────────────────────────────────────────────────────────
# # DATA CONTAINERS  (mirror DB rows as plain dicts / dataclasses)
# # ─────────────────────────────────────────────────────────────
#
# @dataclass
# class UserProfile:
#     user_id: int
#     full_name: str
#     gender: str          # "Male" / "Female"
#     status: str          # "pending" / "approved" / "rejected"
#
#
# @dataclass
# class DocumentInfo:
#     income_average: float        # monthly income in MMK
#     occupation: str              # e.g. "Government", "Self-Employed", "Business"
#     ownership_image: Optional[str]            # path → None means no asset
#     business_license_photo_front: Optional[str]
#     business_license_photo_back: Optional[str]
#
#
# @dataclass
# class LoanApplication:
#     amount_request: float        # requested amount in MMK
#     month: int                   # requested tenure in months
#     category_id: int             # loan category
#
#
# @dataclass
# class GuarantorRecord:
#     guarantor_id: int
#
#
# @dataclass
# class RepaymentRecord:
#     amount: float
#     monthly_savings: float
#     flag: int                    # 0=on-time, 1=late
#     remain_amount: float
#     remain_month: int
#
#
# @dataclass
# class HistoryRecord:
#     status: str                  # "approved" / "rejected" / "pending"
#     amount_request: float
#
#
# # ─────────────────────────────────────────────────────────────
# # FEATURE ENGINEERING ENGINE
# # ─────────────────────────────────────────────────────────────
#
# class LoanFeatureEngine:
#     """
#     Converts raw database records into a flat feature dictionary
#     suitable for XGBoost prediction.
#     """
#
#     # Occupation risk multipliers  (lower = more stable)
#     OCCUPATION_RISK = {
#         "government":     0.1,
#         "military":       0.1,
#         "employee":       0.2,
#         "teacher":        0.2,
#         "nurse":          0.2,
#         "doctor":         0.15,
#         "engineer":       0.2,
#         "business":       0.35,
#         "self-employed":  0.45,
#         "farmer":         0.50,
#         "freelancer":     0.55,
#         "unemployed":     0.95,
#         "other":          0.40,
#     }
#
#     def __init__(
#         self,
#         user: UserProfile,
#         document: DocumentInfo,
#         application: LoanApplication,
#         guarantors: List[GuarantorRecord],
#         repayment_history: List[RepaymentRecord],
#         loan_history: List[HistoryRecord],
#     ):
#         self.user = user
#         self.doc = document
#         self.app = application
#         self.guarantors = guarantors
#         self.repayments = repayment_history
#         self.history = loan_history
#
#     # ── Individual feature groups ─────────────────────────────
#
#     def _income_features(self) -> Dict[str, float]:
#         income = max(self.doc.income_average, 1.0)   # avoid div/0
#         loan_amount = self.app.amount_request
#
#         loan_to_income_ratio = loan_amount / income
#
#         # Monthly installment estimate at flat 2% for ratio calculation
#         estimated_monthly = loan_amount / max(self.app.month, 1)
#         debt_to_income_ratio = estimated_monthly / income
#
#         return {
#             "income_average":       income,
#             "loan_to_income_ratio": round(loan_to_income_ratio, 4),
#             "debt_to_income_ratio": round(debt_to_income_ratio, 4),
#         }
#
#     def _loan_features(self) -> Dict[str, float]:
#         return {
#             "requested_amount": self.app.amount_request,
#             "loan_tenure_months": self.app.month,
#             "loan_category_id": self.app.category_id,
#         }
#
#     def _guarantor_features(self) -> Dict[str, int]:
#         return {
#             "guarantor_count": len(self.guarantors),
#         }
#
#     def _repayment_features(self) -> Dict[str, float]:
#         if not self.repayments:
#             return {
#                 "monthly_savings":       0.0,
#                 "remain_amount":         0.0,
#                 "remain_month":          0,
#                 "late_payment_count":    0,
#                 "repayment_score":       50.0,   # neutral for new customers
#                 "active_loan_count":     0,
#                 "total_outstanding":     0.0,
#             }
#
#         late_count = sum(1 for r in self.repayments if r.flag == 1)
#         total_records = len(self.repayments)
#         on_time_rate = (total_records - late_count) / total_records
#
#         # Repayment score: 0-100
#         # Base = on_time_rate × 80 → penalties for outstanding debt
#         base_score = on_time_rate * 80
#         avg_remain_amount = np.mean([r.remain_amount for r in self.repayments])
#         # Penalise up to 20 points for large remaining debt relative to savings
#         avg_savings = np.mean([r.monthly_savings for r in self.repayments]) or 1
#         debt_penalty = min(20, (avg_remain_amount / (avg_savings * 12)) * 10)
#         repayment_score = max(0, min(100, base_score - debt_penalty))
#
#         # Active loans = records where remain_month > 0
#         active_loans = sum(1 for r in self.repayments if r.remain_month > 0)
#
#         latest = self.repayments[-1]    # most recent record
#
#         return {
#             "monthly_savings":    latest.monthly_savings,
#             "remain_amount":      latest.remain_amount,
#             "remain_month":       latest.remain_month,
#             "late_payment_count": late_count,
#             "repayment_score":    round(repayment_score, 2),
#             "active_loan_count":  active_loans,
#             "total_outstanding":  sum(r.remain_amount for r in self.repayments),
#         }
#
#     def _history_features(self) -> Dict[str, float]:
#         if not self.history:
#             return {
#                 "previous_loan_count":    0,
#                 "previous_approval_rate": 0.5,   # neutral prior
#             }
#
#         approved = sum(1 for h in self.history if h.status == "approved")
#         total = len(self.history)
#
#         return {
#             "previous_loan_count":    total,
#             "previous_approval_rate": round(approved / total, 4),
#         }
#
#     def _collateral_features(self) -> Dict[str, int]:
#         return {
#             "ownership_exists":        int(bool(self.doc.ownership_image)),
#             "business_license_exists": int(
#                 bool(self.doc.business_license_photo_front)
#                 or bool(self.doc.business_license_photo_back)
#             ),
#         }
#
#     def _occupation_features(self) -> Dict[str, float]:
#         occ = self.doc.occupation.lower().strip()
#         risk = self.OCCUPATION_RISK.get(occ, self.OCCUPATION_RISK["other"])
#
#         # One-hot for most common occupations
#         categories = [
#             "government", "employee", "business",
#             "self-employed", "farmer", "freelancer",
#         ]
#         occ_encoded = {f"occ_{c.replace('-','_')}": int(occ == c) for c in categories}
#         occ_encoded["occupation_risk_score"] = risk
#         return occ_encoded
#
#     def _demographic_features(self) -> Dict[str, int]:
#         return {
#             "gender_male": int(self.user.gender.lower() == "male"),
#         }
#
#     def _stability_score(
#         self,
#         income_feats: Dict,
#         repayment_feats: Dict,
#         collateral_feats: Dict,
#         history_feats: Dict,
#     ) -> Dict[str, float]:
#         """
#         Composite customer stability score (0–100).
#
#         Weights:
#           repayment_score        → 35%
#           previous_approval_rate → 20%
#           collateral bonus       → 15%
#           income sufficiency     → 20%
#           guarantor bonus        → 10%
#         """
#         repayment_score   = repayment_feats["repayment_score"]
#         approval_rate     = history_feats["previous_approval_rate"] * 100
#         collateral_bonus  = (
#             collateral_feats["ownership_exists"] * 10
#             + collateral_feats["business_license_exists"] * 5
#         )
#         income            = min(100, (income_feats["income_average"] / 500_000) * 100)
#         # If loan_to_income_ratio > 5 → penalise
#         lti_penalty       = max(0, (income_feats["loan_to_income_ratio"] - 5) * 5)
#         guarantor_bonus   = min(10, len(self.guarantors) * 5)
#
#         stability = (
#             repayment_score   * 0.35
#             + approval_rate   * 0.20
#             + collateral_bonus
#             + income          * 0.20
#             - lti_penalty
#             + guarantor_bonus
#         )
#         stability = max(0, min(100, stability))
#
#         return {"customer_stability_score": round(stability, 2)}
#
#     # ── Public API ─────────────────────────────────────────────
#
#     def build(self) -> Dict[str, Any]:
#         """
#         Assemble the complete feature vector.
#         Returns a flat dict ready for model inference.
#         """
#         income_feats     = self._income_features()
#         loan_feats       = self._loan_features()
#         guarantor_feats  = self._guarantor_features()
#         repayment_feats  = self._repayment_features()
#         history_feats    = self._history_features()
#         collateral_feats = self._collateral_features()
#         occupation_feats = self._occupation_features()
#         demographic_feats = self._demographic_features()
#         stability_feats  = self._stability_score(
#             income_feats, repayment_feats, collateral_feats, history_feats
#         )
#
#         features = {
#             **income_feats,
#             **loan_feats,
#             **guarantor_feats,
#             **repayment_feats,
#             **history_feats,
#             **collateral_feats,
#             **occupation_feats,
#             **demographic_feats,
#             **stability_feats,
#         }
#
#         logger.debug(f"Built feature vector with {len(features)} features")
#         return features
#
#     def to_dataframe(self) -> pd.DataFrame:
#         """Return features as a single-row DataFrame."""
#         return pd.DataFrame([self.build()])
#
#
# # ─────────────────────────────────────────────────────────────
# # DEMO / UNIT TEST
# # ─────────────────────────────────────────────────────────────
#
# if __name__ == "__main__":
#     engine = LoanFeatureEngine(
#         user=UserProfile(1, "Ko Aung", "Male", "pending"),
#         document=DocumentInfo(
#             income_average=450_000,
#             occupation="Business",
#             ownership_image="path/to/house.jpg",
#             business_license_photo_front="path/to/biz.jpg",
#             business_license_photo_back=None,
#         ),
#         application=LoanApplication(
#             amount_request=2_000_000,
#             month=24,
#             category_id=2,
#         ),
#         guarantors=[GuarantorRecord(1), GuarantorRecord(2)],
#         repayment_history=[
#             RepaymentRecord(500_000, 100_000, 0, 300_000, 12),
#             RepaymentRecord(500_000, 120_000, 1, 200_000, 6),
#             RepaymentRecord(500_000, 130_000, 0, 0, 0),
#         ],
#         loan_history=[
#             HistoryRecord("approved", 1_000_000),
#             HistoryRecord("approved", 1_500_000),
#         ],
#     )
#
#     features = engine.build()
#     print("\n" + "=" * 60)
#     print("GENERATED FEATURE VECTOR")
#     print("=" * 60)
#     for k, v in features.items():
#         print(f"  {k:<35} {v}")
#     print(f"\nTotal features: {len(features)}")

"""
=============================================================
PART 2 (UPDATED): FEATURE ENGINEERING — Myanmar Microfinance
=============================================================
Generates ML features from the Myanmar LMS database.
Supports 3 loan categories:
  1 → Agriculture   (Paddy/Land, Orchard/Garden)
  2 → MSME          (Retail, Wholesale, Manufacturing, …)
  3 → Consumer      (Vehicle, Housing, Others)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA CONTAINERS
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
    occupation: str
    ownership_image: Optional[str]
    business_license_photo_front: Optional[str]
    business_license_photo_back: Optional[str]


@dataclass
class LoanApplication:
    amount_request: float
    month: int
    category_id: int             # 1=Agriculture, 2=MSME, 3=Consumer
    sub_category: str            # e.g. "paddy", "retail", "vehicle"


# ── Category-specific detail dataclasses ─────────────────────

@dataclass
class AgricultureDetails:
    """Extra fields collected for Category 1 — Agriculture loans."""
    has_form_7: int              # 1=Yes, 0=No  (Land Ownership Form 7)
    land_size_acre: float        # cultivated area in acres
    seasonal_income: float       # estimated seasonal cash-flow in MMK
    crop_type: str               # e.g. "paddy", "fruit", "vegetable"


@dataclass
class MSMEDetails:
    """Extra fields collected for Category 2 — MSME loans."""
    has_business_license: int    # 1=Yes, 0=No
    years_in_business: float     # years operating
    daily_cash_flow: float       # average daily sales in MMK
    business_type: str           # "retail", "wholesale", "manufacturing", …


@dataclass
class ConsumerDetails:
    """Extra fields collected for Category 3 — Consumer loans."""
    fixed_monthly_salary: float  # regular salary in MMK
    has_collateral: int          # 1=Yes (asset/guarantor), 0=No
    consumer_purpose: str        # "vehicle", "housing", "other"


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
    status: str
    amount_request: float


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING ENGINE
# ─────────────────────────────────────────────────────────────

class LoanFeatureEngine:
    """
    Converts raw DB records into a flat feature dict for XGBoost.
    Category-specific feature groups are selected dynamically.
    """

    OCCUPATION_RISK = {
        "government":    0.10,
        "military":      0.10,
        "employee":      0.20,
        "teacher":       0.20,
        "nurse":         0.20,
        "doctor":        0.15,
        "engineer":      0.20,
        "business":      0.35,
        "self-employed": 0.45,
        "farmer":        0.45,
        "freelancer":    0.55,
        "unemployed":    0.95,
        "other":         0.40,
    }

    # Maximum allowed loan-to-income ratios per category
    LTI_CAPS = {1: 10.0, 2: 8.0, 3: 6.0}

    def __init__(
        self,
        user: UserProfile,
        document: DocumentInfo,
        application: LoanApplication,
        guarantors: List[GuarantorRecord],
        repayment_history: List[RepaymentRecord],
        loan_history: List[HistoryRecord],
        # Exactly one of the three below should be provided
        agriculture_details: Optional[AgricultureDetails] = None,
        msme_details: Optional[MSMEDetails] = None,
        consumer_details: Optional[ConsumerDetails] = None,
    ):
        self.user = user
        self.doc = document
        self.app = application
        self.guarantors = guarantors
        self.repayments = repayment_history
        self.history = loan_history
        self.agri = agriculture_details
        self.msme = msme_details
        self.consumer = consumer_details

        if self.app.category_id not in (1, 2, 3):
            raise ValueError(f"category_id must be 1, 2, or 3 — got {self.app.category_id}")

    # ── Shared feature groups ─────────────────────────────────

    def _income_features(self) -> Dict[str, float]:
        income = max(self.doc.income_average, 1.0)
        lti = self.app.amount_request / income
        dti = (self.app.amount_request / max(self.app.month, 1)) / income
        return {
            "income_average":       income,
            "loan_to_income_ratio": round(lti, 4),
            "debt_to_income_ratio": round(dti, 4),
        }

    def _loan_features(self) -> Dict[str, float]:
        return {
            "requested_amount":   self.app.amount_request,
            "loan_tenure_months": self.app.month,
            "category_id":        self.app.category_id,
            # sub-category one-hot
            "sub_cat_paddy":      int(self.app.sub_category == "paddy"),
            "sub_cat_orchard":    int(self.app.sub_category == "orchard"),
            "sub_cat_retail":     int(self.app.sub_category == "retail"),
            "sub_cat_wholesale":  int(self.app.sub_category == "wholesale"),
            "sub_cat_manufacturing": int(self.app.sub_category == "manufacturing"),
            "sub_cat_vehicle":    int(self.app.sub_category == "vehicle"),
            "sub_cat_housing":    int(self.app.sub_category == "housing"),
        }

    def _guarantor_features(self) -> Dict[str, int]:
        return {"guarantor_count": len(self.guarantors)}

    def _repayment_features(self) -> Dict[str, float]:
        if not self.repayments:
            return {
                "monthly_savings":    0.0,
                "remain_amount":      0.0,
                "remain_month":       0,
                "late_payment_count": 0,
                "repayment_score":    50.0,
                "active_loan_count":  0,
                "total_outstanding":  0.0,
            }
        late_count = sum(1 for r in self.repayments if r.flag == 1)
        total = len(self.repayments)
        on_time_rate = (total - late_count) / total
        base_score = on_time_rate * 80
        avg_remain = np.mean([r.remain_amount for r in self.repayments])
        avg_savings = np.mean([r.monthly_savings for r in self.repayments]) or 1
        debt_penalty = min(20, (avg_remain / (avg_savings * 12)) * 10)
        repayment_score = max(0, min(100, base_score - debt_penalty))
        active = sum(1 for r in self.repayments if r.remain_month > 0)
        latest = self.repayments[-1]
        return {
            "monthly_savings":    latest.monthly_savings,
            "remain_amount":      latest.remain_amount,
            "remain_month":       latest.remain_month,
            "late_payment_count": late_count,
            "repayment_score":    round(repayment_score, 2),
            "active_loan_count":  active,
            "total_outstanding":  sum(r.remain_amount for r in self.repayments),
        }

    def _history_features(self) -> Dict[str, float]:
        if not self.history:
            return {"previous_loan_count": 0, "previous_approval_rate": 0.5}
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
        categories = ["government", "employee", "business", "self-employed", "farmer", "freelancer"]
        encoded = {f"occ_{c.replace('-','_')}": int(occ == c) for c in categories}
        encoded["occupation_risk_score"] = risk
        return encoded

    def _demographic_features(self) -> Dict[str, int]:
        return {"gender_male": int(self.user.gender.lower() == "male")}

    def _stability_score(self, income_f, repayment_f, collateral_f, history_f) -> Dict[str, float]:
        repayment_score  = repayment_f["repayment_score"]
        approval_rate    = history_f["previous_approval_rate"] * 100
        collateral_bonus = (
            collateral_f["ownership_exists"] * 10
            + collateral_f["business_license_exists"] * 5
        )
        income_score = min(100, (income_f["income_average"] / 500_000) * 100)
        lti_penalty  = max(0, (income_f["loan_to_income_ratio"] - 5) * 5)
        guarantor_bonus = min(10, len(self.guarantors) * 5)
        stability = (
            repayment_score * 0.35
            + approval_rate * 0.20
            + collateral_bonus
            + income_score  * 0.20
            - lti_penalty
            + guarantor_bonus
        )
        return {"customer_stability_score": round(max(0, min(100, stability)), 2)}

    # ── Category-specific feature groups ─────────────────────

    def _agriculture_features(self) -> Dict[str, float]:
        """Features exclusive to Category 1 — Agriculture."""
        if not self.agri:
            # Zero-fill if called without details (cross-category inference)
            return {
                "has_form_7":            0,
                "land_size_acre":        0.0,
                "seasonal_income":       0.0,
                "seasonal_income_ratio": 0.0,
                "crop_paddy":            0,
                "crop_fruit":            0,
                "crop_vegetable":        0,
            }
        a = self.agri
        # Seasonal income relative to requested loan — measures repayment capacity
        seasonal_ratio = a.seasonal_income / max(self.app.amount_request, 1)
        return {
            "has_form_7":            a.has_form_7,
            "land_size_acre":        a.land_size_acre,
            "seasonal_income":       a.seasonal_income,
            "seasonal_income_ratio": round(seasonal_ratio, 4),
            "crop_paddy":            int(a.crop_type.lower() == "paddy"),
            "crop_fruit":            int(a.crop_type.lower() in ("fruit", "orchard")),
            "crop_vegetable":        int(a.crop_type.lower() == "vegetable"),
        }

    def _msme_features(self) -> Dict[str, float]:
        """Features exclusive to Category 2 — MSME."""
        if not self.msme:
            return {
                "has_business_license":    0,
                "years_in_business":       0.0,
                "daily_cash_flow":         0.0,
                "monthly_cash_flow":       0.0,
                "cashflow_to_loan_ratio":  0.0,
                "biz_retail":              0,
                "biz_wholesale":           0,
                "biz_manufacturing":       0,
            }
        m = self.msme
        monthly_cf = m.daily_cash_flow * 30
        cf_ratio   = monthly_cf / max(self.app.amount_request / max(self.app.month, 1), 1)
        btype = m.business_type.lower()
        return {
            "has_business_license":   m.has_business_license,
            "years_in_business":      m.years_in_business,
            "daily_cash_flow":        m.daily_cash_flow,
            "monthly_cash_flow":      monthly_cf,
            "cashflow_to_loan_ratio": round(cf_ratio, 4),
            "biz_retail":             int(btype == "retail"),
            "biz_wholesale":          int(btype == "wholesale"),
            "biz_manufacturing":      int(btype == "manufacturing"),
        }

    def _consumer_features(self) -> Dict[str, float]:
        """Features exclusive to Category 3 — Consumer."""
        if not self.consumer:
            return {
                "fixed_monthly_salary":   0.0,
                "has_collateral":         0,
                "salary_to_installment":  0.0,
                "purpose_vehicle":        0,
                "purpose_housing":        0,
                "purpose_other":          0,
            }
        c = self.consumer
        estimated_installment = self.app.amount_request / max(self.app.month, 1)
        salary_ratio = c.fixed_monthly_salary / max(estimated_installment, 1)
        purpose = c.consumer_purpose.lower()
        return {
            "fixed_monthly_salary":  c.fixed_monthly_salary,
            "has_collateral":        c.has_collateral,
            "salary_to_installment": round(salary_ratio, 4),
            "purpose_vehicle":       int(purpose == "vehicle"),
            "purpose_housing":       int(purpose == "housing"),
            "purpose_other":         int(purpose == "other"),
        }

    # ── Public API ─────────────────────────────────────────────

    def build(self) -> Dict[str, Any]:
        """
        Assemble the complete feature vector.
        Category-specific blocks are included for all categories
        (zero-filled for inactive categories) so the model always
        receives the same fixed-width input.
        """
        income_f      = self._income_features()
        loan_f        = self._loan_features()
        guarantor_f   = self._guarantor_features()
        repayment_f   = self._repayment_features()
        history_f     = self._history_features()
        collateral_f  = self._collateral_features()
        occupation_f  = self._occupation_features()
        demographic_f = self._demographic_features()
        stability_f   = self._stability_score(income_f, repayment_f, collateral_f, history_f)

        # Always include all three category blocks (zeros for inactive)
        agri_f     = self._agriculture_features()
        msme_f     = self._msme_features()
        consumer_f = self._consumer_features()

        features = {
            **income_f,
            **loan_f,
            **guarantor_f,
            **repayment_f,
            **history_f,
            **collateral_f,
            **occupation_f,
            **demographic_f,
            **stability_f,
            **agri_f,
            **msme_f,
            **consumer_f,
        }

        logger.debug(f"Built feature vector with {len(features)} features")
        return features

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.build()])


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = LoanFeatureEngine(
        user=UserProfile(1, "Ko Aung", "Male", "pending"),
        document=DocumentInfo(
            income_average=450_000,
            occupation="Farmer",
            ownership_image="path/to/form7.jpg",
            business_license_photo_front=None,
            business_license_photo_back=None,
        ),
        application=LoanApplication(
            amount_request=3_000_000,
            month=12,
            category_id=1,
            sub_category="paddy",
        ),
        guarantors=[GuarantorRecord(1)],
        repayment_history=[
            RepaymentRecord(500_000, 100_000, 0, 200_000, 6),
        ],
        loan_history=[HistoryRecord("approved", 1_000_000)],
        agriculture_details=AgricultureDetails(
            has_form_7=1,
            land_size_acre=5.0,
            seasonal_income=2_000_000,
            crop_type="paddy",
        ),
    )
    features = engine.build()
    print("\n" + "=" * 60)
    print("GENERATED FEATURE VECTOR (Agriculture)")
    print("=" * 60)
    for k, v in features.items():
        print(f"  {k:<40} {v}")
    print(f"\nTotal features: {len(features)}")
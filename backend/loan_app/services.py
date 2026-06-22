"""
=============================================================
PART 10: LOAN PREDICTION SERVICE
=============================================================
Orchestrates the entire prediction pipeline:
  1. Load user data from DB
  2. Build feature vector
  3. Run model
  4. Generate risk score
  5. Generate recommendation
  6. Persist result
  7. Return final response dict
"""

import logging
from decimal import Decimal
from pathlib import Path
from django.conf import settings

# ML engines (adjust path based on your repo layout)
import sys
sys.path.insert(0, str(Path(settings.BASE_DIR).parent / "ml" / "scripts"))

from feature_engineering import (          # type: ignore
    LoanFeatureEngine, UserProfile, DocumentInfo,
    LoanApplication, GuarantorRecord, RepaymentRecord as RepRecord,
    HistoryRecord,
)
from engines import (                    # type: ignore
    LoanPredictor, RiskScoringEngine,
    LoanAmountRecommender, InterestRateEngine,
    RejectionReasonEngine,
)

from .models import (
    User, Document, LoanHistory, Guarantor,
    RepaymentRecord, RiskScore,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Singleton model loader (loaded once per worker process)
# ─────────────────────────────────────────────────────────────

_predictor      = None
_risk_engine    = RiskScoringEngine()
_recommender    = LoanAmountRecommender()
_rate_engine    = InterestRateEngine()
_reason_engine  = RejectionReasonEngine()


def _get_predictor() -> LoanPredictor:
    global _predictor
    if _predictor is None:
        models_dir = Path(settings.BASE_DIR).parent / "ml" / "models"
        _predictor = LoanPredictor(models_dir=str(models_dir))
        logger.info("LoanPredictor loaded into memory.")
    return _predictor


# ─────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────

def _get_db_objects(user_id: int) -> tuple:
    """Fetch all DB objects needed for feature engineering."""
    user      = User.objects.get(pk=user_id)
    doc       = Document.objects.get(user_id=user_id)
    guarantors = list(Guarantor.objects.filter(user_id=user_id))
    past_loans = list(
        LoanHistory.objects.filter(user_id=user_id)
                           .exclude(status="pending")
                           .order_by("created_at")
    )
    # Repayment records from all past loans
    past_loan_ids = [l.id for l in past_loans]
    repayments    = list(
        RepaymentRecord.objects.filter(history_id__in=past_loan_ids)
                               .order_by("date")
    )
    return user, doc, guarantors, past_loans, repayments


# ─────────────────────────────────────────────────────────────
# Main service function
# ─────────────────────────────────────────────────────────────

def predict_loan(
    user_id: int,
    amount_request: float,
    month: int,
    category_id: int,
) -> dict:
    """
    Full prediction pipeline.
    Returns response dict matching the API response format (Part 11).
    """

    # ── 1. Load DB objects ────────────────────────────────────
    user, doc, guarantors, past_loans, repayments = _get_db_objects(user_id)
    logger.info(f"Loaded DB data for user {user_id}")

    # ── 2. Build feature vector ───────────────────────────────
    engine = LoanFeatureEngine(
        user=UserProfile(
            user_id=user.id,
            full_name=user.full_name,
            gender=user.gender,
            status=user.status,
        ),
        document=DocumentInfo(
            income_average=float(doc.income_average),
            occupation=doc.occupation,
            ownership_image=doc.ownership_image or None,
            business_license_photo_front=doc.business_license_photo_front or None,
            business_license_photo_back=doc.business_license_photo_back or None,
        ),
        application=LoanApplication(
            amount_request=float(amount_request),
            month=month,
            category_id=category_id,
        ),
        guarantors=[GuarantorRecord(g.id) for g in guarantors],
        repayment_history=[
            RepRecord(
                amount=float(r.amount),
                monthly_savings=float(r.monthly_savings),
                flag=r.flag,
                remain_amount=float(r.remain_amount),
                remain_month=r.remain_month,
            ) for r in repayments
        ],
        loan_history=[
            HistoryRecord(
                status=l.status,
                amount_request=float(l.amount_request),
            ) for l in past_loans
        ],
    )

    features = engine.build()
    logger.debug(f"Feature vector: {features}")

    # ── 3. Run model ──────────────────────────────────────────
    predictor  = _get_predictor()
    prediction = predictor.predict(features)

    approved      = prediction["approved"]
    approval_prob = prediction["approval_probability"]
    logger.info(f"Prediction: approved={approved}, prob={approval_prob:.3f}")

    # ── 4. Risk score ─────────────────────────────────────────
    risk       = _risk_engine.compute(approval_prob, features)
    risk_score = risk["score"]
    risk_level = risk["risk_level"]
    factors    = risk["factors"]

    # ── 5. Persist a LoanHistory row (pending) ────────────────
    loan_record = LoanHistory.objects.create(
        user_id=user_id,
        category_id=category_id,
        document=doc,
        status="pending",
        amount_request=Decimal(str(amount_request)),
        month=month,
    )

    # ── 6. Build response ─────────────────────────────────────
    if approved:
        rec_amount = _recommender.recommend(
            income_average          = features["income_average"],
            tenure_months           = features["loan_tenure_months"],
            risk_score              = risk_score,
            repayment_score         = features["repayment_score"],
            previous_approval_rate  = features["previous_approval_rate"],
            ownership_exists        = features["ownership_exists"],
            requested_amount        = features["requested_amount"],
            active_loan_count       = features["active_loan_count"],
            total_outstanding       = features["total_outstanding"],
        )
        interest_rate  = _rate_engine.get_rate(risk_level) * 100   # as %
        installment    = _rate_engine.monthly_installment(
            rec_amount, risk_level, month
        )

        response = {
            "approved":             True,
            "approval_probability": approval_prob,
            "risk_score":           risk_score,
            "risk_level":           risk_level,
            "recommended_amount":   rec_amount,
            "interest_rate":        interest_rate,
            "monthly_installment":  installment,
            "risk_factors":         factors,
        }

        # Update loan status
        loan_record.status = "approved"
        loan_record.save()

    else:
        reasons = _reason_engine.generate(features, approval_prob)
        response = {
            "approved":             False,
            "approval_probability": approval_prob,
            "risk_score":           risk_score,
            "risk_level":           risk_level,
            "reasons":              reasons,
        }

        loan_record.status = "rejected"
        loan_record.save()

    # ── 7. Save RiskScore audit record ────────────────────────
    RiskScore.objects.create(
        history=loan_record,
        risk_level=risk_level,
        score=risk_score,
        factors=factors,
        recommendation=response,
    )

    logger.info(f"Prediction saved. LoanHistory id={loan_record.id}")
    return response

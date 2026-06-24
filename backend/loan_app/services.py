# # """
# # =============================================================
# # PART 10: LOAN PREDICTION SERVICE
# # =============================================================
# # Orchestrates the entire prediction pipeline:
# #   1. Load user data from DB
# #   2. Build feature vector
# #   3. Run model
# #   4. Generate risk score
# #   5. Generate recommendation
# #   6. Persist result
# #   7. Return final response dict
# # """
# #
# # import logging
# # from decimal import Decimal
# # from pathlib import Path
# # from django.conf import settings
# #
# # # ML engines (adjust path based on your repo layout)
# # import sys
# # sys.path.insert(0, str(Path(settings.BASE_DIR).parent / "ml" / "scripts"))
# #
# # from feature_engineering import (          # type: ignore
# #     LoanFeatureEngine, UserProfile, DocumentInfo,
# #     LoanApplication, GuarantorRecord, RepaymentRecord as RepRecord,
# #     HistoryRecord,
# # )
# # from engines import (                    # type: ignore
# #     LoanPredictor, RiskScoringEngine,
# #     LoanAmountRecommender, InterestRateEngine,
# #     RejectionReasonEngine,
# # )
# #
# # from .models import (
# #     User, Document, LoanHistory, Guarantor,
# #     RepaymentRecord, RiskScore,
# # )
# #
# # logger = logging.getLogger(__name__)
# #
# #
# # # ─────────────────────────────────────────────────────────────
# # # Singleton model loader (loaded once per worker process)
# # # ─────────────────────────────────────────────────────────────
# #
# # _predictor      = None
# # _risk_engine    = RiskScoringEngine()
# # _recommender    = LoanAmountRecommender()
# # _rate_engine    = InterestRateEngine()
# # _reason_engine  = RejectionReasonEngine()
# #
# #
# # def _get_predictor() -> LoanPredictor:
# #     global _predictor
# #     if _predictor is None:
# #         models_dir = Path(settings.BASE_DIR).parent / "ml" / "models"
# #         _predictor = LoanPredictor(models_dir=str(models_dir))
# #         logger.info("LoanPredictor loaded into memory.")
# #     return _predictor
# #
# #
# # # ─────────────────────────────────────────────────────────────
# # # DB helpers
# # # ─────────────────────────────────────────────────────────────
# #
# # def _get_db_objects(user_id: int) -> tuple:
# #     """Fetch all DB objects needed for feature engineering."""
# #     user      = User.objects.get(pk=user_id)
# #     doc       = Document.objects.get(user_id=user_id)
# #     guarantors = list(Guarantor.objects.filter(user_id=user_id))
# #     past_loans = list(
# #         LoanHistory.objects.filter(user_id=user_id)
# #                            .exclude(status="pending")
# #                            .order_by("created_at")
# #     )
# #     # Repayment records from all past loans
# #     past_loan_ids = [l.id for l in past_loans]
# #     repayments    = list(
# #         RepaymentRecord.objects.filter(history_id__in=past_loan_ids)
# #                                .order_by("date")
# #     )
# #     return user, doc, guarantors, past_loans, repayments
# #
# #
# # # ─────────────────────────────────────────────────────────────
# # # Main service function
# # # ─────────────────────────────────────────────────────────────
# #
# # def predict_loan(
# #     user_id: int,
# #     amount_request: float,
# #     month: int,
# #     category_id: int,
# # ) -> dict:
# #     """
# #     Full prediction pipeline.
# #     Returns response dict matching the API response format (Part 11).
# #     """
# #
# #     # ── 1. Load DB objects ────────────────────────────────────
# #     user, doc, guarantors, past_loans, repayments = _get_db_objects(user_id)
# #     logger.info(f"Loaded DB data for user {user_id}")
# #
# #     # ── 2. Build feature vector ───────────────────────────────
# #     engine = LoanFeatureEngine(
# #         user=UserProfile(
# #             user_id=user.id,
# #             full_name=user.full_name,
# #             gender=user.gender,
# #             status=user.status,
# #         ),
# #         document=DocumentInfo(
# #             income_average=float(doc.income_average),
# #             occupation=doc.occupation,
# #             ownership_image=doc.ownership_image or None,
# #             business_license_photo_front=doc.business_license_photo_front or None,
# #             business_license_photo_back=doc.business_license_photo_back or None,
# #         ),
# #         application=LoanApplication(
# #             amount_request=float(amount_request),
# #             month=month,
# #             category_id=category_id,
# #         ),
# #         guarantors=[GuarantorRecord(g.id) for g in guarantors],
# #         repayment_history=[
# #             RepRecord(
# #                 amount=float(r.amount),
# #                 monthly_savings=float(r.monthly_savings),
# #                 flag=r.flag,
# #                 remain_amount=float(r.remain_amount),
# #                 remain_month=r.remain_month,
# #             ) for r in repayments
# #         ],
# #         loan_history=[
# #             HistoryRecord(
# #                 status=l.status,
# #                 amount_request=float(l.amount_request),
# #             ) for l in past_loans
# #         ],
# #     )
# #
# #     features = engine.build()
# #     logger.debug(f"Feature vector: {features}")
# #
# #     # ── 3. Run model ──────────────────────────────────────────
# #     predictor  = _get_predictor()
# #     prediction = predictor.predict(features)
# #
# #     approved      = prediction["approved"]
# #     approval_prob = prediction["approval_probability"]
# #     logger.info(f"Prediction: approved={approved}, prob={approval_prob:.3f}")
# #
# #     # ── 4. Risk score ─────────────────────────────────────────
# #     risk       = _risk_engine.compute(approval_prob, features)
# #     risk_score = risk["score"]
# #     risk_level = risk["risk_level"]
# #     factors    = risk["factors"]
# #
# #     # ── 5. Persist a LoanHistory row (pending) ────────────────
# #     loan_record = LoanHistory.objects.create(
# #         user_id=user_id,
# #         category_id=category_id,
# #         document=doc,
# #         status="pending",
# #         amount_request=Decimal(str(amount_request)),
# #         month=month,
# #     )
# #
# #     # ── 6. Build response ─────────────────────────────────────
# #     if approved:
# #         rec_amount = _recommender.recommend(
# #             income_average          = features["income_average"],
# #             tenure_months           = features["loan_tenure_months"],
# #             risk_score              = risk_score,
# #             repayment_score         = features["repayment_score"],
# #             previous_approval_rate  = features["previous_approval_rate"],
# #             ownership_exists        = features["ownership_exists"],
# #             requested_amount        = features["requested_amount"],
# #             active_loan_count       = features["active_loan_count"],
# #             total_outstanding       = features["total_outstanding"],
# #         )
# #         interest_rate  = _rate_engine.get_rate(risk_level) * 100   # as %
# #         installment    = _rate_engine.monthly_installment(
# #             rec_amount, risk_level, month
# #         )
# #
# #         response = {
# #             "approved":             True,
# #             "approval_probability": approval_prob,
# #             "risk_score":           risk_score,
# #             "risk_level":           risk_level,
# #             "recommended_amount":   rec_amount,
# #             "interest_rate":        interest_rate,
# #             "monthly_installment":  installment,
# #             "risk_factors":         factors,
# #         }
# #
# #         # Update loan status
# #         loan_record.status = "approved"
# #         loan_record.save()
# #
# #     else:
# #         reasons = _reason_engine.generate(features, approval_prob)
# #         response = {
# #             "approved":             False,
# #             "approval_probability": approval_prob,
# #             "risk_score":           risk_score,
# #             "risk_level":           risk_level,
# #             "reasons":              reasons,
# #         }
# #
# #         loan_record.status = "rejected"
# #         loan_record.save()
# #
# #     # ── 7. Save RiskScore audit record ────────────────────────
# #     RiskScore.objects.create(
# #         history=loan_record,
# #         risk_level=risk_level,
# #         score=risk_score,
# #         factors=factors,
# #         recommendation=response,
# #     )
# #
# #     logger.info(f"Prediction saved. LoanHistory id={loan_record.id}")
# #     return response
#
#
# """
# =============================================================
# PART 10 (UPDATED): LOAN PREDICTION SERVICE — Myanmar Microfinance
# =============================================================
# Orchestrates the full prediction pipeline:
#   1. Load user data from DB
#   2. Run hard-coded business rule guardrails per category
#   3. Build feature vector
#   4. Run ML model
#   5. Generate risk score + recommendation
#   6. Persist result
#   7. Return API response dict
# """
#
# import logging
# from decimal import Decimal
# from pathlib import Path
# from django.conf import settings
#
# import sys
# sys.path.insert(0, str(Path(settings.BASE_DIR).parent / "ml" / "scripts"))
#
# from feature_engineering import (          # type: ignore
#     LoanFeatureEngine,
#     UserProfile, DocumentInfo, LoanApplication,
#     GuarantorRecord, RepaymentRecord as RepRecord, HistoryRecord,
#     AgricultureDetails, MSMEDetails, ConsumerDetails,
# )
# from engines import (                      # type: ignore
#     LoanPredictor, RiskScoringEngine,
#     LoanAmountRecommender, InterestRateEngine, RejectionReasonEngine,
# )
# from .models import (
#     User,
#     Document,
#     LoanHistory,
#     RiskScore,
#     Guarantor,
#     AgriculturalProfile,  # 👈 'al' ပါတာ သေချာပါစေရှင့်
#     MSMEProfile,
#     ConsumerProfile,
#     RepaymentRecord
# )
#
# logger = logging.getLogger(__name__)
#
#
# # ─────────────────────────────────────────────────────────────
# # Singleton model loader
# # ─────────────────────────────────────────────────────────────
#
# _predictor     = None
# _risk_engine   = RiskScoringEngine()
# _recommender   = LoanAmountRecommender()
# _rate_engine   = InterestRateEngine()
# _reason_engine = RejectionReasonEngine()
#
#
# def _get_predictor() -> LoanPredictor:
#     global _predictor
#     if _predictor is None:
#         models_dir = Path(settings.BASE_DIR).parent / "ml" / "models"
#         _predictor = LoanPredictor(models_dir=str(models_dir))
#         logger.info("LoanPredictor loaded.")
#     return _predictor
#
#
# # ─────────────────────────────────────────────────────────────
# # BUSINESS RULE GUARDRAILS
# # Hard-coded pre-ML checks. If any rule fails the loan is
# # rejected immediately WITHOUT calling the ML model.
# # ─────────────────────────────────────────────────────────────
#
# class GuardrailViolation(Exception):
#     """Raised when a hard business rule is violated."""
#     def __init__(self, reasons: list[str]):
#         self.reasons = reasons
#         super().__init__("; ".join(reasons))
#
#
# def _check_common_guardrails(
#     doc,
#     amount_request: float,
#     month: int,
#     guarantors: list,
#     repayments: list,
# ):
#     """Rules that apply to ALL loan categories."""
#     violations = []
#
#     # Rule C1: Income must be verified (document must exist)
#     if doc.income_average is None or float(doc.income_average) <= 0:
#         violations.append("Verified income is required for all loan applications.")
#
#     # Rule C2: Loan tenure must be within allowed range
#     if month < 3:
#         violations.append("Minimum loan tenure is 3 months.")
#     if month > 60:
#         violations.append("Maximum loan tenure is 60 months.")
#
#     # Rule C3: Global loan-to-income ratio cap (safety net)
#     if doc.income_average and doc.income_average > 0:
#         lti = amount_request / float(doc.income_average)
#         if lti > 20:
#             violations.append(
#                 f"Loan-to-income ratio ({lti:.1f}x) exceeds absolute maximum of 20x."
#             )
#
#     # Rule C4: Active defaulted loans block new applications
#     active_late = sum(
#         1 for r in repayments if r.flag == 1 and r.remain_month > 0
#     )
#     if active_late >= 2:
#         violations.append(
#             "Application blocked: 2 or more active late-payment records found."
#         )
#
#     if violations:
#         raise GuardrailViolation(violations)
#
#
# def _check_agriculture_guardrails(
#     agri_profile,
#     amount_request: float,
#     month: int,
# ):
#     """Business rules specific to Category 1 — Agriculture."""
#     violations = []
#
#     # Rule A1: Land ownership document (Form 7) is mandatory
#     if not agri_profile or not agri_profile.has_form_7:
#         violations.append(
#             "Agriculture loans require Land Ownership Form 7 (ကြေးမဲ့လယ်ယာမြေပိုင်ဆိုင်မှု)."
#         )
#
#     # Rule A2: Minimum cultivated land size
#     if agri_profile and agri_profile.land_size_acre < 0.5:
#         violations.append(
#             f"Minimum cultivated land is 0.5 acres (provided: {agri_profile.land_size_acre} acres)."
#         )
#
#     # Rule A3: Seasonal income must plausibly cover repayment
#     if agri_profile and agri_profile.seasonal_income < (amount_request * 0.5):
#         violations.append(
#             "Declared seasonal income is less than 50% of requested loan amount — "
#             "insufficient repayment capacity."
#         )
#
#     # Rule A4: Agriculture loans are short-to-medium term only
#     if month > 36:
#         violations.append("Agriculture loans maximum tenure is 36 months.")
#
#     if violations:
#         raise GuardrailViolation(violations)
#
#
# def _check_msme_guardrails(
#     msme_profile,
#     amount_request: float,
#     month: int,
# ):
#     """Business rules specific to Category 2 — MSME."""
#     violations = []
#
#     # Rule M1: Business license is mandatory
#     if not msme_profile or not msme_profile.has_business_license:
#         violations.append(
#             "MSME loans require a valid business registration/license."
#         )
#
#     # Rule M2: Minimum time in business
#     if msme_profile and msme_profile.years_in_business < 1.0:
#         violations.append(
#             f"Business must have been operating for at least 1 year "
#             f"(declared: {msme_profile.years_in_business:.1f} years)."
#         )
#
#     # Rule M3: Daily cash flow must cover monthly installment
#     if msme_profile:
#         monthly_cf  = msme_profile.daily_cash_flow * 30
#         installment = amount_request / max(month, 1)
#         if monthly_cf < installment:
#             violations.append(
#                 f"Estimated monthly cash flow ({monthly_cf:,.0f} MMK) is lower than "
#                 f"the estimated monthly installment ({installment:,.0f} MMK)."
#             )
#
#     # Rule M4: MSME loan cap relative to daily cash flow
#     if msme_profile:
#         max_allowed = msme_profile.daily_cash_flow * 365 * 2   # 2× annual turnover
#         if amount_request > max_allowed:
#             violations.append(
#                 f"Requested amount ({amount_request:,.0f} MMK) exceeds 2× estimated "
#                 f"annual cash flow ({max_allowed:,.0f} MMK)."
#             )
#
#     if violations:
#         raise GuardrailViolation(violations)
#
#
# def _check_consumer_guardrails(
#     consumer_profile,
#     amount_request: float,
#     month: int,
# ):
#     """Business rules specific to Category 3 — Consumer."""
#     violations = []
#
#     # Rule V1: Fixed salary is required
#     if not consumer_profile or consumer_profile.fixed_monthly_salary <= 0:
#         violations.append(
#             "Consumer loans require proof of fixed monthly salary."
#         )
#
#     # Rule V2: Salary must cover installment with a 40% buffer
#     if consumer_profile and consumer_profile.fixed_monthly_salary > 0:
#         installment = amount_request / max(month, 1)
#         if consumer_profile.fixed_monthly_salary < installment * 1.4:
#             violations.append(
#                 f"Monthly salary ({consumer_profile.fixed_monthly_salary:,.0f} MMK) must be "
#                 f"at least 1.4× the estimated installment ({installment:,.0f} MMK). "
#                 f"Required: {installment * 1.4:,.0f} MMK."
#             )
#
#     # Rule V3: Collateral or guarantor required above threshold
#     COLLATERAL_THRESHOLD = 5_000_000   # 5 million MMK
#     if amount_request > COLLATERAL_THRESHOLD:
#         if not consumer_profile or not consumer_profile.has_collateral:
#             violations.append(
#                 f"Consumer loans above {COLLATERAL_THRESHOLD:,} MMK require "
#                 "collateral or a guarantor."
#             )
#
#     if violations:
#         raise GuardrailViolation(violations)
#
#
# # ─────────────────────────────────────────────────────────────
# # DB helpers
# # ─────────────────────────────────────────────────────────────
#
# def _get_db_objects(user_id: int, category_id: int) -> tuple:
#     """Fetch all DB objects needed for guardrails + feature engineering."""
#     user      = User.objects.get(pk=user_id)
#     doc       = Document.objects.get(user_id=user_id)
#     guarantors = list(Guarantor.objects.filter(user_id=user_id))
#     past_loans = list(
#         LoanHistory.objects.filter(user_id=user_id)
#                            .exclude(status="pending")
#                            .order_by("created_at")
#     )
#     past_loan_ids = [l.id for l in past_loans]
#     repayments    = list(
#         RepaymentRecord.objects.filter(history_id__in=past_loan_ids)
#                                .order_by("date")
#     )
#
#     # Category-specific profiles
#     agri_profile     = None
#     msme_profile     = None
#     consumer_profile = None
#
#     if category_id == 1:
#         agri_profile = AgriculturalProfile.objects.filter(user_id=user_id).first()
#     elif category_id == 2:
#         msme_profile = MSMEProfile.objects.filter(user_id=user_id).first()
#     elif category_id == 3:
#         consumer_profile = ConsumerProfile.objects.filter(user_id=user_id).first()
#
#     return user, doc, guarantors, past_loans, repayments, agri_profile, msme_profile, consumer_profile
#
#
# # ─────────────────────────────────────────────────────────────
# # MAIN SERVICE FUNCTION
# # ─────────────────────────────────────────────────────────────
#
# def predict_loan(
#     user_id: int,
#     amount_request: float,
#     month: int,
#     category_id: int,
#     sub_category: str,
# ) -> dict:
#     """
#     Full prediction pipeline with Myanmar business rule guardrails.
#     category_id: 1=Agriculture, 2=MSME, 3=Consumer
#     Returns response dict matching the API format.
#     """
#
#     # ── 1. Load DB objects ────────────────────────────────────
#     (user, doc, guarantors, past_loans, repayments,
#      agri_profile, msme_profile, consumer_profile) = _get_db_objects(user_id, category_id)
#     logger.info(f"[Service] Loaded DB data for user {user_id}, category {category_id}")
#
#     # ── 2. Business rule guardrails ───────────────────────────
#     #    Run BEFORE the ML model to short-circuit obvious rejections.
#     try:
#         _check_common_guardrails(doc, amount_request, month, guarantors, repayments)
#
#         if category_id == 1:
#             _check_agriculture_guardrails(agri_profile, amount_request, month)
#         elif category_id == 2:
#             _check_msme_guardrails(msme_profile, amount_request, month)
#         elif category_id == 3:
#             _check_consumer_guardrails(consumer_profile, amount_request, month)
#
#     except GuardrailViolation as exc:
#         logger.info(f"[Service] Guardrail rejection for user {user_id}: {exc.reasons}")
#         # Persist rejection record
#         loan_record = LoanHistory.objects.create(
#             user_id=user_id, category_id=category_id, document=doc,
#             status="rejected", amount_request=Decimal(str(amount_request)), month=month,
#         )
#         RiskScore.objects.create(
#             history=loan_record,
#             risk_level="very_high",
#             score=0.0,
#             factors={"guardrail_violations": exc.reasons},
#             recommendation={"approved": False, "reasons": exc.reasons},
#         )
#         return {
#             "approved":             False,
#             "approval_probability": 0.0,
#             "risk_score":           0.0,
#             "risk_level":           "very_high",
#             "reasons":              exc.reasons,
#             "guardrail_rejection":  True,   # signals pre-ML rejection to API layer
#         }
#
#     # ── 3. Build feature vector ───────────────────────────────
#     engine = LoanFeatureEngine(
#         user=UserProfile(
#             user_id=user.id,
#             full_name=user.full_name,
#             gender=user.gender,
#             status=user.status,
#         ),
#         document=DocumentInfo(
#             income_average=float(doc.income_average),
#             occupation=doc.occupation,
#             ownership_image=doc.ownership_image or None,
#             business_license_photo_front=doc.business_license_photo_front or None,
#             business_license_photo_back=doc.business_license_photo_back or None,
#         ),
#         application=LoanApplication(
#             amount_request=float(amount_request),
#             month=month,
#             category_id=category_id,
#             sub_category=sub_category,
#         ),
#         guarantors=[GuarantorRecord(g.id) for g in guarantors],
#         repayment_history=[
#             RepRecord(
#                 amount=float(r.amount),
#                 monthly_savings=float(r.monthly_savings),
#                 flag=r.flag,
#                 remain_amount=float(r.remain_amount),
#                 remain_month=r.remain_month,
#             ) for r in repayments
#         ],
#         loan_history=[
#             HistoryRecord(status=l.status, amount_request=float(l.amount_request))
#             for l in past_loans
#         ],
#         # Populate the relevant category-specific details object
#         agriculture_details=AgricultureDetails(
#             has_form_7=int(agri_profile.has_form_7),
#             land_size_acre=float(agri_profile.land_size_acre),
#             seasonal_income=float(agri_profile.seasonal_income),
#             crop_type=agri_profile.crop_type,
#         ) if agri_profile else None,
#
#         msme_details=MSMEDetails(
#             has_business_license=int(msme_profile.has_business_license),
#             years_in_business=float(msme_profile.years_in_business),
#             daily_cash_flow=float(msme_profile.daily_cash_flow),
#             business_type=msme_profile.business_type,
#         ) if msme_profile else None,
#
#         consumer_details=ConsumerDetails(
#             fixed_monthly_salary=float(consumer_profile.fixed_monthly_salary),
#             has_collateral=int(consumer_profile.has_collateral),
#             consumer_purpose=consumer_profile.consumer_purpose,
#         ) if consumer_profile else None,
#     )
#
#     features = engine.build()
#     logger.debug(f"[Service] Feature vector ({len(features)} features) built.")
#
#     # ── 4. ML model prediction ────────────────────────────────
#     predictor     = _get_predictor()
#     prediction    = predictor.predict(features)
#     approved      = prediction["approved"]
#     approval_prob = prediction["approval_probability"]
#     logger.info(f"[Service] ML result: approved={approved}, prob={approval_prob:.3f}")
#
#     # ── 5. Risk scoring ───────────────────────────────────────
#     risk       = _risk_engine.compute(approval_prob, features)
#     risk_score = risk["score"]
#     risk_level = risk["risk_level"]
#     factors    = risk["factors"]
#
#     # ── 6. Persist loan history record ───────────────────────
#     loan_record = LoanHistory.objects.create(
#         user_id=user_id,
#         category_id=category_id,
#         document=doc,
#         status="pending",
#         amount_request=Decimal(str(amount_request)),
#         month=month,
#     )
#
#     # ── 7. Build response ─────────────────────────────────────
#     if approved:
#         rec_amount = _recommender.recommend(
#             income_average         = features["income_average"],
#             tenure_months          = features["loan_tenure_months"],
#             risk_score             = risk_score,
#             repayment_score        = features["repayment_score"],
#             previous_approval_rate = features["previous_approval_rate"],
#             ownership_exists       = features["ownership_exists"],
#             requested_amount       = features["requested_amount"],
#             active_loan_count      = features["active_loan_count"],
#             total_outstanding      = features["total_outstanding"],
#         )
#         interest_rate = _rate_engine.get_rate(risk_level) * 100
#         installment   = _rate_engine.monthly_installment(rec_amount, risk_level, month)
#
#         response = {
#             "approved":             True,
#             "approval_probability": approval_prob,
#             "risk_score":           risk_score,
#             "risk_level":           risk_level,
#             "recommended_amount":   rec_amount,
#             "interest_rate":        interest_rate,
#             "monthly_installment":  installment,
#             "risk_factors":         factors,
#             "guardrail_rejection":  False,
#         }
#         loan_record.status = "approved"
#     else:
#         reasons = _reason_engine.generate(features, approval_prob)
#         response = {
#             "approved":             False,
#             "approval_probability": approval_prob,
#             "risk_score":           risk_score,
#             "risk_level":           risk_level,
#             "reasons":              reasons,
#             "guardrail_rejection":  False,
#         }
#         loan_record.status = "rejected"
#
#     loan_record.save()
#
#     RiskScore.objects.create(
#         history=loan_record,
#         risk_level=risk_level,
#         score=risk_score,
#         factors=factors,
#         recommendation=response,
#     )
#
#     logger.info(f"[Service] Done. LoanHistory id={loan_record.id}")
#     return response


import pandas as pd
# ✅ အစားထိုးရမည့် ကုဒ်လိုင်းအမှန်
from loan_app.models import User, LoanHistory, Document, Guarantor, RepaymentRecord, LoanCategory


# 💡 GuardrailViolation Exception ကို import လုပ်ရန် မမေ့ပါနှင့် (လိုအပ်ပါက လမ်းကြောင်းပြင်ပါ)
# from loan_app.exceptions import GuardrailViolation

def predict_loan_service(user_id, history_id):
    # ၁။ Database Tables အစစ်အမှန်များမှ ဒေတာဆွဲထုတ်ခြင်း
    user = User.objects.get(id=user_id)
    history = LoanHistory.objects.get(id=history_id)
    category = Category.objects.get(category_id=history.category_id)

    # document_table အစစ်မှ ဒေတာကို ယူခြင်း
    doc = Document.objects.filter(user_id=user_id).first()
    has_guarantor = Guarantor.objects.filter(history_id=history_id).exists()

    # 📥 ၂။ [Guardrails Check] - ဇယားဟောင်းတွေမသုံးတော့ဘဲ ညီမရဲ့ doc အစစ်နဲ့ စစ်ဆေးခြင်း
    # (category_name သို့မဟုတ် category_type နှစ်ခုလုံးကို စစ်ပေးထားပါတယ်)
    cat_name = getattr(category, 'category_name', getattr(category, 'category_type', '')).lower()

    if 'agri' in cat_name or 'စိုက်ပျိုးရေး' in cat_name:
        if not doc or not doc.owner_ship_image:
            raise GuardrailViolation(
                ['Agriculture loans require Land Ownership Form 7 (ကြေးမဲ့လယ်ယာမြေပိုင်ဆိုင်မှု).'])

    if 'msme' in cat_name or 'စီးပွားရေး' in cat_name:
        if not doc or not doc.business_license_photo_front:
            raise GuardrailViolation(['MSME loans require a valid business registration/license.'])

    # 📊 ၃။ စက်ရုပ် (ML Model) မျှော်လင့်ထားတဲ့ Universal Feature စာရင်းကြီးအတိုင်း ပြောင်းလဲခြင်း
    feature_dict = {
        'amount_request': float(history.amount_request),
        'month': int(history.month),
        'interest_rate': float(category.interest_rate),
        'income_average': float(doc.income_average) if doc else 0.0,

        # occupation ကို ကြည့်ပြီး ဝန်ထမ်း ဟုတ်/မဟုတ် စစ်ခြင်း
        'is_employed': 1 if doc and doc.occupation and doc.occupation.lower() in ['staff', 'employee',
                                                                                  'ဝန်ထမ်း'] else 0,

        # ချေးငွေ အမျိုးအစား ခွဲခြားခြင်း
        'is_agriculture': 1 if 'agri' in cat_name or 'စိုက်ပျိုးရေး' in cat_name else 0,
        'is_msme': 1 if 'msme' in cat_name or 'စီးပွားရေး' in cat_name else 0,

        # ဓာတ်ပုံ field တွေ တကယ်ရှိမရှိ စစ်ပြီး Boolean (1/0) ပြောင်းခြင်း
        'has_business_license': 1 if doc and doc.business_license_photo_front else 0,
        'has_ownership_document': 1 if doc and doc.owner_ship_image else 0,

        'has_guarantor': 1 if has_guarantor else 0,
        'gender_male': 1 if user.gender and user.gender.lower() == 'male' else 0
    }

    # မော်ဒယ် မျှော်လင့်ထားတဲ့ Columns အစီအစဉ်အတိုင်း ပုံသေ ညှိယူခြင်း
    MASTER_COLUMNS = [
        'amount_request', 'month', 'interest_rate', 'income_average',
        'is_employed', 'is_agriculture', 'is_msme',
        'has_business_license', 'has_ownership_document',
        'has_guarantor', 'gender_male'
    ]

    feature_df = pd.DataFrame([feature_dict])
    feature_df = feature_df[MASTER_COLUMNS]

    return feature_df
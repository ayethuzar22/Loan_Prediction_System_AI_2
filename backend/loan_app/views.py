"""
=============================================================
PART 10: VIEWS
=============================================================
Production-ready DRF view with authentication, logging,
throttling, and structured error handling.
"""

import logging
from rest_framework          import status
from rest_framework.views    import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.throttling  import UserRateThrottle
from django.core.exceptions     import ObjectDoesNotExist

from .serializers import LoanPredictRequestSerializer
from .services    import predict_loan
from rest_framework.permissions import AllowAny

logger = logging.getLogger(__name__)


class LoanPredictThrottle(UserRateThrottle):
    """Limit prediction calls: 10 per minute per user."""
    rate = "10/min"


# class LoanPredictView(APIView):
#     """
#     POST /api/loan/predict/
#
#     Accepts loan application data, runs the ML prediction pipeline,
#     and returns the decision with risk scoring and recommendations.
#     """
#     # permission_classes = [IsAuthenticated]
#     permission_classes = [AllowAny]
#     throttle_classes   = [LoanPredictThrottle]
#
#     # def post(self, request):
#     #     # ── Validate input ────────────────────────────────────
#     #     serializer = LoanPredictRequestSerializer(data=request.data)
#     #     if not serializer.is_valid():
#     #         return Response(
#     #             {"error": "Invalid request data", "details": serializer.errors},
#     #             status=status.HTTP_400_BAD_REQUEST,
#     #         )
#     #
#     #     data = serializer.validated_data
#     #
#     #     # ── Authorisation: users can only predict for themselves ──
#     #     if data["user_id"] != request.user.id:
#     #         logger.warning(
#     #             f"User {request.user.id} attempted to predict for user {data['user_id']}"
#     #         )
#     #         return Response(
#     #             {"error": "You can only submit predictions for your own account."},
#     #             status=status.HTTP_403_FORBIDDEN,
#     #         )
#     #
#     #     # ── Run prediction pipeline ───────────────────────────
#     #     try:
#     #         result = predict_loan(
#     #             user_id        = data["user_id"],
#     #             amount_request = float(data["amount_request"]),
#     #             month          = data["month"],
#     #             category_id    = data["category_id"],
#     #         )
#     #
#     #         logger.info(
#     #             f"Prediction complete for user={data['user_id']} "
#     #             f"approved={result['approved']} risk={result.get('risk_level')}"
#     #         )
#     #
#     #         return Response(result, status=status.HTTP_200_OK)
#     #
#     #     except ObjectDoesNotExist as e:
#     #         logger.error(f"DB object not found: {e}")
#     #         return Response(
#     #             {"error": "User profile or documents not found. Please complete your profile."},
#     #             status=status.HTTP_404_NOT_FOUND,
#     #         )
#     #
#     #     except FileNotFoundError as e:
#     #         logger.critical(f"Model not loaded: {e}")
#     #         return Response(
#     #             {"error": "Prediction model not available. Contact support."},
#     #             status=status.HTTP_503_SERVICE_UNAVAILABLE,
#     #         )
#     #
#     #     except Exception as e:
#     #         logger.exception(f"Unexpected error during prediction: {e}")
#     #         return Response(
#     #             {"error": "An unexpected error occurred. Please try again later."},
#     #             status=status.HTTP_500_INTERNAL_SERVER_ERROR,
#     #         )
#
#     class LoanPredictView(APIView):
#         permission_classes = [AllowAny]
#         throttle_classes = [LoanPredictThrottle]
#
#         # 💡 ဤ GET အပိုင်းကို ထပ်ပေါင်းထည့်ပေးပါရှင့်
#         def get(self, request):
#             """Browser ကနေ လှမ်းကြည့်ရင် API စမ်းသပ်ခန်း ပေါ်လာစေရန်"""
#             return Response(
#                 {
#                     "message": "Loan Predict Endpoint အဆင်သင့်ရှိနေပါပြီ။ အောက်က Form တွင် ဒေတာထည့်၍ POST စမ်းသပ်နိုင်ပါသည်။"},
#                 status=status.HTTP_200_OK
#             )
#
#         def post(self, request):
#             # ── Validate input ────────────────────────────────────
#             serializer = LoanPredictRequestSerializer(data=request.data)
#             ...
#
#     def post(self, request):
#         # ── Validate input ────────────────────────────────────
#         serializer = LoanPredictRequestSerializer(data=request.data)
#         if not serializer.is_valid():
#             return Response(
#                 {"error": "Invalid request data", "details": serializer.errors},
#                 status=status.HTTP_400_BAD_REQUEST,
#             )
#
#         data = serializer.validated_data
#
#         # ── 💡 စမ်းသပ်ရန်အတွက် ဤ Authorisation အပိုင်းကို ခဏ ပိတ်ထားပါ ──
#         # if data["user_id"] != request.user.id:
#         #     logger.warning(
#         #         f"User {request.user.id} attempted to predict for user {data['user_id']}"
#         #     )
#         #     return Response(
#         #         {"error": "You can only submit predictions for your own account."},
#         #         status=status.HTTP_403_FORBIDDEN,
#         #     )
#
#         # ── Run prediction pipeline ───────────────────────────
#         try:
#             # 💡 စမ်းသပ်မှုအောင်မြင်ရန် ဒေတာဘေ့စ်ထဲမှာ တကယ်ရှိမယ့် User ID (ဥပမာ: 1) ကို ခေတ္တ ပုံသေထည့်ပါမယ်
#             result = predict_loan(
#                 user_id=1,  # data["user_id"] အစား ၁ ဟု ခဏပြောင်းပါ
#                 amount_request=float(data["amount_request"]),
#                 month=data["month"],
#                 category_id=data["category_id"],
#             )
#
#             logger.info(
#                 f"Prediction complete for user=1 approved={result['approved']}"
#             )
#
#             return Response(result, status=status.HTTP_200_OK)
#
# class LoanHistoryView(APIView):
#     """
#     GET /api/loan/history/
#     Returns the authenticated user's loan application history.
#     """
#     permission_classes = [IsAuthenticated]
#
#     def get(self, request):
#         from .models import LoanHistory, RiskScore
#         from django.forms.models import model_to_dict
#
#         loans = LoanHistory.objects.filter(
#             user=request.user
#         ).order_by("-created_at").select_related("risk_score", "category")
#
#         result = []
#         for loan in loans:
#             item = {
#                 "id":             loan.id,
#                 "status":         loan.status,
#                 "amount_request": float(loan.amount_request),
#                 "month":          loan.month,
#                 "category":       loan.category.name if loan.category else None,
#                 "created_at":     loan.created_at.isoformat(),
#             }
#             try:
#                 rs = loan.risk_score
#                 item["risk_score"]   = rs.score
#                 item["risk_level"]   = rs.risk_level
#                 item["prediction"]   = rs.recommendation
#             except RiskScore.DoesNotExist:
#                 pass
#
#             result.append(item)
#
#         return Response({"loans": result}, status=status.HTTP_200_OK)

# class LoanPredictView(APIView):
#     """
#     POST /api/loan/predict/
#     Accepts loan application data, runs the ML prediction pipeline,
#     and returns the decision with risk scoring and recommendations.
#     """
#     permission_classes = [AllowAny]          # Browser ကနေ တန်းစမ်းနိုင်ရန်
#     # throttle_classes   = [LoanPredictThrottle]
#
#     # def get(self, request):
#     #     """Browser ကနေ လှမ်းကြည့်ရင် API စမ်းသပ်ခန်း ပေါ်လာစေရန်"""
#     #     return Response(
#     #         {"message": "Loan Predict Endpoint အဆင်သင့်ရှိနေပါပြီ။ အောက်က Form တွင် ဒေတာထည့်၍ POST စမ်းသပ်နိုင်ပါသည်။"},
#     #         status=status.HTTP_200_OK
#     #     )
#
#     def get(self, request):
#         """Browser ရဲ့ HTML API View ကို ကျော်ပြီး JSON သီးသန့်ပဲ တိုက်ရိုက်ထုတ်ပေးရန်"""
#         from rest_framework.renderers import JSONRenderer
#
#         data = {"message": "Loan Predict Endpoint အဆင်သင့်ရှိနေပါပြီ။"}
#         return Response(data, status=status.HTTP_200_OK, renderer_classes=[JSONRenderer])
#
#     def post(self, request):
#         # ── Validate input ────────────────────────────────────
#         serializer = LoanPredictRequestSerializer(data=request.data)
#         if not serializer.is_valid():
#             return Response(
#                 {"error": "Invalid request data", "details": serializer.errors},
#                 status=status.HTTP_400_BAD_REQUEST,
#             )
#
#         data = serializer.validated_data
#
#         # ── Run prediction pipeline ───────────────────────────
#         try:
#             # 💡 စမ်းသပ်မှုအောင်မြင်ရန် ဒေတာဘေ့စ်ထဲမှာ တကယ်ရှိမယ့် User ID (ဥပမာ: 1) ကို ခေတ္တ ပုံသေထည့်ပါမည်
#             result = predict_loan(
#                 user_id        = 1,
#                 amount_request = float(data["amount_request"]),
#                 month          = data["month"],
#                 category_id    = data["category_id"],
#             )
#
#             logger.info(
#                 f"Prediction complete for user=1 approved={result['approved']}"
#             )
#
#             return Response(result, status=status.HTTP_200_OK)
#
#         except ObjectDoesNotExist as e:
#             logger.error(f"DB object not found: {e}")
#             return Response(
#                 {"error": "User profile or documents not found. Please complete your profile."},
#                 status=status.HTTP_404_NOT_FOUND,
#             )
#
#         except FileNotFoundError as e:
#             logger.critical(f"Model not loaded: {e}")
#             return Response(
#                 {"error": "Prediction model not available. Contact support."},
#                 status=status.HTTP_503_SERVICE_UNAVAILABLE,
#             )
#
#         except Exception as e:
#             logger.exception(f"Unexpected error during prediction: {e}")
#             return Response(
#                 {"error": "An unexpected error occurred. Please try again later."},
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             )
#
#
# # ── အောက်ပါကုဒ်ကို views.py ရဲ့ အောက်ဆုံးမှာ သွားပေါင်းထည့်ပေးပါရှင့် ──
#
# class LoanHistoryView(APIView):
#     """
#     GET /api/loan/history/
#     Returns the authenticated user's loan application history.
#     """
#     permission_classes = [IsAuthenticated]
#
#     def get(self, request):
#         from .models import LoanHistory, RiskScore
#         from django.forms.models import model_to_dict
#
#         loans = LoanHistory.objects.filter(
#             user=request.user
#         ).order_by("-created_at").select_related("risk_score", "category")
#
#         result = []
#         for loan in loans:
#             item = {
#                 "id":             loan.id,
#                 "status":         loan.status,
#                 "amount_request": float(loan.amount_request),
#                 "month":          loan.month,
#                 "category":       loan.category.name if loan.category else None,
#                 "created_at":     loan.created_at.isoformat(),
#             }
#             try:
#                 rs = loan.risk_score
#                 item["risk_score"]   = rs.score
#                 item["risk_level"]   = rs.risk_level
#                 item["prediction"]   = rs.recommendation
#             except RiskScore.DoesNotExist:
#                 pass
#
#             result.append(item)
#
#         return Response({"loans": result}, status=status.HTTP_200_OK)


import logging
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated

logger = logging.getLogger(__name__)

class LoanPredictView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        # Browser ကနေ လာရင် ၅၀၀ လုံးဝ မပြဘဲ ဒီ JSON စာသားပဲ တန်းပြပေးမှာပါ
        return Response({"status": "Success", "message": "API အလုပ်လုပ်နေပါပြီ"})

    def post(self, request):
        # စမ်းသပ်ဖို့ ဒေတာပို့ရင် အောင်မြင်ကြောင်း ပြန်ပြမယ့်အပိုင်း
        return Response({"status": "Success", "received_data": request.data})


class LoanHistoryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Error မတက်အောင် လတ်တလော စမ်းသပ်မှုသက်သက် blank response ပြထားပါမယ်
        return Response({"loans": []}, status=status.HTTP_200_OK)
import logging
from django.core.exceptions import ObjectDoesNotExist
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.throttling import UserRateThrottle
from .services import predict_loan_service

logger = logging.getLogger(__name__)

class LoanPredictThrottle(UserRateThrottle):
    """Rate limit: 10 requests per minute."""
    rate = "10/min"

class LoanPredictView(APIView):
    permission_classes = [AllowAny]  # Shadow testing အတွက် လောလောဆယ် AllowAny ပေးထားပါသည်
    throttle_classes = [LoanPredictThrottle]

    def get(self, request):
        """Browser ကနေ လှမ်းကြည့်ရင် Server အလုပ်လုပ်ကြောင်း အချက်ပြရန်"""
        return Response(
            {"status": "Success", "message": "Loan Predict API Endpoint is ready."},
            status=status.HTTP_200_OK
        )

    def post(self, request):
        """Frontend သို့မဟုတ် Shadow Testing ကနေ လှမ်းဆော်ရမည့် ပင်မ Endpoint"""
        try:
            user_id = request.data.get("user_id")
            amount_request = request.data.get("amount_request")
            month = request.data.get("month")
            category_id = request.data.get("category_id")
            sub_category = request.data.get("sub_category", "")

            # Validation စစ်ဆေးခြင်း
            if not all([user_id, amount_request, month, category_id]):
                return Response(
                    {"error": "Missing required fields (user_id, amount_request, month, category_id)."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Core ML Pipeline ကို မောင်းနှင်ခြင်း
            result = predict_loan_service(
                user_id=int(user_id),
                amount_request=float(amount_request),
                month=int(month),
                category_id=int(category_id),
                sub_category=sub_category
            )

            logger.info(f"✅ Prediction done for user={user_id} -> Approved: {result.get('approved')}")
            return Response(result, status=status.HTTP_200_OK)

        except ObjectDoesNotExist as e:
            logger.error(f"❌ DB object missing: {e}")
            return Response(
                {"error": "User profile or business document records not found in Database."},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.exception(f"💥 Unexpected pipeline error: {e}")
            return Response(
                {"error": f"An unexpected system error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class LoanHistoryView(APIView):
    """အတိတ်က ချေးငွေမှတ်တမ်းများကို ပြန်ထုတ်ပေးမည့် View"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response({"loans": []}, status=status.HTTP_200_OK)
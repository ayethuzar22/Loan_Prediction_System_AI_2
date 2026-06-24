from django.urls import path
from .views import LoanPredictView, LoanHistoryView

urlpatterns = [
    # Class-Based View ကို သုံးပြီး URL လမ်းကြောင်း ချိတ်ဆက်ခြင်း
    path('predict-loan/', LoanPredictView.as_view(), name='predict_loan_api'),
    path('loan-history/', LoanHistoryView.as_view(), name='loan_history_api'),
]
"""
=============================================================
PART 10: URL CONFIGURATION
=============================================================
"""

from django.urls import path
from .views import LoanPredictView, LoanHistoryView

urlpatterns = [
    path("loan/predict/", LoanPredictView.as_view(),  name="loan-predict"),
    path("loan/history/", LoanHistoryView.as_view(), name="loan-history"),
]

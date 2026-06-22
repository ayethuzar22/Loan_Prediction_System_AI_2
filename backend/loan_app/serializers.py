"""
=============================================================
PART 10: SERIALIZERS
=============================================================
"""

from rest_framework import serializers
from .models import User, Document, LoanHistory, Guarantor, RepaymentRecord


class GuarantorSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Guarantor
        fields = ["guarantor_name", "guarantor_address", "guarantor_phone",
                  "nrc_front", "nrc_back"]


class RepaymentRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model  = RepaymentRecord
        fields = ["amount", "monthly_savings", "date", "flag",
                  "remain_amount", "remain_month"]


class LoanPredictRequestSerializer(serializers.Serializer):
    """
    Input for POST /api/loan/predict/
    The frontend passes user_id + the new application details.
    All historical data is fetched from the DB inside the service.
    """
    user_id        = serializers.IntegerField()
    amount_request = serializers.DecimalField(max_digits=15, decimal_places=2)
    month          = serializers.IntegerField(min_value=1, max_value=60)
    category_id    = serializers.IntegerField()


class LoanPredictApprovedResponseSerializer(serializers.Serializer):
    """Schema for approved loan response."""
    approved             = serializers.BooleanField()
    approval_probability = serializers.FloatField()
    risk_score           = serializers.IntegerField()
    risk_level           = serializers.CharField()
    recommended_amount   = serializers.FloatField()
    interest_rate        = serializers.FloatField()
    monthly_installment  = serializers.FloatField()
    risk_factors         = serializers.ListField(child=serializers.CharField())


class LoanPredictRejectedResponseSerializer(serializers.Serializer):
    """Schema for rejected loan response."""
    approved             = serializers.BooleanField()
    approval_probability = serializers.FloatField()
    risk_score           = serializers.IntegerField()
    risk_level           = serializers.CharField()
    reasons              = serializers.ListField(child=serializers.CharField())

"""
=============================================================
PART 10: DJANGO MODELS
=============================================================
Mirrors the existing database schema with Django ORM models,
plus the RiskScore audit model.
"""

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager

# class User(models.Model):
#     STATUS_CHOICES = [
#         ("pending",  "Pending"),
#         ("approved", "Approved"),
#         ("rejected", "Rejected"),
#     ]
#     full_name    = models.CharField(max_length=255)
#     nrc_number   = models.CharField(max_length=100, unique=True)
#     phone_number = models.CharField(max_length=20)
#     address      = models.TextField(blank=True)
#     email        = models.EmailField(unique=True)
#     gender       = models.CharField(max_length=10)
#     photo_url    = models.URLField(blank=True)
#     status       = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
#     created_at   = models.DateTimeField(auto_now_add=True)
#
#     class Meta:
#         db_table = "users_table"
#
#     def __str__(self):
#         return f"{self.full_name} ({self.nrc_number})"

# 🌟 User တွေကို Create လုပ်ပေးမယ့် Manager Class တစ်ခု လိုအပ်ပါတယ်
class UserManager(BaseUserManager):
    def create_user(self, email, full_name, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, full_name=full_name, **extra_fields)
        if password:
            user.set_password(password)  # Password ကို hash လုပ်ပြီး သိမ်းပေးခြင်း
        user.save(using=self._db)
        return user

    def create_superuser(self, email, full_name, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, full_name, password, **extra_fields)


# 🌟 မူရင်း User Model ကို အောက်ပါအတိုင်း ပြင်ပါ
class User(AbstractBaseUser, PermissionsMixin):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("approved", "Approved"),
        ("rejected", "Rejected"),
    ]
    full_name = models.CharField(max_length=255)
    nrc_number = models.CharField(max_length=100, unique=True)
    phone_number = models.CharField(max_length=20)
    address = models.TextField(blank=True)
    email = models.EmailField(unique=True)
    gender = models.CharField(max_length=10)
    photo_url = models.URLField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")

    # Django Auth System အတွက် လိုအပ်သော အကွက်များ
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = UserManager()

    USERNAME_FIELD = 'email'  # Email နဲ့ Login ဝင်ခိုင်းမည်
    REQUIRED_FIELDS = ['full_name']  # Admin ဆောက်ရင် မဖြစ်မနေ တောင်းမည့် ကွက်လပ်

    class Meta:
        db_table = "users_table"

    def __str__(self):
        return f"{self.full_name} ({self.nrc_number})"

class LoanCategory(models.Model):
    name        = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    interest_rate = models.DecimalField(max_digits=5, decimal_places=2, default=0.00)  # 🌟 ဖြည့်စွက်ရန်

    class Meta:
        db_table = "category_table"

    def __str__(self):
        return self.name


class Document(models.Model):
    user         = models.OneToOneField(User, on_delete=models.CASCADE, related_name="document")
    phone_number = models.CharField(max_length=20)
    occupation   = models.CharField(max_length=100)

    nrc_photo_front            = models.URLField(blank=True)
    nrc_photo_back             = models.URLField(blank=True)
    household_list_photo_front = models.URLField(blank=True)
    household_list_photo_back  = models.URLField(blank=True)
    ownership_image            = models.URLField(blank=True)
    business_license_photo_front = models.URLField(blank=True)
    business_license_photo_back  = models.URLField(blank=True)
    face_scan_photo            = models.URLField(blank=True)

    income_average = models.DecimalField(max_digits=15, decimal_places=2)

    class Meta:
        db_table = "document_table"

    def __str__(self):
        return f"Document for {self.user.full_name}"


class LoanHistory(models.Model):
    STATUS_CHOICES = [
        ("pending",  "Pending"),
        ("approved", "Approved"),
        ("rejected", "Rejected"),
    ]
    user           = models.ForeignKey(User, on_delete=models.CASCADE, related_name="history")
    category       = models.ForeignKey(LoanCategory, on_delete=models.SET_NULL, null=True)
    document       = models.ForeignKey(Document, on_delete=models.SET_NULL, null=True)
    status         = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    amount_request = models.DecimalField(max_digits=15, decimal_places=2)
    month          = models.PositiveIntegerField()
    insurance_cost = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    created_at     = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "history_table"

    def __str__(self):
        return f"Loan #{self.id} by {self.user.full_name} ({self.status})"


class Guarantor(models.Model):
    user             = models.ForeignKey(User, on_delete=models.CASCADE, related_name="guarantors")
    document         = models.ForeignKey(Document, on_delete=models.SET_NULL, null=True)
    guarantor_name   = models.CharField(max_length=255)
    guarantor_address = models.TextField()
    guarantor_phone  = models.CharField(max_length=20)
    nrc_front        = models.URLField(blank=True)
    nrc_back         = models.URLField(blank=True)

    class Meta:
        db_table = "guarantor_table"

    def __str__(self):
        return f"Guarantor: {self.guarantor_name} for user {self.user_id}"


class RepaymentRecord(models.Model):
    FLAG_CHOICES = [(0, "On Time"), (1, "Late")]
    history        = models.ForeignKey(LoanHistory, on_delete=models.CASCADE, related_name="repayments")
    amount         = models.DecimalField(max_digits=15, decimal_places=2)
    monthly_savings = models.DecimalField(max_digits=15, decimal_places=2)
    date           = models.DateField()
    flag           = models.IntegerField(choices=FLAG_CHOICES, default=0)
    remain_amount  = models.DecimalField(max_digits=15, decimal_places=2)
    remain_month   = models.PositiveIntegerField()

    class Meta:
        db_table = "list_table"

    def __str__(self):
        return f"Repayment for loan #{self.history_id} on {self.date}"


class Notification(models.Model):
    user    = models.ForeignKey(User, on_delete=models.CASCADE, related_name="notifications")
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    sent_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "notification_table"


class RiskScore(models.Model):
    RISK_LEVEL_CHOICES = [
        ("LOW",    "Low"),
        ("MEDIUM", "Medium"),
        ("HIGH",   "High"),
    ]
    history        = models.OneToOneField(
        LoanHistory, on_delete=models.CASCADE, related_name="risk_score"
    )
    risk_level     = models.CharField(max_length=10, choices=RISK_LEVEL_CHOICES)
    score          = models.IntegerField()                     # 0–100
    factors        = models.JSONField(default=list)            # list of strings
    recommendation = models.JSONField(default=dict)            # full prediction payload
    generated_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "risk_score_table"

    def __str__(self):
        return f"Risk: {self.risk_level} ({self.score}) for loan #{self.history_id}"

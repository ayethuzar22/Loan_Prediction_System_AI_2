import os
import sys
import django

# 💡 Django ရဲ့ ပင်မ backend လမ်းကြောင်းတွေကို Python ထံ အသိပေးခြင်း
BASE_DIR = r"D:\Loan_System\backend"
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "backend"))

# Django Environment ကို ချိတ်ဆက်ခြင်း
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from django.db import connection
from loan_app.models import (
    User, Document, LoanHistory, RiskScore,
    RepaymentRecord, Guarantor, AgriculturalProfile,
    MSMEProfile, ConsumerProfile
)

# မော်ဒယ်လ် စာရင်း
models = [User, Document, LoanHistory, RiskScore, RepaymentRecord, Guarantor, AgriculturalProfile, MSMEProfile, ConsumerProfile]

print("🛠️ Checking and creating missing database tables...")

# မရှိသေးတဲ့ Tables တွေကို အတင်းလိုက်ဆောက်ခြင်း
existing = connection.introspection.table_names()
with connection.schema_editor() as ctx:
    for m in models:
        if m._meta.db_table not in existing:
            ctx.create_model(m)
            print(f"✅ Created table: {m._meta.db_table}")
        else:
            print(f"ℹ️ Table already exists: {m._meta.db_table}")

print("🚀 All tables are synchronized successfully!")
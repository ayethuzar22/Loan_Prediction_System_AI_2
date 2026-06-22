# test_shadow.py
import os
import sys
from pathlib import Path

print("🚀 Starting Shadow Testing Verification...")

# backend လမ်းကြောင်း တစ်ခုပဲ ပေးရန်
BACKEND_DIR = Path(__file__).parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "loan_project.settings")

import django

django.setup()

# Django setup ပြီးမှ service ကို ခေါ်ယူခြင်း
from loan_app.services import predict_loan


def run_local_shadow_test():
    TEST_USER_ID = 1
    try:
        print(f"Running predict_loan for User ID: {TEST_USER_ID}...")
        response = predict_loan(
            user_id=TEST_USER_ID,
            amount_request=2000000.0,
            month=12,
            category_id=1
        )
        print("\n✅ Service Execution: SUCCESS")
        print(f"Response Output: {response}")

    except Exception as e:
        print(f"\n❌ Service Execution: FAILED -> {str(e)}")


if __name__ == "__main__":
    run_local_shadow_test()
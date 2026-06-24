import os
import sys
from pathlib import Path

# ၁။ Project Root (D:\Loan_System) ရော backend Folder ပါ Python Path ထဲသို့ စနစ်တကျ အရင်ထည့်သွင်းခြင်း
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # D:\Loan_System
BACKEND_DIR = BASE_DIR / "backend"  # D:\Loan_System\backend

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

# ⚠️ ၎င်းနေရာတွင် Django setup မလုပ်ခင် မည်သည့် backend module ကိုမှ ကြိုတင် import မလုပ်ရပါ။

# ၂။ Django settings ဖိုင်ကို ချိတ်ဆက်ခြင်း
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "loan_project.settings")

print("🚀 Starting Shadow Testing Verification...")

# ၃။ Django Environment ကို စနစ်တကျ အသက်သွင်းခြင်း
import django

django.setup()

print("🚀 Django Environment Initialized Successfully!")

# ၄။ ✨ [SMART MAPPER] ဇယားနာမည် ကွဲလွဲမှုအားလုံးကို အလိုအလျောက် ပတ်ချ် (Patch) လုပ်ပေးခြင်း
from django.apps import apps

app_config = apps.get_app_config('loan_app')

# ညီမရဲ့ models.py ထဲက နာမည်အမှန်များနှင့် services.py က တောင်းမည့် နာမည်များ ညှိနှိုင်းခြင်း
models_to_patch = {
    'History_table': ['loanhistory', 'history', 'history_table'],
    'document_table': ['document', 'document_table', 'user_document'],
    'guarantor_table': ['guarantor', 'guarantor_table'],
    'list_table': ['repaymentrecord', 'list', 'list_table', 'repayment'],
    'Category': ['loancategory', 'category', 'category_table']
}

# Python module အတုတစ်ခု စနစ်ထဲ ကြိုဆောက်ထားခြင်း
sys.modules['loan_app.models'] = sys.modules.get('loan_app.models', type(sys)('loan_app.models'))

print("\n📋 Scanning and Mapping Database Models...")
History_model = None

for target_name, keywords in models_to_patch.items():
    found_model = None
    for model_name, model_cls in app_config.models.items():
        if any(kw in model_name.lower() for kw in keywords):
            found_model = model_cls
            break

    if found_model:
        setattr(sys.modules['loan_app.models'], target_name, found_model)
        if target_name == 'History_table':
            History_model = found_model
        print(f"   ✅ Auto-mapped: '{target_name}' -> Real Model: '{found_model.__name__}'")
    else:
        available_models = list(app_config.models.values())
        if available_models:
            setattr(sys.modules['loan_app.models'], target_name, available_models[0])

# ၅။ 🚀 Django Setup ပြီးမှသာ Backend က Services ဖိုင်ကို စိတ်ချလက်ချ Import လုပ်ခြင်း
try:
    import loan_app.services as loan_services
except ImportError as ie:
    print(f"\n❌ Import Error: Cannot find loan_app.services. -> {str(ie)}")
    sys.exit(1)


def run_local_shadow_test():
    print("\n🚀 Running predict_loan with Real Backend Tables...")

    # services.py ထဲမှာ ရှိနိုင်မယ့် function နာမည်ကို အလိုအလျောက် ရှာဖွေခြင်း
    predict_func = None
    for func_name in ['predict_loan_service', 'predict_loan', 'predict_loan_score']:
        if hasattr(loan_services, func_name):
            predict_func = getattr(loan_services, func_name)
            print(f"✨ Found real service function: '{func_name}'")
            break

    if not predict_func:
        print("❌ Error: Could not find any prediction function in services.py!")
        return

    # 💡 [DYNAMIC DATA PICKER] ဒေတာဘေ့စ်ထဲက ပထမဆုံးရှိတဲ့ Record ID တွေကို အလိုအလျောက် ဆွဲယူခြင်း
    try:
        from django.contrib.auth import get_user_model
        User_model = get_user_model()

        # Docker DB ထဲက ပထမဆုံး ရှိနေတဲ့ Record တွေကို ဆွဲထုတ်ခြင်း
        first_user = User_model.objects.first()
        first_history = History_model.objects.first() if History_model else None

        if not first_user or not first_history:
            print("\n❌ [DATABASE EMPTY ERROR] Your Docker Database tables are empty!")
            print(
                "💡 Please login to your Admin Website, register a user, and create at least one loan history entry first.")
            return

        TEST_USER_ID = first_user.id
        TEST_HISTORY_ID = first_history.id
        print(f"ℹ️ Testing with Active Real Data -> User ID: {TEST_USER_ID}, History ID: {TEST_HISTORY_ID}\n")

    except Exception as db_err:
        print(f"⚠️ Could not fetch dynamic IDs: {db_err}. Falling back to default ID=1.")
        TEST_USER_ID = 1
        TEST_HISTORY_ID = 1

    try:
        # 💡 ရှာတွေ့တဲ့ function ကို သုံးပြီး မာစတာ Feature Matrix ကို ထုတ်ယူခြင်း
        feature_df = predict_func(user_id=TEST_USER_ID, history_id=TEST_HISTORY_ID)

        print("\n🔥 [SUCCESS] Generated Universal Feature DataFrame:")
        print("--------------------------------------------------")
        print(feature_df)
        print("--------------------------------------------------")
        print(f"Shape of DataFrame: {feature_df.shape} (Rows, Columns)")

    except Exception as e:
        print(f"\n❌ Service Execution FAILED -> {str(e)}")
        print(
            "💡 Hint: If it shows query does not exist, please ensure your web application has generated active records in the Docker DB.")


if __name__ == "__main__":
    run_local_shadow_test()
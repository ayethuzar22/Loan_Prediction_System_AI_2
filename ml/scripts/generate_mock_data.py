import pandas as pd
import random
import os

# ညီမတို့ feature_engineering.py ထဲက Classes တွေကို လှမ်းခေါ်ခြင်း
from feature_engineering import (
    UserProfile, DocumentInfo, LoanApplication,
    AgricultureDetails, MSMEDetails, ConsumerDetails,
    GuarantorRecord, RepaymentRecord, HistoryRecord,
    LoanFeatureEngine
)


def generate_myanmar_mock_dataset(num_records=1000, output_path="ml/data/myanmar_loan_mock_data.csv"):
    print(f"🔄 Generating {num_records} Myanmar localized mock loan profiles...")

    occupations = ["farmer", "business", "employee", "government", "self-employed", "freelancer"]
    genders = ["Male", "Female"]

    dataset_features = []
    labels = []  # 1 = Approved, 0 = Rejected

    for i in range(num_records):
        user_id = i + 1
        category_id = random.choice([1, 2, 3])  # 1=Agri, 2=MSME, 3=Consumer
        gender = random.choice(genders)

        # --- Default/Shared Data setup ---
        amount_request = random.choice([1_000_000, 2_000_000, 3_000_000, 5_000_000, 10_000_000])
        month = random.choice([6, 12, 18, 24])
        income_average = random.randint(300_000, 1_500_000)

        # Repayment history mock
        late_flag = random.choice([0, 0, 0, 1])  # 25% chance of late payment history
        repayments = [RepaymentRecord(amount=200_000, monthly_savings=50_000, flag=late_flag, remain_amount=100_000,
                                      remain_month=2)]
        loan_history = [HistoryRecord(status="approved", amount_request=1_000_000)]
        guarantors = [GuarantorRecord(guarantor_id=random.randint(10, 100))]

        # --- Initialize category variables ---
        agri, msme, consumer = None, None, None
        is_approved = 1  # Default assumption
        ownership_image = None
        biz_front = None

        # --- Category Specific Generation & Ground-Truth Labels ---
        if category_id == 1:  # 🌾 Agriculture
            occupation = "farmer"
            sub_category = random.choice(["paddy", "orchard"])
            has_form_7 = random.choice([0, 1, 1])  # 66% have Form 7
            land_size = random.uniform(2.0, 15.0)
            seasonal_income = amount_request * random.uniform(0.5, 2.0)

            agri = AgricultureDetails(has_form_7=has_form_7, land_size_acre=land_size, seasonal_income=seasonal_income,
                                      crop_type=sub_category)
            ownership_image = "form7_img.jpg" if has_form_7 else None

            # Microfinance Rule: No Form 7 = High rejection rate
            if has_form_7 == 0:
                is_approved = 0 if random.random() < 0.9 else 1  # 90% rejection

        elif category_id == 2:  # 🏪 MSME
            occupation = random.choice(["business", "self-employed"])
            sub_category = random.choice(["retail", "wholesale", "manufacturing"])
            has_license = random.choice([0, 1, 1])
            years_in_biz = random.uniform(0.5, 10.0)
            daily_cf = random.randint(20_000, 200_000)

            msme = MSMEDetails(has_business_license=has_license, years_in_business=years_in_biz,
                               daily_cash_flow=daily_cf, business_type=sub_category)
            biz_front = "license_front.jpg" if has_license else None

            # Microfinance Rule: Less than 1 year business experience = High rejection
            if years_in_biz < 1.0 or has_license == 0:
                is_approved = 0 if random.random() < 0.8 else 1

        else:  # 🚗 Consumer
            occupation = random.choice(["employee", "government", "freelancer"])
            sub_category = random.choice(["vehicle", "housing", "other"])
            has_collateral = random.choice([0, 1])
            fixed_salary = income_average if occupation in ["employee", "government"] else 0

            consumer = ConsumerDetails(fixed_monthly_salary=fixed_salary, has_collateral=has_collateral,
                                       consumer_purpose=sub_category)

            # Rule: High request but no collateral or regular income = Reject
            if has_collateral == 0 and amount_request > 3_000_000:
                is_approved = 0 if random.random() < 0.85 else 1

        # Build Objects
        user = UserProfile(user_id=user_id, full_name=f"Customer {user_id}", gender=gender, status="pending")
        doc = DocumentInfo(income_average=income_average, occupation=occupation, ownership_image=ownership_image,
                           business_license_photo_front=biz_front, business_license_photo_back=None)
        app = LoanApplication(amount_request=amount_request, month=month, category_id=category_id,
                              sub_category=sub_category)

        # Run through ညီမရဲ့ LoanFeatureEngine to flatten features!
        engine = LoanFeatureEngine(
            user=user, document=doc, application=app, guarantors=guarantors,
            repayment_history=repayments, loan_history=loan_history,
            agriculture_details=agri, msme_details=msme, consumer_details=consumer
        )

        # Flattened feature vector dictionary ရယူခြင်း
        flat_features = engine.build()

        dataset_features.append(flat_features)
        labels.append(is_approved)

    # DataFrame အဖြစ်ပြောင်းပြီး CSV ထုတ်ယူခြင်း
    df = pd.DataFrame(dataset_features)
    df["loan_status_target"] = labels  # ML Model အတွက် Target column (Y-value)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Success! Mock dataset saved at: {output_path} with shape {df.shape}")


if __name__ == "__main__":
    generate_myanmar_mock_dataset(num_records=1200)
# ml/scripts/evaluate_shadow.py

import pandas as pd
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def analyze_shadow_results(log_file_path: str):
    records = []
    with open(log_file_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    df = pd.DataFrame(records)

    # တကယ့်လူရဲ့ ဆုံးဖြတ်ချက် (Ground Truth) နှင့် AI ၏ ခန့်မှန်းချက် (Predicted)
    y_true = df["production_actual_approved"].astype(bool)
    y_pred = df["approved"].astype(bool)

    print("=====================================================")
    print("          SHADOW TESTING REPORT (AI vs HUMAN)        ")
    print("=====================================================")
    print(f"စုစုပေါင်း စမ်းသပ်ခဲ့သည့် အရေအတွက်: {len(df)}")
    print(f"ကိုက်ညီမှု တိကျနှုန်း (Accuracy Score): {accuracy_score(y_true, y_pred):.2%}\n")

    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Rejected", "Approved"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    # Audit Trail သိမ်းထားတဲ့ ဖိုင်လမ်းကြောင်းကို ပေးသွင်းပါ
    analyze_shadow_results("../../backend/logs/audit_trail.jsonl")
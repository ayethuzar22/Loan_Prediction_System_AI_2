# Myanmar Loan Management System — ML Prediction Engine

Production-ready Loan Approval Prediction System built with Django REST Framework, XGBoost, and Flutter.

---

## Project Structure

```
loan_system/
│
├── ml/
│   ├── data/
│   │   └── loan_approval_dataset.csv        ← Kaggle dataset (download separately)
│   ├── models/
│   │   ├── loan_model.joblib                ← Production model
│   │   ├── scaler.joblib                    ← Feature scaler
│   │   ├── selected_features.joblib         ← Feature list
│   │   ├── model_metadata.json              ← Version + metrics
│   │   ├── monitoring.jsonl                 ← Live prediction log
│   │   └── versions/                        ← Timestamped backups
│   └── scripts/
│       ├── 01_data_analysis.py              ← Part 1: Data Understanding
│       ├── 02_feature_engineering.py        ← Part 2: Feature Engineering
│       ├── 03_preprocessing.py              ← Part 3: Preprocessing
│       ├── 04_model_training.py             ← Part 4: Model Training
│       ├── 05_09_engines.py                 ← Parts 5–9: Risk/Rate/Reasons
│       └── 13_production.py                 ← Part 13: Deployment utilities
│
├── backend/
│   ├── loan_app/
│   │   ├── models.py                        ← Django ORM models
│   │   ├── serializers.py                   ← DRF serializers
│   │   ├── services.py                      ← Prediction orchestrator
│   │   ├── views.py                         ← API views
│   │   ├── urls.py                          ← URL routing
│   │   └── management/commands/
│   │       └── retrain_model.py             ← CLI retraining command
│   ├── loan_project/
│   │   └── settings.py                      ← Production settings
│   └── requirements.txt
│
├── flutter/
│   └── lib/
│       ├── services/loan_service.dart       ← Dio API client
│       └── screens/loan_screens.dart        ← Form + Result UI
│
└── docker-compose.yml
```

---

## Quick Start

### 1. Get Kaggle Dataset
```
https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
→ Save as: ml/data/loan_approval_dataset.csv
```

### 2. Install ML dependencies
```bash
pip install numpy pandas scikit-learn xgboost joblib
```

### 3. Train the model
```bash
cd ml/scripts
python model_training.py
```

### 4. Start Django backend
```bash
cd backend
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### 5. Run with Docker (Production)
```bash
cp .env.example .env     # Fill in DB_PASSWORD, SECRET_KEY, etc.
docker-compose up -d
docker-compose exec web python manage.py migrate
docker-compose exec web python manage.py collectstatic
```

### 6. Retrain model
```bash
python manage.py retrain_model
python manage.py retrain_model --kaggle-csv /path/to/new_data.csv
python manage.py retrain_model --dry-run    # test without saving
```

---

## API Reference

### POST /api/loan/predict/
**Headers:** `Authorization: Bearer <JWT>`

**Request:**
```json
{
  "user_id": 42,
  "amount_request": 2000000,
  "month": 24,
  "category_id": 2
}
```

**Approved Response:**
```json
{
  "approved": true,
  "approval_probability": 0.91,
  "risk_score": 22,
  "risk_level": "LOW",
  "recommended_amount": 4500000,
  "interest_rate": 1.5,
  "monthly_installment": 420000,
  "risk_factors": ["No significant risk factors identified"]
}
```

**Rejected Response:**
```json
{
  "approved": false,
  "approval_probability": 0.24,
  "risk_score": 81,
  "risk_level": "HIGH",
  "reasons": [
    "Low income relative to the requested loan amount",
    "High debt-to-income ratio",
    "Poor repayment history"
  ]
}
```

### GET /api/loan/history/
Returns the authenticated user's loan history with risk scores.

---

## ML Architecture

| Component | Description |
|---|---|
| Feature Engine | 25+ engineered features from 6 DB tables |
| Model | XGBoost (selected by ROC-AUC comparison) |
| Risk Scorer | Probability + 10 penalty/bonus factors → 0–100 score |
| Amount Recommender | Income × tenure × risk × repayment × collateral formula |
| Interest Engine | LOW=1.5% / MEDIUM=2.0% / HIGH=2.5% monthly (reducing balance EMI) |
| Rejection Engine | 10 threshold-based rules generating human-readable reasons |

## Risk Score Thresholds

| Score | Level | Monthly Rate |
|---|---|---|
| 0–30 | LOW | 1.5% |
| 31–60 | MEDIUM | 2.0% |
| 61–100 | HIGH | 2.5% |

## Model Comparison (Expected on Kaggle Dataset)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.88 | ~0.87 | ~0.90 | ~0.88 | ~0.95 |
| Random Forest | ~0.97 | ~0.97 | ~0.98 | ~0.97 | ~0.99 |
| **XGBoost** | **~0.98** | **~0.98** | **~0.99** | **~0.98** | **~0.99** |

---

## Production Checklist

- [x] JWT authentication on all endpoints
- [x] Per-user rate limiting (10 req/min)
- [x] Model versioning with rollback
- [x] Daily drift monitoring via Celery Beat
- [x] Weekly automated retraining
- [x] Rotating log files (10MB × 5 backups)
- [x] Audit trail (immutable JSONL)
- [x] Redis caching
- [x] PostgreSQL connection pooling
- [x] Gunicorn with 4 workers + preload
- [x] Nginx reverse proxy
- [x] Docker Compose orchestration

---

## Environment Variables (.env)

```
DJANGO_SECRET_KEY=your-very-secret-key-here
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com
DB_NAME=loan_db
DB_USER=postgres
DB_PASSWORD=strong-password-here
DB_HOST=db
DB_PORT=5432
REDIS_URL=redis://redis:6379/1
CORS_ORIGINS=https://app.yourdomain.com
```

"""
=============================================================
PART 13: PRODUCTION DEPLOYMENT
=============================================================
  - Retraining pipeline
  - Model versioning
  - Performance monitoring
  - Audit trail
  - Management command for retraining
"""

# ─────────────────────────────────────────────────────────────
# ml/scripts/retrain_pipeline.py
# ─────────────────────────────────────────────────────────────

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import django
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR    = Path(__file__).parent.parent / "models"
VERSIONS_DIR  = MODELS_DIR / "versions"
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)


class ModelVersionManager:
    """
    Manages model versioning with rollback support.
    Every retrain creates a timestamped backup of the current model.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir  = models_dir
        self.versions_dir = models_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)

    def backup_current(self) -> str:
        """Create a versioned backup of the current production model."""
        current_model = self.models_dir / "loan_model.joblib"
        if not current_model.exists():
            return None

        ts      = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup  = self.versions_dir / f"loan_model_{ts}.joblib"
        joblib.dump(joblib.load(current_model), backup)

        logger.info(f"[Version] Backup created: {backup.name}")
        return str(backup)

    def rollback(self, version_timestamp: str) -> bool:
        """Restore a previous model version."""
        version_file = self.versions_dir / f"loan_model_{version_timestamp}.joblib"
        if not version_file.exists():
            logger.error(f"[Version] Version not found: {version_timestamp}")
            return False

        current = self.models_dir / "loan_model.joblib"
        model   = joblib.load(version_file)
        joblib.dump(model, current)
        logger.info(f"[Version] Rolled back to: {version_timestamp}")
        return True

    def list_versions(self) -> list:
        return sorted([f.name for f in self.versions_dir.glob("loan_model_*.joblib")])


class ModelMonitor:
    """
    Monitors model performance over time.
    Logs prediction statistics and triggers retraining alerts.
    """

    ALERT_THRESHOLD_APPROVAL_RATE = 0.85  # alert if >85% approvals (possible drift)
    ALERT_THRESHOLD_REJECTION_RATE = 0.85  # alert if >85% rejections
    MIN_SAMPLE_SIZE = 100                  # minimum predictions before alerting

    def __init__(self):
        self.log_path = MODELS_DIR / "monitoring.jsonl"

    def log_prediction(self, prediction: dict, features: dict):
        """Append a prediction log entry (JSONL format for easy streaming analysis)."""
        entry = {
            "ts":                  datetime.utcnow().isoformat(),
            "approved":            prediction["approved"],
            "approval_probability": prediction["approval_probability"],
            "risk_score":          prediction["risk_score"],
            "risk_level":          prediction["risk_level"],
            "income":              features.get("income_average"),
            "lti_ratio":           features.get("loan_to_income_ratio"),
            "repayment_score":     features.get("repayment_score"),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def check_drift(self) -> dict:
        """
        Reads monitoring log and returns drift indicators.
        Call this daily via a cron job or Celery beat.
        """
        if not self.log_path.exists():
            return {"status": "no_data"}

        records = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if len(records) < self.MIN_SAMPLE_SIZE:
            return {"status": "insufficient_data", "count": len(records)}

        # Last 500 predictions
        recent = records[-500:]
        df = pd.DataFrame(recent)

        approval_rate = df["approved"].mean()
        avg_risk      = df["risk_score"].mean()
        avg_prob      = df["approval_probability"].mean()

        alerts = []
        if approval_rate > self.ALERT_THRESHOLD_APPROVAL_RATE:
            alerts.append(f"High approval rate detected: {approval_rate:.1%} — model may be too lenient")
        if approval_rate < (1 - self.ALERT_THRESHOLD_REJECTION_RATE):
            alerts.append(f"Very high rejection rate: {1-approval_rate:.1%} — model may be too strict")
        if avg_risk > 70:
            alerts.append(f"Average risk score very high: {avg_risk:.1f}")

        return {
            "status":          "ok" if not alerts else "alert",
            "sample_size":     len(recent),
            "approval_rate":   round(float(approval_rate), 3),
            "avg_risk_score":  round(float(avg_risk), 1),
            "avg_probability": round(float(avg_prob), 3),
            "alerts":          alerts,
        }


class AuditTrail:
    """
    Writes immutable audit logs for all prediction decisions.
    Format: JSONL, append-only.
    """

    def __init__(self):
        audit_dir = MODELS_DIR.parent.parent / "backend" / "logs"
        audit_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = audit_dir / "audit_trail.jsonl"

    def record(
        self,
        user_id: int,
        loan_history_id: int,
        features: dict,
        prediction: dict,
        model_version: str,
    ):
        entry = {
            "ts":               datetime.utcnow().isoformat(),
            "user_id":          user_id,
            "loan_history_id":  loan_history_id,
            "model_version":    model_version,
            "features_hash":    hash(json.dumps(features, sort_keys=True)),
            "approved":         prediction["approved"],
            "probability":      prediction["approval_probability"],
            "risk_score":       prediction.get("risk_score"),
            "risk_level":       prediction.get("risk_level"),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────────────────────
# Django management command: python manage.py retrain_model
# backend/loan_app/management/commands/retrain_model.py
# ─────────────────────────────────────────────────────────────

RETRAIN_COMMAND = '''
"""
Usage:
  python manage.py retrain_model
  python manage.py retrain_model --kaggle-csv /path/to/data.csv
  python manage.py retrain_model --dry-run
"""
import sys
from pathlib import Path
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Retrain the loan prediction model and update production artifact."

    def add_arguments(self, parser):
        parser.add_argument(
            "--kaggle-csv",
            type=str,
            default=None,
            help="Path to Kaggle training CSV",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Run without saving new model",
        )

    def handle(self, *args, **options):
        from django.conf import settings
        ml_dir = Path(settings.BASE_DIR).parent / "ml"
        sys.path.insert(0, str(ml_dir / "scripts"))

        from retrain_pipeline import ModelVersionManager
        from scripts_04_model_training import train_all   # type: ignore

        kaggle_csv = options["kaggle_csv"] or str(ml_dir / "data" / "loan_approval_dataset.csv")

        if not Path(kaggle_csv).exists():
            self.stderr.write(f"CSV not found: {kaggle_csv}")
            return

        version_mgr = ModelVersionManager(ml_dir / "models")

        # Backup current model
        backup_path = version_mgr.backup_current()
        if backup_path:
            self.stdout.write(f"Backed up current model to: {backup_path}")

        if options["dry_run"]:
            self.stdout.write("[DRY RUN] Would retrain. Skipping save.")
            return

        # Retrain
        self.stdout.write("Starting retraining...")
        model_path = train_all(kaggle_csv)
        self.stdout.write(self.style.SUCCESS(f"✓ Retraining complete: {model_path}"))

        # Reload predictor singleton
        import importlib
        import loan_app.services as svc
        svc._predictor = None   # force reload on next request
        self.stdout.write("✓ Model singleton cleared — new model loads on next request.")
'''


# ─────────────────────────────────────────────────────────────
# docker-compose.yml (production)
# ─────────────────────────────────────────────────────────────

DOCKER_COMPOSE = '''
version: "3.9"

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: loan_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru

  web:
    build: ./backend
    command: gunicorn loan_project.wsgi:application
             --bind 0.0.0.0:8000
             --workers 4
             --threads 2
             --timeout 120
             --access-logfile -
             --error-logfile -
    volumes:
      - ./ml/models:/app/ml/models:ro
      - ./backend/logs:/app/logs
      - ./backend/media:/app/media
    env_file: .env
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    ports:
      - "8000:8000"

  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./backend/staticfiles:/app/staticfiles:ro
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - web

  celery:
    build: ./backend
    command: celery -A loan_project worker --loglevel=info --concurrency=2
    volumes:
      - ./ml/models:/app/ml/models:ro
    env_file: .env
    depends_on:
      - redis
      - db

  celery-beat:
    build: ./backend
    command: celery -A loan_project beat --loglevel=info
    env_file: .env
    depends_on:
      - redis

volumes:
  postgres_data:
'''


# ─────────────────────────────────────────────────────────────
# requirements.txt
# ─────────────────────────────────────────────────────────────

REQUIREMENTS = '''
# Django backend
django==4.2.*
djangorestframework==3.15.*
djangorestframework-simplejwt==5.3.*
django-cors-headers==4.3.*
django-filter==23.5
django-redis==5.4.*
psycopg2-binary==2.9.*
gunicorn==21.2.*
whitenoise==6.6.*
celery[redis]==5.3.*

# ML
numpy==1.26.*
pandas==2.1.*
scikit-learn==1.4.*
xgboost==2.0.*
joblib==1.3.*

# Monitoring / utilities
python-dotenv==1.0.*
sentry-sdk==1.39.*
'''


# ─────────────────────────────────────────────────────────────
# Celery periodic tasks (model drift monitoring)
# ─────────────────────────────────────────────────────────────

CELERY_TASKS = '''
# backend/loan_app/tasks.py

from celery import shared_task
from celery.utils.log import get_task_logger
import sys
from pathlib import Path
from django.conf import settings

logger = get_task_logger(__name__)

sys.path.insert(0, str(Path(settings.BASE_DIR).parent / "ml" / "scripts"))


@shared_task
def check_model_drift():
    """Run daily drift check and alert if thresholds exceeded."""
    from retrain_pipeline import ModelMonitor
    monitor = ModelMonitor()
    report  = monitor.check_drift()

    if report.get("status") == "alert":
        logger.warning(f"[Monitor] DRIFT ALERT: {report}")
        # TODO: send Slack/email alert here
    else:
        logger.info(f"[Monitor] Drift check OK: {report}")

    return report


@shared_task
def scheduled_retraining():
    """Weekly model retraining task."""
    import subprocess
    result = subprocess.run(
        ["python", "manage.py", "retrain_model"],
        cwd=str(Path(settings.BASE_DIR)),
        capture_output=True,
        text=True,
    )
    logger.info(f"[Retrain] stdout: {result.stdout}")
    if result.returncode != 0:
        logger.error(f"[Retrain] FAILED: {result.stderr}")
    return result.returncode


# backend/loan_project/celery.py
CELERY_CONFIG = """
from celery.schedules import crontab

CELERYBEAT_SCHEDULE = {
    "daily-drift-check": {
        "task":     "loan_app.tasks.check_model_drift",
        "schedule": crontab(hour=8, minute=0),   # Every day at 8am
    },
    "weekly-retrain": {
        "task":     "loan_app.tasks.scheduled_retraining",
        "schedule": crontab(day_of_week="sunday", hour=2),  # Sunday 2am
    },
}
"""
'''


if __name__ == "__main__":
    print("Production deployment configuration generated.")
    print("\nFiles to create manually:")
    print("  backend/loan_app/management/commands/retrain_model.py")
    print("  backend/loan_app/tasks.py")
    print("  docker-compose.yml")
    print("  requirements.txt")
    print("\nSee this file for the content of each.")

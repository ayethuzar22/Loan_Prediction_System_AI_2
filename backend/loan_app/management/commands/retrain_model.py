"""
Django management command: python manage.py retrain_model

Usage:
  python manage.py retrain_model
  python manage.py retrain_model --kaggle-csv /path/to/data.csv
  python manage.py retrain_model --dry-run
"""

import sys
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Retrain the loan prediction model and update production artifact."

    def add_arguments(self, parser):
        parser.add_argument(
            "--kaggle-csv",
            type=str,
            default=None,
            help="Path to Kaggle training CSV (default: ml/data/myanmar_loan_mock_data.csv)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Run preprocessing only — don't save new model",
        )

    def handle(self, *args, **options):
        from django.conf import settings
        ml_dir = Path(settings.BASE_DIR).parent / "ml"
        sys.path.insert(0, str(ml_dir / "scripts"))

        try:
            from retrain_pipeline import ModelVersionManager  # type: ignore
            from scripts_04_model_training import train_all   # type: ignore
        except ImportError as e:
            raise CommandError(f"Failed to import ML scripts: {e}")

        kaggle_csv = options["kaggle_csv"] or str(
            ml_dir / "data" / "loan_approval_dataset.csv"
        )

        if not Path(kaggle_csv).exists():
            raise CommandError(f"CSV not found: {kaggle_csv}")

        # ── Backup current model ──────────────────────────────
        version_mgr = ModelVersionManager(ml_dir / "models")
        backup_path = version_mgr.backup_current()
        if backup_path:
            self.stdout.write(f"  Backed up current model: {backup_path}")

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING("[DRY RUN] Retraining skipped."))
            return

        # ── Retrain ───────────────────────────────────────────
        self.stdout.write("  Starting retraining pipeline...")
        try:
            model_path = train_all(kaggle_csv)
            self.stdout.write(
                self.style.SUCCESS(f"✓ Retraining complete: {model_path}")
            )
        except Exception as e:
            # Rollback on failure
            self.stderr.write(f"Retraining failed: {e}")
            if backup_path:
                ts = Path(backup_path).stem.replace("loan_model_", "")
                version_mgr.rollback(ts)
                self.stdout.write("  Rolled back to previous model.")
            raise CommandError(str(e))

        # ── Reload predictor singleton ────────────────────────
        try:
            import loan_app.services as svc
            svc._predictor = None
            self.stdout.write("  Model singleton cleared — new model loads on next request.")
        except Exception:
            pass

        self.stdout.write(self.style.SUCCESS("✓ Model deployment complete."))

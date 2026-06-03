"""APScheduler-backed daily signal scheduler.

Registers a single daily job at the configurable IST time.
Re-schedulable from the config panel without restarting the app.
Safe for single-process Flask (no asyncio, no double-scheduling on reloader).
"""

import os

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from utils.logging import get_logger

logger = get_logger(__name__)

_JOB_ID = "daily_signals"
_scheduler: BackgroundScheduler | None = None


def _ist_to_utc(hour_ist: int, minute_ist: int) -> tuple[int, int]:
    """Convert IST (UTC+5:30) to UTC."""
    total_minutes = hour_ist * 60 + minute_ist - 330  # -5h30m
    total_minutes %= 1440  # wrap around midnight
    return divmod(total_minutes, 60)


def _run_job() -> None:
    """Entry point called by APScheduler."""
    from services.signal_service import run_signals

    logger.info("APScheduler: starting scheduled signal run")
    try:
        run_id = run_signals()
        logger.info("APScheduler: signal run %d completed", run_id)
    except Exception:
        logger.exception("APScheduler: signal run failed")


def get_scheduler() -> BackgroundScheduler:
    """Return the shared scheduler instance (creates it on first call)."""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(timezone="UTC")
    return _scheduler


def schedule_job(hour_ist: int, minute_ist: int) -> None:
    """Add or replace the daily job with new IST time.

    Args:
        hour_ist: Hour component of the desired IST run time (0-23).
        minute_ist: Minute component of the desired IST run time (0-59).
    """
    scheduler = get_scheduler()
    hour_utc, minute_utc = _ist_to_utc(hour_ist, minute_ist)
    trigger = CronTrigger(hour=hour_utc, minute=minute_utc, timezone="UTC")

    if scheduler.get_job(_JOB_ID):
        scheduler.reschedule_job(_JOB_ID, trigger=trigger)
        logger.info(
            "Rescheduled daily job: IST %02d:%02d → UTC %02d:%02d",
            hour_ist, minute_ist, hour_utc, minute_utc,
        )
    else:
        scheduler.add_job(_run_job, trigger=trigger, id=_JOB_ID, replace_existing=True)
        logger.info(
            "Scheduled daily job: IST %02d:%02d → UTC %02d:%02d",
            hour_ist, minute_ist, hour_utc, minute_utc,
        )


def init_scheduler(hour_ist: int, minute_ist: int) -> None:
    """Start the scheduler and register the daily job.

    Safe to call from the Flask app factory; uses ``WERKZEUG_RUN_MAIN``
    to avoid double-scheduling when Flask's reloader forks the process.

    Args:
        hour_ist: Desired IST hour for the daily run.
        minute_ist: Desired IST minute for the daily run.
    """
    # Avoid double-scheduling in Werkzeug's reloader (parent process)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "false":
        logger.debug("Skipping scheduler init in Werkzeug parent process.")
        return

    scheduler = get_scheduler()
    if not scheduler.running:
        scheduler.start()
        logger.info("APScheduler started.")

    schedule_job(hour_ist, minute_ist)


def shutdown_scheduler() -> None:
    """Stop the scheduler gracefully."""
    scheduler = get_scheduler()
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped.")

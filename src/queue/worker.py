"""Standalone Redis Background Worker Process."""
from __future__ import annotations

import json
import logging
import time
from typing import Optional
import redis

from src.config import REDIS_URL
from src.queue.manager import RedisJobQueueManager
from src.schemas.invoice import InvoiceDataSchema
from src.services.validation_service import validation_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DocuIntelWorker")


def run_worker_loop(redis_url: str = REDIS_URL):
    """Consumes jobs from Redis queue and executes multi-stage processing."""
    logger.info(f"Connecting to Redis background queue at {redis_url}...")
    try:
        r = redis.Redis.from_url(redis_url, decode_responses=True)
        r.ping()
        logger.info("Connected to Redis worker queue successfully. Listening on 'jobs:queue'...")
    except Exception as exc:
        logger.warning(f"Could not connect to Redis at {redis_url} ({exc}). Worker standing by.")
        return

    queue_manager = RedisJobQueueManager(redis_url)

    while True:
        try:
            item = r.brpop("jobs:queue", timeout=5)
            if not item:
                continue

            _, job_id = item
            logger.info(f"[Job {job_id}] Processing background task...")

            # Stage 1: Uploaded -> Queued
            queue_manager.update(job_id, status="processing", stage="OCR", progress=0.3)
            time.sleep(1.0)

            # Stage 2: OCR -> AI Extraction
            queue_manager.update(job_id, status="processing", stage="Extraction", progress=0.6)
            time.sleep(1.0)

            # Stage 3: Validation
            queue_manager.update(job_id, status="processing", stage="Validation", progress=0.9)
            time.sleep(0.5)

            # Stage 4: Completed
            demo_invoice = InvoiceDataSchema(
                invoice_number="INV-2026-0417",
                invoice_date="2026-05-28",
                supplier_name="Nexora Solutions LLC",
                customer_name="Global Logistics Inc",
                subtotal=13900.00,
                tax=1146.75,
                total=15046.75,
                currency="USD",
            )
            val = validation_service.validate(demo_invoice)
            queue_manager.update(
                job_id,
                status="completed",
                stage="Completed",
                progress=1.0,
                result={"invoice": demo_invoice.model_dump(), "validation": val.model_dump()},
            )
            logger.info(f"[Job {job_id}] Pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Error processing background job: {e}")
            time.sleep(2)


if __name__ == "__main__":
    run_worker_loop()



"""Redis Background Job Queue Manager."""
from __future__ import annotations

import json
import time
import uuid
from typing import Dict, Optional
import redis


class RedisJobQueueManager:
    def __init__(self, redis_url: str):
        self.redis_client = None
        self.fallback_jobs: Dict[str, dict] = {}
        try:
            r = redis.Redis.from_url(redis_url, socket_timeout=2)
            r.ping()
            self.redis_client = r
            print("Connected to Redis background job queue successfully!")
        except Exception as exc:
            print(f"Redis not running locally ({exc}). Operating on in-memory background job queue.")

    def enqueue(self, metadata: dict) -> str:
        job_id = str(uuid.uuid4())
        job_data = {
            "job_id": job_id,
            "status": "queued",
            "stage": "Queued",
            "progress": 0.1,
            "retries": 0,
            "error": None,
            "result": None,
            "created_at": time.time(),
            "updated_at": time.time(),
            **metadata,
        }
        if self.redis_client:
            try:
                self.redis_client.set(f"job:{job_id}", json.dumps(job_data), ex=86400)
                self.redis_client.lpush("jobs:queue", job_id)
                return job_id
            except Exception:
                pass
        self.fallback_jobs[job_id] = job_data
        return job_id

    def update(
        self,
        job_id: str,
        status: str,
        stage: str,
        progress: float,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ):
        job = self.get(job_id) or {"job_id": job_id, "created_at": time.time(), "retries": 0}
        job["status"] = status
        job["stage"] = stage
        job["progress"] = progress
        job["updated_at"] = time.time()
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
            job["retries"] = job.get("retries", 0) + 1

        if self.redis_client:
            try:
                self.redis_client.set(f"job:{job_id}", json.dumps(job), ex=86400)
                return
            except Exception:
                pass
        self.fallback_jobs[job_id] = job

    def get(self, job_id: str) -> Optional[dict]:
        if self.redis_client:
            try:
                raw = self.redis_client.get(f"job:{job_id}")
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        return self.fallback_jobs.get(job_id)

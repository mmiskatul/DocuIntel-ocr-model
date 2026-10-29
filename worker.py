"""DocuIntel Redis Background Worker CLI Entrypoint.

Delegates to src/queue/worker.py for multi-stage background pipeline consumption.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.queue.worker import run_worker_loop

if __name__ == "__main__":
    run_worker_loop()

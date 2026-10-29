"""Forwarding script to scripts/test_client.py"""
from scripts.test_client import test_service, main if False else None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test OCR & LLM microservice")
    parser.add_argument("--url", default="http://localhost:7860", help="Service base URL")
    parser.add_argument("--key", default="", help="Optional X-API-Key")
    args = parser.parse_args()
    test_service(args.url, args.key)

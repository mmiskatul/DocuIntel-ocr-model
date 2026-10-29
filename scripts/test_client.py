"""Quick test script to verify your OCR & Multi-LLM microservice.

Usage:
    python scripts/test_client.py --url http://localhost:7860
"""
import argparse
import sys
import httpx


def test_service(url: str, api_key: str = ""):
    url = url.rstrip("/")
    headers = {"X-API-Key": api_key} if api_key else {}

    print(f"Testing OCR & Multi-LLM Service at: {url}\n")

    # 1. Healthcheck
    try:
        r = httpx.get(f"{url}/", headers=headers, timeout=10.0)
        print("[1/3] Root Info Check:", r.status_code)
        print("      Response:", r.json())
    except Exception as e:
        print("      FAILED:", e)
        sys.exit(1)

    # 2. Test Embedding (BAAI/bge-small-en-v1.5)
    try:
        sample_text = "Invoice INV-2024-001 from Acme Corp total amount $1,250.00"
        r = httpx.post(f"{url}/embed", json={"text": sample_text}, headers=headers, timeout=15.0)
        print("\n[2/3] Embedding Check (bge-small-en-v1.5):", r.status_code)
        if r.status_code == 200:
            data = r.json()
            print(f"      Model: {data['model']}, Dim: {data['dimension']}, Time: {data['processing_time_ms']}ms")
            print(f"      First 5 dims: {data['embeddings'][0][:5]}...")
    except Exception as e:
        print("      FAILED:", e)

    # 3. Test Reconciliation
    try:
        r = httpx.post(f"{url}/reconcile", headers=headers, timeout=10.0)
        print("\n[3/3] Reconciliation Endpoint Check:", r.status_code)
        if r.status_code == 200:
            data = r.json()
            summary = data["summary"]
            print(f"      Matched: {summary['matched_count']}/{summary['total_transactions']}, Flagged: {summary['flagged_count']}")
    except Exception as e:
        print("      FAILED:", e)

    print("\nAll checks completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OCR & LLM microservice")
    parser.add_argument("--url", default="http://localhost:7860", help="Service base URL")
    parser.add_argument("--key", default="", help="Optional X-API-Key")
    args = parser.parse_args()
    test_service(args.url, args.key)


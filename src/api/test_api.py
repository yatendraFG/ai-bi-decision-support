"""
API smoke test (no server needed - FastAPI's TestClient calls the app directly).

    python -m src.api.test_api
"""
from fastapi.testclient import TestClient

from src.api.main import app

CASES = [
    ("get", "/health", None, 200),
    ("get", "/forecast?region=West", None, 200),
    ("get", "/sales/trend?region=West&category=Electronics", None, 200),
    ("get", "/churn?region=North&top_n=3", None, 200),
    ("get", "/segments", None, 200),
    ("get", "/tickets/insights?category=Billing", None, 200),
    ("get", "/tickets/urgent?top_n=2", None, 200),
    ("get", "/anomalies?top_n=2", None, 200),
    ("get", "/customers/c00509", None, 200),
    ("get", "/customers/C99999", None, 404),          # unknown customer
    ("get", "/forecast?region=Mars", None, 422),       # invalid input is rejected
    ("post", "/tickets/classify", {"text": "Refund abhi tak nahi aaya, 3 hafte ho gaye!!", "priority": "High"}, 200),
    ("post", "/tickets/classify", {"text": "hi"}, 422),  # too short
    ("post", "/ask/documents", {"question": "Who approves a refund of Rs 25,000?"}, 200),
    ("post", "/ask/agent", {"question": "Why are West sales falling?"}, 200),
]


def main():
    c, failed = TestClient(app), 0
    for method, url, body, expected in CASES:
        r = c.post(url, json=body) if method == "post" else c.get(url)
        ok = r.status_code == expected
        failed += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {r.status_code} {method.upper():4s} {url}")
    j = c.post("/tickets/classify", json=CASES[11][2]).json()
    assert j["sentiment"]["label"] == "negative" and j["category"]["label"] == "Refund", j
    print(f"\n{len(CASES) - failed}/{len(CASES)} endpoint checks passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

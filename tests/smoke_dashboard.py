"""
Opens every dashboard page headlessly and fails if any page raises an error.
Used by GitHub Actions on every push.

    python tests/smoke_dashboard.py
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")
PAGES = ["Overview", "Sales", "Customers", "Support", "Anomalies", "Ask AI"]


def main():
    failed = 0
    for page in PAGES:
        at = AppTest.from_file(APP, default_timeout=180)
        at.run()
        at.sidebar.radio[0].set_value(page).run()
        errs = [e.value for e in at.exception]
        failed += bool(errs)
        print(f"{'OK  ' if not errs else 'FAIL'} {page}" + (f": {errs[0][:200]}" if errs else ""))
    print(f"\n{len(PAGES) - failed}/{len(PAGES)} dashboard pages OK")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

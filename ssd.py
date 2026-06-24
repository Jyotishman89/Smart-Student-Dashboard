"""Smart Student Dashboard — Streamlit entrypoint.

Kept as `ssd.py` so the existing Streamlit Community Cloud deployment (whose
"Main file path" points here) keeps working with no dashboard change. All logic
lives in the `ssd/` package.

Run locally with:  streamlit run ssd.py
"""
from ssd.main import run

run()

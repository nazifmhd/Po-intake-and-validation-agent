"""Streamlit review desk - where uncertain POs live until a person acts on them.

    streamlit run review_ui/app.py

Overview · Send a PO · Review queue · Filed POs · Knowledge base · Audit log. The queue shows why a PO was held,
the source document with the flagged values marked on it, the RAG evidence, and a form to correct misread values.
All rules are enforced in app/actions/review.py, not here, so the API can't be used to get around them.
"""
from __future__ import annotations

import hmac
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# This file is itself named app.py: drop its folder from sys.path (and any shadowing module)
# so `import app` resolves to the project package, not to this script.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.path.insert(0, str(ROOT))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import streamlit as st  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.rag.rules import load_rules  # noqa: E402
from review_ui import data, pages  # noqa: E402
from review_ui.theme import CSS, esc, put  # noqa: E402

settings = get_settings()
st.set_page_config(page_title="PO review desk", page_icon=":material/receipt_long:", layout="wide",
                   initial_sidebar_state="expanded")
put(CSS)

if settings.review_ui_password and not st.session_state.get("authenticated"):
    st.title("PO review desk")
    entered = st.text_input("Password", type="password")
    if entered and hmac.compare_digest(entered.encode(), settings.review_ui_password.encode()):
        st.session_state["authenticated"] = True
        st.rerun()
    elif entered:
        st.error("Wrong password")
    st.stop()

waiting = len(data.pending())
PAGES = [
    ("overview", "Overview", ":material/monitoring:"), ("send", "Send a PO", ":material/outgoing_mail:"),
    ("queue", f"Review queue ({waiting})" if waiting else "Review queue", ":material/rule:"),
    ("filed", "Filed POs", ":material/inventory_2:"), ("knowledge", "Knowledge base", ":material/menu_book:"),
    ("audit", "Audit log", ":material/history:"),
]
current = st.query_params.get("page", "overview")
if current not in {p[0] for p in PAGES}:
    current = "overview"

with st.sidebar:
    put('<div class="brand"><div class="mark"><i></i><i></i><i></i></div><div><b>PO review desk</b>'
        '<span>Northbridge Manufacturing</span></div></div>')
    for code, label, icon in PAGES:
        if st.button(label, key=f"nav-{code}", icon=icon, type="primary" if code == current else "tertiary", width="stretch"):
            pages.go(code)
            st.rerun()
    reviewer = st.text_input("Reviewer name", key="reviewer", placeholder="Your name")
    rules = load_rules(settings)
    model = {"anthropic": settings.anthropic_model, "openai": settings.openai_model}.get(settings.llm_provider, "offline parser")
    put(f'<div class="side-foot">Reads with <b>{esc(model)}</b><br>Auto-approves up to <b>${rules.auto_approval_limit:,.0f}</b><br>'
        f'Price tolerance <b>±{rules.price_tolerance_pct:g}%</b><br>Confidence at least <b>{settings.critical_field_confidence_min:.2f}</b></div>')

if st.session_state.get("flash"):
    st.success(st.session_state.pop("flash"), icon=":material/check_circle:")

if current == "queue":
    pages.queue(reviewer)
else:
    getattr(pages, current)()

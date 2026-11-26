"""The pages of the review desk. Rules live in app/actions/review.py; nothing here decides anything."""
from __future__ import annotations

import json
import re
import textwrap

import pandas as pd
import streamlit as st

from app.actions import review as review_actions
from app.config import PROJECT_ROOT, get_settings
from app.extraction.schema import PurchaseOrder
from app.intake.mailer import send_email, smtp_ready
from app.intake.simulate import NotConfigured, post_email
from app.models import ReviewQueueItem
from app.rag.rules import load_rules
from app.rag.store import load_catalog, load_policy_chunks, load_vendors
from review_ui import data, document
from review_ui.theme import (
    CHECKS, FIELD_LABELS, ago, chip, conf_meter, decision_chip, empty, esc, flag_info, gauge, money, put, title,
)

jl = data.jl
DEMO = [
    ("01_clean_northwind.pdf", "Clean PDF", "Known vendor, catalog prices, $4,200"),
    ("02_scanned_globex.jpg", "Photographed PO", "Noisy, rotated scan with an inflated price"),
    ("03_unknown_vendor.csv", "CSV, unknown vendor", "Not in the registry, no delivery date"),
    ("05_blocked_vendor.pdf", "Blocked vendor", "Failed due diligence in June"),
    ("06_math_error.pdf", "Total doesn't add up", "Printed total is $100 too high"),
    ("10_lookalike_vendor.pdf", "Lookalike vendor", "Apex: spelled 92% like approved vendor Acme"),
]


def go(page: str, item: int | None = None) -> None:
    st.query_params["page"] = page
    if item is None:
        st.query_params.pop("item", None)
    else:
        st.query_params["item"] = str(item)


def done(message: str) -> None:
    st.session_state["flash"] = message  # shown after the rerun
    st.rerun()


# ================================================================== overview
def _activity_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        if r["status"] in ("queued", "processing"):
            sub = "reading and checking…" if r["status"] == "processing" else "waiting for a worker"
        elif r["flags"]:
            names = [flag_info(f)[0] for f in r["flags"]]
            sub = " · ".join(names[:2]) + (f" +{len(names) - 2}" if len(names) > 2 else "")
        elif r["status"] == "refused":
            sub = r["note"] or ""
        elif r["status"] == "duplicate":
            sub = "identical PO already on record"
        else:
            sub = "all checks passed" if r["status"] == "approved" else (r["from"] or "")
        head = r["po_number"] or r["file"]
        out.append(f'<div class="row"><span class="t">{ago(r["when"])}</span>'
                   f'<span class="ell"><b>{esc(head)}</b><span class="sub">{esc(sub)}</span></span>'
                   f'<span class="ell">{esc(r["vendor"])}</span>'
                   f'<span class="num">{money(r["total"], r["currency"]) if r["total"] is not None else ""}</span>'
                   f'<span>{decision_chip(r["status"])}</span></div>')
    return "".join(out)


@st.fragment(run_every="3s")
def _live_overview() -> None:
    s = data.summary()
    processed = s["processed"]
    rate = f"{s['auto'] / processed:.0%}" if processed else "0%"
    parts = [("Filed automatically", s["auto"], "#4fb6a6"), ("Approved by a person", s["by_person"], "#8fb7f0"),
             ("Waiting for review", s["waiting"], "#f2a93b"), ("Rejected", s["rejected"], "#ec7f73")]
    seg = "".join(f'<i style="flex:{n};background:{c}" title="{esc(label)}: {n}"></i>' for label, n, c in parts if n)
    leg = "".join(f'<span style="--c:{c}">{esc(label)} {n}</span>' for label, n, c in parts if n)
    put(f'<div class="tiles"><div class="card tile lead"><span class="l">Purchase orders processed</span>'
        f'<div class="v">{processed}</div><div class="seg">{seg}</div><div class="leg">{leg}</div></div>'
        f'<div class="card tile"><span class="l">Waiting for review</span><div class="v">{s["waiting"]}</div>'
        f'<span class="s">{money(s["value_waiting"])} held</span></div>'
        f'<div class="card tile"><span class="l">Filed automatically</span><div class="v">{rate}</div>'
        f'<span class="s">{s["auto"]} of {processed}, every check passed</span></div>'
        f'<div class="card tile"><span class="l">Value filed</span><div class="v">{money(s["value_filed"])}</div>'
        f'<span class="s">{s["duplicates"]} duplicate(s) skipped, {s["refused"]} file(s) refused</span></div></div>')
    left, right = st.columns([5, 2], gap="medium")
    with left:
        rows = data.activity()
        live = f'<small>{s["running"]} in progress</small>' if s["running"] else "<small>updates by itself</small>"
        body = _activity_rows(rows) or '<div class="empty"><b>No purchase orders yet</b>Send one from the Send a PO page.</div>'
        put(f'<div class="card"><h3 class="sec">Latest activity{live}</h3>{body}</div>')
    with right:
        top = s["flags"][0][1] if s["flags"] else 1
        bars = "".join(f'<div class="bar" title="{esc(code)}: {n}"><span class="ell">{esc(flag_info(code)[0])}</span>'
                       f'<span class="num">{n}</span><span class="track"><i style="width:{n / top * 100:.0f}%"></i></span></div>'
                       for code, n in s["flags"])
        put('<div class="card"><h3 class="sec">Why POs were held</h3>'
            + (f'<div class="bars">{bars}</div>' if bars else '<div class="empty">Nothing has been held.</div>') + "</div>")


def overview() -> None:
    title("Overview", "Every purchase order is read, checked against the vendor registry, the catalog and the "
                      "procurement policy, then filed or handed to a person.")
    _live_overview()


# ================================================================== send a PO
def _send(files: list[tuple[str, bytes]], sender: str, subject: str, forged: bool = False) -> None:
    settings = get_settings()
    if st.session_state.get("transport") == "Real email" and not forged:
        try:
            with st.spinner("Sending the email…"):
                send_email(files, settings, subject=subject)
            st.session_state["sent"] = ("good", "Email sent. It appears below as soon as the agent picks it up from "
                                                f"{settings.test_email_to or settings.imap_user}.")
        except Exception as exc:
            from app.doctor import _explain
            st.session_state["sent"] = ("bad", f"The email could not be sent: {_explain(exc)}")
        return
    try:
        r = post_email(files, settings, url=settings.api_base_url, sender=sender, subject=subject, bad_signature=forged)
    except NotConfigured as exc:
        st.session_state["sent"] = ("bad", str(exc))
        return
    except Exception as exc:
        st.session_state["sent"] = ("bad", f"The API at {settings.api_base_url} did not answer: {exc}")
        return
    if r.status_code == 200:
        body = r.json()
        n = len(body["jobs"])
        why = {"rejected_sender": f"{sender} is not an allowed sender (ALLOWED_SENDERS in .env), so the agent refused it",
               "duplicate": "this exact email was already received", "all_rejected": "no attachment could be used",
               "no_attachment": "the email had no attachment"}.get(body["status"], body["status"])
        text = f"Accepted: {n} document{'s' if n > 1 else ''} on the way to the agent" if body["jobs"] else f"Not processed: {why}"
        if body["rejected"]:
            text += "; refused: " + ", ".join(f"{x['filename']} ({x['reason']})" for x in body["rejected"])
        st.session_state["sent"] = ("good" if body["jobs"] else "warn", text)
    else:
        st.session_state["sent"] = ("bad", f"HTTP {r.status_code}: {r.json().get('detail', r.text)}")


def default_sender() -> str:
    """Demo mail comes from someone the agent accepts: the first allowed sender, if a list is configured."""
    s = get_settings()
    if s.allowed_sender_list:
        return s.allowed_sender_list[0]
    if s.allowed_sender_domain_list:
        return f"Buyer <buyer@{s.allowed_sender_domain_list[0]}>"
    return "Dana Buyer <dana@northbridge.example>"


@st.fragment(run_every="2s")
def _live_feed() -> None:
    rows = data.activity(6)
    put('<div class="card"><h3 class="sec">What the agent is doing<small>updates by itself</small></h3>'
        + (_activity_rows(rows) or '<div class="empty">Nothing received yet.</div>') + "</div>")


def send() -> None:
    real = smtp_ready(get_settings())
    title("Send a PO", "Sends a purchase order to the agent, as a real email or straight to its signed webhook." if real else
          "Posts a signed inbound email to the agent's webhook, the way the mail provider does. "
          "Only the mail transport is simulated.")
    if real:
        st.segmented_control("Send by", ["Real email", "Webhook"], default="Real email", key="transport",
                             label_visibility="collapsed")
    left, right = st.columns([2, 3], gap="medium")
    with left:
        with st.container(border=True, key="box-demo"):
            put('<h3 class="sec">Demo documents</h3>')
            for name, head, body in DEMO:
                path = PROJECT_ROOT / "data" / "samples" / name
                a, b = st.columns([3, 1], vertical_alignment="center")
                a.html(f"<b style='font-weight:600'>{esc(head)}</b><br><span style='color:var(--ink2);font-size:.86rem'>{esc(body)}</span>")
                if b.button("Send", key=f"demo-{name}", width="stretch", disabled=not path.exists()):
                    _send([(name, path.read_bytes())], default_sender(), f"PO attached: {head}")
        with st.container(border=True, key="box-own"):
            put('<h3 class="sec">Your own document</h3>')
            files = st.file_uploader("PDF, image or CSV", accept_multiple_files=True, label_visibility="collapsed")
            sender = st.text_input("From", default_sender())
            x, y = st.columns(2)
            if x.button("Send email", type="primary", width="stretch", disabled=not files):
                _send([(f.name, f.getvalue()) for f in files], sender, "Purchase order attached")
            if y.button("Send with a forged signature", width="stretch"):
                sample = PROJECT_ROOT / "data" / "samples" / DEMO[0][0]
                _send([(f.name, f.getvalue()) for f in files] or [(sample.name, sample.read_bytes())],
                      "attacker@example.com", "Urgent PO", forged=True)
    with right:
        tone, text = st.session_state.get("sent", (None, None))
        if text:
            {"good": st.success, "warn": st.warning, "bad": st.error}[tone](text)
        _live_feed()
        if st.button("Open the review queue", icon=":material/arrow_forward:"):
            go("queue")
            st.rerun()


# ================================================================== review queue
def _header(item: ReviewQueueItem, po: dict, vm: dict) -> None:
    vendor = item.vendor_name_raw or "Vendor not read"
    known, sure = vm.get("canonical_name"), vm.get("status") == "matched"
    meta = [f"Received {ago(item.created_at)}", {"pdf_text": "Text PDF", "pdf_scanned": "Scanned PDF", "pdf_mixed":
            "PDF, text and scan", "image": "Photo or scan", "csv": "CSV"}.get(item.doc_type or "", "Unreadable file")]
    if po:
        meta.append({"csv_columns": "Read from columns", "heuristic": "Read offline"}.get(
            po.get("extraction_method"), "Read by the vision model" if item.doc_type != "pdf_text" else "Read by the model"))
        meta.append(f"Overall confidence {po.get('overall_confidence', 0):.2f}")
    put(f'<div class="pohead"><div><span class="k">Review item {item.id}</span><h2>{esc(item.po_number or "No PO number")}</h2>'
        f'<div class="meta"><b style="color:var(--ink);font-weight:600">{esc(vendor)}</b>'
        + (f"<span>{'matched to' if sure else 'resembles'} {esc(known)}{'' if sure else ', not confirmed'}</span>"
           if known and known.lower() != vendor.lower() else "")
        + "".join(f"<span>{esc(m)}</span>" for m in meta)
        + f'</div></div><div class="amt"><span class="k">Total</span><b>{money(item.total, item.currency)}</b>'
          '<span class="stamp">Held for review</span></div></div>')


def _checks(flags: list[str], po: dict) -> None:
    groups = {flag_info(f)[1] for f in flags}
    cells = []
    for code, label, source in CHECKS:
        n = sum(flag_info(f)[1] == code for f in flags)
        if code in groups:
            state, text = "no", f"✕ {n} issue{'s' if n > 1 else ''}"
        elif (not po and code not in ("reading", "document")) or                 (code in ("items", "math") and not po.get("line_items")):  # nothing to check is not "passed"
            state, text = "na", "not possible"
        else:
            state, text = "ok", "✓ passed"
        cells.append(f'<div class="check {state}"><em>{text}</em><b>{esc(label)}</b><span>{esc(source)}</span></div>')
    put(f'<div class="checks">{"".join(cells)}</div>')


def _flags(flags: list[str], reasons: dict) -> None:
    for code in sorted(flags, key=lambda f: flag_info(f)[2] != "critical"):
        head, _, severity, todo = flag_info(code)
        reason = reasons.get(code) or ""
        cite = re.findall(r"\[policy: ([^\]]+)\]", reason)
        reason = re.sub(r"\s*\[policy: [^\]]+\]", "", reason)
        put(f'<div class="flag {severity}"><span class="ic">{"✕" if severity == "critical" else "!"}</span>'
            f'<span><b>{esc(head)}</b><code>{esc(code)}</code></span><p>{esc(reason)}'
            + "".join(f'<span class="cite">Policy: {esc(c)}</span>' for c in dict.fromkeys(cite))
            + f'</p><span class="do">→ {esc(todo)}</span></div>')


def _fields(po: dict) -> None:
    thr = get_settings().critical_field_confidence_min
    rows = []
    for f in PurchaseOrder.HEADER_FIELDS:
        c = po.get(f) or {}
        missing = c.get("value") is None
        low = not missing and (c.get("confidence") or 0) < thr
        rows.append(f'<tr class="{"hot" if missing and f not in ("vendor_tax_id", "buyer_department") or low else ""}">'
                    f'<td class="k">{FIELD_LABELS[f]}</td><td>{chip("missing", "serious") if missing else "<b style=font-weight:600>" + esc(c["value"]) + "</b>"}'
                    f'<div class="ev">{esc(c.get("evidence"))}</div></td>'
                    f'<td class="n">{"" if missing else conf_meter(c.get("confidence"), thr)}</td></tr>')
    cur = (po.get("currency") or {}).get("value")
    for f in review_actions.AMOUNT_FIELDS:
        v = po.get(f)
        rows.append(f'<tr><td class="k">{f.capitalize()}</td><td class="n" style="text-align:left">'
                    + (f"<b style=font-weight:600>{money(v, cur)}</b>" if v is not None else chip("not printed", "mute"))
                    + (f'<div class="ev">{esc(po.get("totals_evidence"))}</div>' if f == "total" else "")
                    + f'</td><td class="n">{conf_meter(po.get("totals_confidence"), thr) if f == "total" and v is not None else ""}</td></tr>')
    put(f'<div class="card"><table class="f"><tr><th>Field</th><th>Value, and where it was read</th><th class="n">Confidence</th></tr>{"".join(rows)}</table></div>')
    if po.get("extraction_warnings"):
        with st.expander(f"{len(po['extraction_warnings'])} note(s) from reading the document"):
            for w in po["extraction_warnings"]:
                st.text(f"• {w}")


def _items(po: dict, checks: list[dict]) -> None:
    by = {c.get("index", k): c for k, c in enumerate(checks)}
    cur = (po.get("currency") or {}).get("value")
    rows = []
    for k, li in enumerate(po.get("line_items") or []):
        c = by.get(k, {})
        bad = c.get("status") not in ("ok", None) or c.get("uom_mismatch") or c.get("sku_mismatch")
        dev = c.get("deviation_pct")
        verdict = {"price_anomaly": chip(f"{dev:+.1f}%" if dev is not None else "price", "serious"),
                   "unknown_item": chip("not in catalog", "serious"), "no_price": chip("no price", "serious"),
                   "ok": chip(f"{dev:+.1f}%" if dev else "matches", "good")}.get(c.get("status"), chip("not checked"))
        via = {"sku_exact": "by SKU", "description_semantic": f"by meaning ({c.get('match_semantic') or 0:.2f})",
               "description_lexical": "by description"}.get(c.get("match_method"), "")
        qty = f"{li['quantity']:g} {esc(li.get('unit') or '')}" if li.get("quantity") is not None else "?"
        price = money(li.get("unit_price"), cur) if li.get("unit_price") is not None else "?"
        amount = money(li.get("line_total"), cur) if li.get("line_total") is not None else "no amount printed"
        ident = " · ".join(x for x in (esc(li.get("sku") or "no SKU printed"), via and f"matched {via}",
                                       esc(c.get("matched_sku")) if c.get("matched_sku") != li.get("sku") else "") if x)
        rows.append(f'<tr class="{"hot" if bad else ""}"><td><b style="font-weight:600">{esc(li.get("description") or li.get("sku"))}</b>'
                    f'<div class="ev">{ident}</div></td><td class="n">{qty} × {price}<div class="ev">{amount}</div></td>'
                    f'<td class="n">{money(c["catalog_unit_price"]) if c.get("catalog_unit_price") is not None else "–"}</td>'
                    f'<td>{verdict}</td></tr>')
    if not rows:
        empty("No line items were read", "Add them under Correct values.")
        return
    put('<div class="card"><table class="f"><tr><th>Item</th><th class="n">Ordered</th><th class="n">Catalog</th>'
        f'<th>Check</th></tr>{"".join(rows)}</table></div>')


def _vendor(po: dict, vm: dict) -> None:
    s = get_settings()
    top = (vm.get("candidates") or [{}])[0]
    record, sure = vm.get("canonical_name"), vm.get("status") == "matched"
    tone = {"blocked": chip("Blocked", "critical"), "active": chip("Active", "good")}.get(vm.get("vendor_status"), "")
    put(f'<div class="match"><div class="side"><span>As printed on the PO</span><b>{esc((po.get("vendor_name") or {}).get("value") or "not read")}</b></div>'
        f'<div class="arrow">{"→" if sure else ("?" if record else "≠")}</div><div class="side"><span>{"Registry record" if sure or not record else "Closest record, not confirmed"}</span>'
        f'<b>{esc(record or "no match")}</b> {esc(vm.get("vendor_id") or "")} {tone}</div></div>'
        f'<p style="color:var(--ink2);font-size:.9rem;margin:0 0 .8rem">{esc(vm.get("reason"))}</p>')
    if top and not top.get("exact"):
        put('<div class="card">' + gauge("Spelling similarity", top.get("fuzzy"), s.vendor_match_high, 100)
            + gauge("Similarity in meaning (embedding)", top.get("semantic"), s.vendor_semantic_min, 1)
            + '<span style="color:var(--ink3);font-size:.82rem">Both must reach the line. An exact name, or a matching tax ID '
              'with a plausible name, is accepted on its own.</span></div>')
    if vm.get("candidates"):
        rows = "".join(f'<tr><td><b style="font-weight:600">{esc(c["canonical_name"])}</b><div class="ev">via {esc(c.get("matched_alias"))}</div></td>'
                       f'<td class="n">{c.get("fuzzy", 0):.0f}</td><td class="n">{"–" if c.get("semantic") is None else format(c["semantic"], ".2f")}</td>'
                       f'<td>{chip("tax ID matches", "good") if c.get("tax_id_match") else ""}</td><td>{esc(c.get("vendor_status"))}</td></tr>'
                       for c in vm["candidates"])
        put('<h3 class="sec" style="margin-top:1rem">Closest registry records</h3><div class="card"><table class="f"><tr><th>Vendor</th>'
            f'<th class="n">Spelling</th><th class="n">Meaning</th><th></th><th>Status</th></tr>{rows}</table></div>')


def _policy(policy: dict) -> None:
    found = policy.get("findings", [])
    if found:
        rows = "".join(f'<tr><td>{chip("violation", "serious") if f.get("severity") == "violation" else chip("note", "mute")}</td>'
                       f'<td>{esc(f.get("message"))}' + (f'<span class="cite">{esc(f["clause"])}</span>' if f.get("clause") else "")
                       + f'</td><td class="k">{"model" if f.get("source") == "llm" else "rule"}</td></tr>' for f in found)
        put(f'<div class="card"><table class="f"><tr><th></th><th>Finding</th><th>Found by</th></tr>{rows}</table></div>')
    clauses = "".join(f'<div class="clause"><b>{esc(c["heading"])}</b> <span class="cite">relevance {c["score"]:.2f}</span>'
                      f'<p>{esc(c["text"][:600])}</p></div>' for c in policy.get("clauses", []))
    if clauses:
        put(f'<h3 class="sec" style="margin-top:1rem">Clauses retrieved for this PO</h3><div class="card">{clauses}</div>')
    elif not found:
        empty("Policy was not checked", "The document could not be read.")


def _pipeline(item: ReviewQueueItem, errors: list[dict]) -> None:
    steps = data.trace(item.job_id)
    if steps:
        names = {"ingest": "Open", "extract": "Read", "enrich": "Check", "score": "Score", "route": "Route", "store": "File",
                 "review": "Queue"}
        put('<div class="card"><h3 class="sec">LangGraph run</h3><div class="steps">' + "".join(
            f'<div class="step {"" if t["ok"] else "bad"}"><b>{esc(names.get(t["node"], t["node"]))}</b>'
            f'<span>{t["ms"] / 1000:.1f} s{"" if t["ok"] else ", failed"}</span></div>' for t in steps) + "</div></div>")
    for e in errors:
        st.error(f"{e.get('node')}: {e.get('message')}")
    _timeline(data.audit(40, item.job_id))


def _correction_form(item: ReviewQueueItem, po: dict, reviewer: str) -> None:
    st.caption("Compare with the document, fix what was misread, then re-check. Every validation runs again and the "
               "change is recorded; correcting never approves.")
    # the revision in the keys makes the form show the new values after a re-check
    rev = f"{item.id}-{abs(hash(item.extracted_json))}"
    header, amounts = {}, {}
    cols = st.columns(2)
    for k, f in enumerate(review_actions.HEADER_FIELDS):
        header[f] = cols[k % 2].text_input(FIELD_LABELS[f], value=(po.get(f) or {}).get("value") or "", key=f"fix-{f}-{rev}")
    cols = st.columns(4)
    for k, f in enumerate(review_actions.AMOUNT_FIELDS):
        v = po.get(f)
        amounts[f] = cols[k].text_input(f.capitalize(), value="" if v is None else f"{v:.2f}", key=f"fix-{f}-{rev}")
    rows = [{k: li.get(k) for k in review_actions.LINE_FIELDS} for li in po.get("line_items", [])]
    df = pd.DataFrame(rows, columns=list(review_actions.LINE_FIELDS))
    for k in ("line_no", "quantity", "unit_price", "line_total"):
        df[k] = pd.to_numeric(df[k])
    for k in ("sku", "description", "unit"):
        df[k] = df[k].astype("object").where(df[k].notna(), None)
    edited = st.data_editor(df, num_rows="dynamic", width="stretch", hide_index=True, key=f"fix-lines-{rev}",
                            column_order=("sku", "description", "quantity", "unit", "unit_price", "line_total"),
                            column_config={
                                "sku": st.column_config.TextColumn("SKU", width=105),
                                "description": st.column_config.TextColumn("Description", width=150),
                                "quantity": st.column_config.NumberColumn("Qty", width=50),
                                "unit": st.column_config.TextColumn("Unit", width=50),
                                "unit_price": st.column_config.NumberColumn("Price", width=70, format="%.2f"),
                                "line_total": st.column_config.NumberColumn("Amount", width=80, format="%.2f")})
    x, y = st.columns(2)
    if x.button("Apply corrections and re-check", key=f"fix-{item.id}", type="primary", width="stretch"):
        lines = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in edited.to_dict("records")]
        try:
            with st.spinner("Running every check again…"):
                out = review_actions.correct(item.id, reviewer, header, amounts, lines)
            left = ", ".join(flag_info(f)[0] for f in out["flags"]) if out["flags"] else "no flags left"
            done(f"Re-checked: {left}" + ("" if out["changes"] else " (nothing was changed)"))
        except review_actions.ReviewError as exc:
            st.error(str(exc))
    if y.button("Read the original file again", key=f"re-{item.id}", width="stretch"):
        try:
            with st.spinner("Reading the document again…"):
                out = review_actions.reprocess(item.id, reviewer)
            done("Read again: " + (", ".join(flag_info(f)[0] for f in out["flags"]) if out["flags"] else "no flags"))
        except review_actions.ReviewError as exc:
            st.error(str(exc))


def _decision(item: ReviewQueueItem, po: dict, vm: dict, flags: list[str], reviewer: str) -> None:
    settings = get_settings()
    with st.container(border=True, key="decision"):
        put('<h3 class="sec">Your decision</h3>')
        if not reviewer:
            put('<span class="hint">Enter your name in the sidebar first.</span>')
        vendors = load_vendors(settings.seed_path)
        keep = f"Keep {vm['canonical_name']}" if vm.get("status") == "matched" else "No record chosen"
        options = [keep] + [
            f"{v.vendor_id} · {v.name}" + (" (blocked)" if v.status == "blocked" else "") for v in vendors]
        choice = st.selectbox("Vendor record", options, key=f"vend-{item.id}")
        learn = False
        if choice != keep:
            learn = st.checkbox("Remember this spelling for next time", key=f"learn-{item.id}")
        supersede = False
        if "po_number_conflict" in flags:
            dup = jl(item.duplicate_json, {})
            st.warning(f"Differs from {dup.get('table')} #{dup.get('id')}: " + "; ".join(dup.get("differences") or []))
            supersede = st.checkbox("This is an amendment: replace the filed PO", key=f"sup-{item.id}")
        note = st.text_area("Reason or sign-off reference", key=f"note-{item.id}", height=80,
                            placeholder="Required to override a flag, or to reject")
        a, b = st.columns(2)
        if a.button("Approve and file", key=f"approve-{item.id}", type="primary", width="stretch", disabled=not po,
                    icon=":material/check:"):
            override = None if choice == keep else choice.split(" · ")[0]
            try:
                po_id = review_actions.approve(item.id, reviewer, note, override, learn, supersede)
                go("queue")
                done(f"{item.po_number} approved and filed as record {po_id}")
            except review_actions.ReviewError as exc:
                st.error(str(exc))
        if b.button("Reject", key=f"reject-{item.id}", width="stretch", icon=":material/close:"):
            try:
                review_actions.reject(item.id, reviewer, note)
                go("queue")
                done(f"{item.po_number or 'Item'} rejected")
            except review_actions.ReviewError as exc:
                st.error(str(exc))


def queue(reviewer: str) -> None:
    items = data.pending()
    if not items:
        title("Review queue")
        empty("Nothing needs a person right now", "Purchase orders with a doubt, an error or a policy issue appear here.")
        return
    wanted = st.query_params.get("item")
    item = next((i for i in items if str(i.id) == wanted), items[0])

    listing, main = st.columns([1.05, 4], gap="medium")
    with listing, st.container(key="queue-list"):
        put(f'<h3 class="sec">Waiting<small>{len(items)}</small></h3>')
        for i in items:
            flags = jl(i.flags_json, [])
            vendor = re.sub(r"([\\`*_\[\]<>#~$:])", r"\\\1", textwrap.shorten(i.vendor_name_raw or "Vendor not read", 27, placeholder="…"))
            number = re.sub(r"([\\`*_\[\]<>#~$:])", r"\\\1", i.po_number or "Unreadable document")
            label = f"**{number}**  \n{vendor}  \n:gray[{money(i.total, i.currency).replace('$', chr(92) + '$')} · {len(flags)} flag{'s' if len(flags) != 1 else ''}]"
            if st.button(label, key=f"pick-{i.id}", type="primary" if i.id == item.id else "secondary", width="stretch"):
                go("queue", i.id)
                st.rerun()
    with main:
        flags, reasons = jl(item.flags_json, []), jl(item.reasons_json, {})
        po, vm, checks = jl(item.extracted_json, {}), jl(item.vendor_match_json, {}), jl(item.price_checks_json, [])
        _header(item, po, vm)
        _checks(flags, po)
        doc_col, work = st.columns([1, 1.12], gap="medium")
        with doc_col:
            _flags(flags, reasons) if flags else st.success("All checks pass after the correction. Ready to approve.")
            _decision(item, po, vm, flags, reviewer)
        with work:
            tabs = st.tabs(["Document", "Values", "JSON", "Items", "Vendor", "Policy", "Correct values", "History"])
            with tabs[0]:
                document.show(item.source_file, item.doc_type, po, flags, checks, str(item.id))
            with tabs[1]:
                _fields(po) if po else empty("Nothing could be read", "Key the values in under Correct values.")
            with tabs[2]:
                _json(po, item.po_number, f"q-{item.id}", {
                    "decision": "needs_review", "flags": flags, "reasons": reasons,
                    "vendor_match": {k: vm.get(k) for k in ("status", "vendor_id", "canonical_name", "vendor_status",
                                                            "score", "reason")},
                    "price_checks": [{k: c.get(k) for k in ("line_no", "matched_sku", "match_method", "unit_price",
                                                            "catalog_unit_price", "deviation_pct", "status")}
                                     for c in checks]})
            with tabs[3]:
                _items(po, checks) if po else empty("Nothing could be read", "Key the values in under Correct values.")
            with tabs[4]:
                _vendor(po, vm) if vm else empty("Vendor was not checked", "The document could not be read.")
            with tabs[5]:
                _policy(jl(item.policy_findings_json, {}))
            with tabs[6]:
                _correction_form(item, po, reviewer)
            with tabs[7]:
                _pipeline(item, jl(item.errors_json, []))


def _json(po: dict, po_number: str | None, key: str, validation: dict | None = None) -> None:
    """The extraction output exactly as stored: the PurchaseOrder schema, per-field confidence and evidence."""
    if not po:
        empty("Nothing could be read", "There is no extracted JSON for this document.")
        return
    put('<h3 class="sec">Extracted purchase order<small>PurchaseOrder schema · app/extraction/schema.py</small></h3>')
    st.json(po, expanded=2)
    st.download_button("Download JSON", json.dumps(po, indent=2), file_name=f"{po_number or 'purchase-order'}.json",
                       mime="application/json", key=f"json-{key}", icon=":material/data_object:")
    if validation:
        with st.expander("Validation result: what the knowledge base and the rules found"):
            st.json(validation, expanded=2)


# ================================================================== filed POs
def filed() -> None:
    title("Filed purchase orders", "Everything the agent booked by itself or a reviewer approved.")
    pos = data.purchase_orders()
    if not pos:
        empty("No purchase orders filed yet", "Send one from the Send a PO page.")
        return
    a, b = st.columns([2, 3])
    text = a.text_input("Search", placeholder="PO number or vendor", label_visibility="collapsed").strip().lower()
    only = b.segmented_control("Show", ["All", "Filed automatically", "Approved by a person"], default="All",
                               label_visibility="collapsed")
    want = {"Filed automatically": "approved", "Approved by a person": "approved_by_review"}.get(only or "")
    shown = [p for p in pos if (not want or p.status == want) and
             (not text or text in f"{p.po_number} {p.vendor_canonical_name} {p.vendor_name_raw}".lower())]
    rows = "".join(
        f'<tr><td><b style="font-weight:600">{esc(p.po_number)}</b><div class="ev">record {p.id}</div></td>'
        f'<td>{esc(p.vendor_canonical_name or p.vendor_name_raw)}'
        + (f'<div class="ev">printed as {esc(p.vendor_name_raw)}</div>'
           if (p.vendor_canonical_name or "").lower() != (p.vendor_name_raw or "").lower() else "")
        + f'</td><td>{esc(p.buyer_department)}</td><td>{esc(p.delivery_date)}</td><td class="n">{money(p.total, p.currency)}</td>'
          f'<td>{decision_chip(p.status)}</td><td>{esc(p.approved_by)}</td><td class="k">{ago(p.created_at)}</td></tr>' for p in shown)
    put('<div class="card"><table class="f"><tr><th>PO</th><th>Vendor</th><th>Department</th><th>Delivery</th><th class="n">Total</th>'
        f'<th>Decision</th><th>By</th><th>Filed</th></tr>{rows}</table></div>')
    if not shown:
        return
    pick = st.selectbox("Details", [p.id for p in shown], format_func=lambda i: next(
        f"{p.po_number} · {p.vendor_canonical_name or p.vendor_name_raw}" for p in shown if p.id == i))
    rec = next(p for p in shown if p.id == pick)
    left, right = st.columns([1, 1.12], gap="medium")
    with left:
        lines = "".join(f'<tr><td>{esc(li.description or li.sku)}<div class="ev">{esc(li.matched_sku or li.sku)}</div></td>'
                        f'<td class="n">{"" if li.quantity is None else format(li.quantity, "g")} {esc(li.unit)}</td><td class="n">{money(li.unit_price, rec.currency)}</td>'
                        f'<td class="n">{"–" if li.price_deviation_pct is None else format(li.price_deviation_pct, "+.1f") + "%"}</td>'
                        f'<td class="n">{money(li.line_total, rec.currency)}</td></tr>' for li in data.line_items(rec.id))
        put('<div class="card"><h3 class="sec">Line items</h3><table class="f"><tr><th>Item</th><th class="n">Quantity</th><th class="n">Price</th>'
            f'<th class="n">vs catalog</th><th class="n">Amount</th></tr>{lines}</table></div>')
        put('<h3 class="sec" style="margin-top:1rem">History</h3>')
        _timeline(data.audit(40, rec.job_id))
    with right:
        doc_tab, json_tab = st.tabs(["Document", "JSON"])
        with doc_tab:
            document.show(rec.source_file, None, jl(rec.extracted_json, {}), [], [], f"po-{rec.id}")
        with json_tab:
            _json(jl(rec.extracted_json, {}), rec.po_number, f"po-{rec.id}")


# ================================================================== knowledge base
def knowledge() -> None:
    settings = get_settings()
    rules = load_rules(settings)
    title("Knowledge base", "What the agent checks every purchase order against. The limits below are read from the policy text.")
    tiles = [("Auto-approval limit", money(rules.auto_approval_limit), rules.clause.get("auto_approval_limit")),
             ("Single-PO limit", money(rules.single_po_hard_limit), rules.clause.get("single_po_hard_limit")),
             ("Price tolerance", f"±{rules.price_tolerance_pct:g}%", rules.clause.get("price_tolerance_pct")),
             ("Payment terms", ", ".join(t.replace("Net ", "") for t in rules.allowed_terms) + " days", rules.clause.get("allowed_terms"))]
    put('<div class="tiles" style="grid-template-columns:repeat(4,1fr)">' + "".join(
        f'<div class="card tile"><span class="l">{esc(a)}</span><div class="v" style="font-size:1.55rem">{esc(b)}</div><span class="s">{esc(c)}</span></div>'
        for a, b, c in tiles) + "</div>")
    vendors = load_vendors(settings.seed_path)
    learned: dict[str, list[str]] = {}
    for a in data.aliases():
        learned.setdefault(a.vendor_id, []).append(a.alias)
    t1, t2, t3 = st.tabs([f"Vendors ({len(vendors)})", "Catalog", "Policy"])
    with t1:
        rows = "".join(
            f'<tr class="{"hot" if v.status == "blocked" else ""}"><td class="k">{esc(v.vendor_id)}</td><td><b style="font-weight:600">{esc(v.name)}</b>'
            f'<div class="ev">{esc(" · ".join(v.aliases))}</div>'
            + "".join(f'{chip("learned: " + a, "info")} ' for a in learned.get(v.vendor_id, []))
            + f'</td><td class="ev">{esc(v.tax_id)}</td><td>{esc(v.payment_terms)}</td>'
              f'<td>{chip("Blocked", "critical") if v.status == "blocked" else chip("Active", "good")}</td></tr>' for v in vendors)
        put(f'<div class="card"><table class="f"><tr><th></th><th>Name and known spellings</th><th>Tax ID</th><th>Terms</th><th>Status</th></tr>{rows}</table></div>')
    with t2:
        st.dataframe(pd.DataFrame([{"SKU": p.sku, "Description": p.description, "Category": p.category, "Unit": p.unit,
                                    "Standard price": p.standard_unit_price} for p in load_catalog(settings.seed_path)]),
                     width="stretch", hide_index=True, height=560,
                     column_config={"Standard price": st.column_config.NumberColumn(format="$%.2f")})
    with t3:
        put('<div class="card">' + "".join(f'<div class="clause"><b>{esc(c.heading)}</b><p>{esc(c.text)}</p></div>'
                                           for c in load_policy_chunks(settings.seed_path)) + "</div>")


# ================================================================== audit
EVENTS = {
    "email_accepted": ("Email accepted", ""), "email_duplicate_skipped": ("Duplicate email skipped", "warn"),
    "email_no_attachment": ("Email without attachment", "warn"), "email_rejected_sender": ("Sender not allowed", "bad"),
    "attachment_rejected": ("Attachment refused", "bad"), "webhook_rejected": ("Forged or invalid webhook refused", "bad"),
    "pipeline_completed": ("Checked by the agent", ""), "review_approved": ("Approved by a person", "good"),
    "review_rejected": ("Rejected by a person", "bad"), "review_corrected": ("Values corrected", "warn"),
    "review_reprocessed": ("Read again", "warn"), "vendor_alias_learned": ("Vendor spelling learned", "good"),
    "job_failed": ("Job failed, will retry", "bad"), "intake_failed": ("Intake failed, released for retry", "bad"),
}


def _timeline(rows) -> None:
    out = []
    for r in rows:
        head, tone = EVENTS.get(r.event, (r.event, ""))
        d = jl(r.details_json, {})
        flags = jl(r.flags_json, [])
        if r.event == "pipeline_completed":
            tone = {"auto_approve": "good", "needs_review": "warn"}.get(r.decision, "")
            head = {"auto_approve": "All checks passed, filed automatically", "needs_review": "Held for review",
                    "duplicate": "Identical PO, skipped"}.get(r.decision, head)
        bits = [d.get("po_number"), d.get("filename"), d.get("sender"), d.get("subject"), d.get("note"), d.get("reason")]
        if r.event == "review_corrected":
            bits.append("changed " + ", ".join(d.get("changes") or {}) if d.get("changes") else "nothing changed")
        if flags:
            bits.append(", ".join(flag_info(f)[0] for f in flags))
        out.append(f'<div class="{tone}"><span>{r.ts:%d %b %H:%M:%S}</span><b style="font-weight:600">{esc(head)}</b> '
                   f'<small>by {esc(r.actor)}{" · " if any(bits) else ""}{esc(" · ".join(str(b) for b in bits if b))}</small></div>')
    put(f'<div class="tl">{"".join(out)}</div>')


def audit() -> None:
    title("Audit log", "Append-only: the database refuses updates and deletes. Times are UTC.")
    rows = data.audit(300)
    if not rows:
        empty("Nothing recorded yet", "Every email, check and decision will appear here.")
        return
    only = st.segmented_control("Show", ["Everything", "People", "Agent", "Refused"], default="Everything",
                                label_visibility="collapsed")
    if only == "People":
        rows = [r for r in rows if r.actor != "agent"]
    elif only == "Agent":
        rows = [r for r in rows if r.actor == "agent"]
    elif only == "Refused":
        rows = [r for r in rows if EVENTS.get(r.event, ("", ""))[1] == "bad"]
    left, right = st.columns([3, 2], gap="medium")
    with left:
        _timeline(rows)
    with right:
        if rows:
            pick = st.selectbox("Full record", [r.id for r in rows], format_func=lambda i: next(
                f"{r.id} · {EVENTS.get(r.event, (r.event,))[0]}" for r in rows if r.id == i))
            st.json(jl(next(r for r in rows if r.id == pick).details_json, {}), expanded=1)

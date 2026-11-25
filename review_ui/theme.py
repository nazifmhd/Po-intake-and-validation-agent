"""Look and vocabulary of the review desk: CSS, small HTML building blocks, and what each flag means to a person.

Every value that can come from a document goes through `esc()` before it reaches HTML.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

# code -> (title, check group, severity, what the reviewer should do)
FLAGS: dict[str, tuple[str, str, str, str]] = {
    "vendor_blocked": ("Blocked vendor", "vendor", "critical", "Reject and report to Compliance"),
    "vendor_unknown": ("Vendor not in the registry", "vendor", "critical", "Map it to a registry record, or reject"),
    "vendor_ambiguous": ("Vendor matches several records", "vendor", "serious", "Choose the right record"),
    "vendor_low_match": ("Vendor match not certain", "vendor", "serious", "Confirm it is really this company"),
    "vendor_not_verified": ("Vendor could not be verified", "vendor", "serious", "Check the vendor by hand"),
    "vendor_tax_id_mismatch": ("Tax ID differs from the registry", "vendor", "critical", "Verify before paying anyone"),
    "vendor_check_failed": ("Vendor check did not run", "vendor", "serious", "Re-check once the service is back"),
    "price_anomaly": ("Price outside tolerance", "items", "serious", "Ask the buyer to justify the price"),
    "unknown_item": ("Item not in the catalog", "items", "serious", "Review specification and price"),
    "unit_mismatch": ("Unit differs from the catalog", "items", "serious", "Check quantity and unit"),
    "sku_mismatch": ("SKU and description disagree", "items", "serious", "Check which product is meant"),
    "catalog_check_failed": ("Catalog check did not run", "items", "serious", "Re-check once the service is back"),
    "over_threshold": ("Above the auto-approval limit", "policy", "serious", "Needs the director's sign-off"),
    "exceeds_po_limit": ("Above the single-PO limit", "policy", "critical", "Needs CFO approval and three quotes"),
    "noncompliant_terms": ("Payment terms not allowed", "policy", "serious", "Needs Finance approval"),
    "non_base_currency": ("Foreign currency", "policy", "serious", "Needs Finance review"),
    "possible_po_split": ("Looks like a split purchase", "policy", "critical", "Review the POs together"),
    "policy_violation": ("Policy reviewer found a violation", "policy", "serious", "Read the cited clause"),
    "policy_rule_missing": ("Policy no longer states a rule", "policy", "critical", "Fix the policy document"),
    "policy_review_unavailable": ("Policy review did not run", "policy", "serious", "Re-check once the model is back"),
    "policy_check_failed": ("Policy check did not run", "policy", "serious", "Re-check once the service is back"),
    "math_mismatch": ("The numbers do not add up", "math", "critical", "Compare with the document and correct"),
    "math_unverified": ("Total could not be verified", "math", "serious", "Check the line amounts"),
    "invalid_amount": ("Zero or negative amounts", "math", "critical", "Check for hidden credits"),
    "low_extraction_confidence": ("Uncertain reading", "reading", "serious", "Compare with the document"),
    "missing_fields": ("Required fields missing", "reading", "serious", "Fill them in, or return the PO"),
    "incomplete_lines": ("Incomplete line items", "reading", "serious", "Fill in quantity and price"),
    "extraction_failed": ("Document could not be read", "reading", "critical", "Key it in, or ask for a new copy"),
    "llm_not_configured": ("No model configured", "reading", "critical", "Set the API key and re-run extraction"),
    "no_text_for_offline_extraction": ("Scan needs a vision model", "reading", "serious", "Set the API key and re-run"),
    "po_number_conflict": ("PO number already used", "document", "critical", "Amendment, misreading or fraud?"),
    "multiple_pos_in_document": ("Several POs in one document", "document", "serious", "Process the others separately"),
    "malformed_csv": ("CSV columns are shifted", "document", "serious", "Ask for a corrected file"),
    "truncated_document": ("Document too long", "document", "serious", "Check the remaining pages"),
    "unread_pages": ("Pages could not be read", "document", "serious", "Read them by hand"),
    "file_missing": ("Original file is missing", "document", "critical", "Ask for a new copy"),
    "unsupported_type": ("Unsupported file type", "document", "critical", "Ask for a PDF, image or CSV"),
    "empty_document": ("Nothing on the pages", "document", "critical", "Ask for a new copy"),
    "llm_error": ("The model could not read it", "reading", "critical", "Re-run extraction, or key it in"),
    "locked_file": ("Password-protected file", "document", "critical", "Ask for an unlocked copy"),
    "corrupt_file": ("Damaged file", "document", "critical", "Ask for a new copy"),
    "empty_file": ("Empty file", "document", "critical", "Ask for a new copy"),
    "storage_error": ("Could not be filed", "document", "critical", "Try approving again"),
    "processing_error": ("Processing failed", "document", "critical", "Re-run extraction"),
}

# the five independent signals of the decision policy, plus the document itself
CHECKS = [
    ("vendor", "Vendor", "approved-vendor registry"), ("items", "Items and prices", "product catalog"),
    ("policy", "Policy", "procurement policy"), ("math", "Arithmetic", "pure Python, no model"),
    ("reading", "Reading confidence", "per-field confidence and evidence"),
    ("document", "Document", "file integrity and duplicates"),
]

FIELD_LABELS = {"po_number": "PO number", "po_date": "PO date", "vendor_name": "Vendor", "vendor_tax_id": "Vendor tax ID",
                "buyer_department": "Department", "currency": "Currency", "payment_terms": "Payment terms",
                "delivery_date": "Delivery date"}

DECISIONS = {  # decision / status -> (label, tone)
    "auto_approve": ("Auto-approved", "good"), "approved": ("Auto-approved", "good"),
    "approved_by_review": ("Approved", "good"), "needs_review": ("Needs review", "warn"),
    "pending_review": ("Needs review", "warn"), "duplicate": ("Duplicate", "mute"), "rejected": ("Rejected", "bad"),
    "superseded": ("Superseded", "mute"), "queued": ("Queued", "mute"), "processing": ("Processing", "info"),
    "failed": ("Failed", "bad"), "refused": ("Refused", "bad"),
}

CSS = """
<style>
:root{--ink:#17212b;--ink2:#5a6673;--ink3:#8b95a1;--line:#e3e7ea;--card:#fff;--plane:#f5f6f4;--accent:#0f5c63;
 --accent-soft:#e3efef;--good:#0a7a1c;--good-soft:#e6f4e8;--warn:#8a5a00;--warn-soft:#fdf1d8;--bad:#b3282a;
 --bad-soft:#fbe7e6;--serious:#a94a1c;--serious-soft:#fdece3;--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace}
[data-testid="stHeader"]{background:transparent}
[data-testid="stMainBlockContainer"]{padding:2.2rem 2.6rem 4rem;max-width:1560px}
[data-testid="stSidebar"]{background:#10262b;border-right:0;min-width:250px;max-width:250px}
[data-testid="stSidebar"] *{color:#d5e0e1}
[data-testid="stSidebar"] .stButton button{justify-content:flex-start;background:transparent;border:0;padding:.5rem .7rem;
 font-weight:500;border-radius:8px;color:#b9c9cb}
[data-testid="stSidebar"] .stButton button:hover{background:#1b383e}
[data-testid="stSidebar"] .stButton button[kind="primary"]{background:#1f454c;color:#fff}
[data-testid="stSidebar"] .stButton button[kind="primary"] *{color:#fff;font-weight:600}
[data-testid="stSidebar"] .stButton button>div{justify-content:flex-start;width:100%}
[data-testid="stSidebar"] input{background:#1b383e!important;color:#fff!important;-webkit-text-fill-color:#fff}
[data-testid="stSidebar"] input::placeholder{color:#8fa9ac!important;-webkit-text-fill-color:#8fa9ac;opacity:1}
[data-testid="stSidebar"] [data-testid="stTextInputRootElement"]{border-color:#2c5058!important;background:#1b383e}
[data-testid="stSidebar"] .stTextInput{margin-top:1rem}
[data-testid="stSidebar"] [data-baseweb="input"],[data-testid="stSidebar"] [data-baseweb="base-input"]{background:#1b383e;border-color:#2c5058}
.brand{display:flex;gap:.65rem;align-items:center;margin:-.6rem 0 1.2rem}
.brand .mark{width:34px;height:40px;border-radius:4px 10px 4px 4px;background:#d9efe9;display:flex;flex-direction:column;
 gap:4px;padding:9px 7px;box-sizing:border-box}.brand .mark i{display:block;height:3px;border-radius:2px;background:#0f5c63}
.brand .mark i:nth-child(2){width:65%}.brand .mark i:nth-child(3){width:82%;background:#e08a1e}
.brand b{display:block;font-size:1.02rem;line-height:1.15;color:#fff!important}.brand span{font-size:.76rem;color:#8fa9ac!important}
.side-foot{font-size:.76rem;line-height:1.55;color:#8fa9ac!important;border-top:1px solid #234148;padding-top:.8rem;margin-top:1.2rem}
.side-foot b{color:#d5e0e1!important;font-weight:600}
h1.page{font-size:1.75rem;font-weight:650;letter-spacing:-.02em;margin:0 0 .15rem;padding:0;color:var(--ink)}
p.lede{color:var(--ink2);margin:0 0 1.2rem;font-size:.95rem;max-width:75ch}
h3.sec{font-size:1rem;font-weight:650;margin:0 0 .7rem;padding:0;color:var(--ink)}h3.sec small{font-weight:400;color:var(--ink3);margin-left:.5rem;font-size:.82rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:1.05rem 1.2rem;overflow-x:auto}
.tiles{display:grid;grid-template-columns:1.5fr 1fr 1fr 1fr;gap:14px;margin-bottom:16px}
.tile .l{color:var(--ink2);font-size:.86rem}.tile .v{font-size:2rem;font-weight:600;letter-spacing:-.02em;line-height:1.2;color:var(--ink)}
.tile .s{color:var(--ink3);font-size:.82rem}.tile.lead{background:#10262b;border-color:#10262b}
.tile.lead .l{color:#a9c1c4}.tile.lead .v{color:#fff;font-size:2.5rem}.tile.lead .s{color:#a9c1c4}
.seg{display:flex;gap:2px;height:10px;border-radius:5px;overflow:hidden;margin:.55rem 0 .5rem}
.seg i{display:block;min-width:4px}.leg{display:flex;gap:1rem;font-size:.8rem;color:#a9c1c4;flex-wrap:wrap}
.leg span::before{content:"";display:inline-block;width:9px;height:9px;border-radius:3px;background:var(--c);margin-right:.4rem}
.chip{display:inline-flex;align-items:center;gap:.3rem;border-radius:999px;padding:.13rem .6rem;font-size:.78rem;font-weight:600;
 white-space:nowrap;line-height:1.5}
.chip.good{background:var(--good-soft);color:var(--good)}.chip.warn{background:var(--warn-soft);color:var(--warn)}
.chip.bad,.chip.critical{background:var(--bad-soft);color:var(--bad)}.chip.serious{background:var(--serious-soft);color:var(--serious)}
.chip.mute{background:#edf0f1;color:var(--ink2)}.chip.info{background:var(--accent-soft);color:var(--accent)}
.row{display:grid;grid-template-columns:72px minmax(0,1.6fr) minmax(0,1.2fr) 118px 132px;gap:.8rem;align-items:center;
 padding:.62rem 0;border-top:1px solid var(--line);font-size:.9rem}.row:first-of-type{border-top:0}
.row .t{color:var(--ink3);font-size:.8rem}.row b{font-weight:600}.row .sub{color:var(--ink3);font-size:.8rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:block}
.num{font-variant-numeric:tabular-nums;text-align:right}.ell{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bars{display:grid;gap:.55rem}.bar{display:grid;grid-template-columns:minmax(0,1fr) 34px;gap:.1rem .6rem;font-size:.86rem;align-items:center}
.bar .track{grid-column:1/-1;height:8px;background:#edf1f1;border-radius:4px}.bar .track i{display:block;height:100%;background:var(--accent);border-radius:4px}
.bar:hover .track i{background:#0b474c}
.pohead{display:flex;justify-content:space-between;gap:1.5rem;align-items:flex-start;margin-bottom:1rem}
.pohead .k{color:var(--ink3);font-size:.85rem}.pohead h2{font-size:1.7rem;font-weight:650;letter-spacing:-.02em;margin:.05rem 0 .3rem;padding:0;color:var(--ink)}
.pohead .meta{display:flex;gap:.1rem 1.1rem;color:var(--ink2);font-size:.88rem;flex-wrap:wrap}
.pohead .amt{text-align:right;white-space:nowrap}.pohead .amt b{display:block;font-size:1.9rem;font-weight:600;letter-spacing:-.02em;line-height:1.2;color:var(--ink)}
.stamp{display:inline-block;border:2.5px solid var(--serious);color:var(--serious);border-radius:6px;padding:.08rem .6rem;font-weight:700;
 font-size:.9rem;transform:rotate(-4deg);opacity:.9;margin-top:.35rem}
.stamp.good{border-color:var(--good);color:var(--good)}.stamp.bad{border-color:var(--bad);color:var(--bad)}
.flag{display:grid;grid-template-columns:auto 1fr;gap:.2rem .8rem;border:1px solid var(--line);border-left:4px solid var(--serious);
 border-radius:10px;background:var(--card);padding:.75rem .95rem;margin-bottom:.55rem}
.flag.critical{border-left-color:var(--bad)}.flag .ic{font-size:1.1rem;line-height:1.45;color:var(--serious);font-weight:700}.flag.critical .ic{color:var(--bad)}
.flag b{font-weight:650;color:var(--ink)}.flag code{font-family:var(--mono);font-size:.74rem;color:var(--ink3);background:none;padding:0;margin-left:.45rem}
.flag p{grid-column:2;margin:0;color:var(--ink2);font-size:.9rem}.flag .do{grid-column:2;font-size:.84rem;color:var(--ink)}
.cite{display:inline-block;font-size:.76rem;background:var(--accent-soft);color:var(--accent);border-radius:5px;padding:.05rem .45rem;margin-left:.2rem}
.checks{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;margin-bottom:1rem}
.check{border:1px solid var(--line);border-radius:10px;background:var(--card);padding:.6rem .7rem;font-size:.85rem}
.check b{display:block;font-weight:600;color:var(--ink)}.check span{color:var(--ink3);font-size:.75rem;line-height:1.3;display:block}
.check em{font-style:normal;font-weight:650;font-size:.82rem}.check.ok em{color:var(--good)}.check.no{background:#fffaf7;border-color:#f3cdb9}
.check.no em{color:var(--bad)}.check.na em{color:var(--ink3)}
.steps{display:flex;align-items:stretch;gap:0;margin:.1rem 0 .3rem;flex-wrap:wrap}
.step{flex:1;min-width:58px;padding:.1rem .3rem .1rem 0;font-size:.83rem;position:relative}
.step::before{content:"";display:block;height:4px;border-radius:2px;background:var(--accent);margin:0 4px .4rem 0}
.step.bad::before{background:var(--bad)}.step b{font-weight:600;display:block}.step span{color:var(--ink3);font-size:.78rem}
table.f{width:100%;border-collapse:collapse;font-size:.88rem}table.f th{font-weight:500;color:var(--ink3);text-align:left;font-size:.78rem;padding:.3rem .5rem;border:0;border-bottom:1px solid var(--line)}
table.f td{padding:.48rem .5rem;border:0;border-bottom:1px solid var(--line);vertical-align:top;color:var(--ink)}table.f tr:last-child td{border-bottom:0}
table.f td.n,table.f th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}table.f td.k{color:var(--ink2);white-space:nowrap}
table.f tr.hot td{background:#fff7f1}table.f .ev{font-family:var(--mono);font-size:.76rem;color:var(--ink3);word-break:break-word}
.conf{display:inline-flex;align-items:center;gap:.45rem;font-variant-numeric:tabular-nums;font-size:.8rem;color:var(--ink2);white-space:nowrap}
.conf i{display:inline-block;width:46px;height:6px;border-radius:3px;background:#e8ecee;overflow:hidden}.conf i u{display:block;height:100%;background:var(--accent)}
.conf.low i u{background:var(--bad)}.conf.low{color:var(--bad);font-weight:600}
.match{display:grid;grid-template-columns:1fr 40px 1fr;gap:.6rem;align-items:center;margin-bottom:1rem}
.match .side{border:1px solid var(--line);border-radius:10px;padding:.7rem .9rem;background:var(--card)}.match .side>span:first-child{color:var(--ink3);font-size:.78rem;display:block}
.match .side b{font-weight:600}.match .arrow{text-align:center;font-size:1.2rem;color:var(--ink3)}
.gauge{margin:.35rem 0 .8rem;font-size:.86rem}.gauge .top{display:flex;justify-content:space-between;color:var(--ink2)}.gauge .top b{color:var(--ink);font-variant-numeric:tabular-nums}
.gauge .tr{position:relative;height:8px;border-radius:4px;background:#e8ecee;margin-top:.3rem}.gauge .tr i{display:block;height:100%;border-radius:4px;background:var(--accent)}
.gauge.no .tr i{background:#d9794b}.gauge .tr u{position:absolute;top:-4px;width:2px;height:16px;background:var(--ink)}
.clause{border-top:1px solid var(--line);padding:.65rem 0}.clause:first-child{border-top:0}.clause b{font-weight:600}.clause p{margin:.2rem 0 0;color:var(--ink2);font-size:.86rem;line-height:1.5;max-width:75ch}
.empty{text-align:center;padding:2.4rem 1rem;color:var(--ink2)}.empty b{display:block;color:var(--ink);font-size:1.05rem;margin-bottom:.2rem}
.tl{border-left:2px solid var(--line);margin-left:.4rem;padding-left:1.1rem}.tl div{position:relative;padding:.35rem 0;font-size:.89rem}
.tl div::before{content:"";position:absolute;left:-1.5rem;top:.75rem;width:9px;height:9px;border-radius:50%;background:var(--accent);border:2px solid var(--plane)}
.tl div.bad::before{background:var(--bad)}.tl div.good::before{background:var(--good)}.tl div.warn::before{background:#e08a1e}
.tl span{color:var(--ink3);font-size:.8rem;margin-right:.5rem;font-variant-numeric:tabular-nums}.tl small{color:var(--ink2)}
.stButton button,.stFormSubmitButton button,.stDownloadButton button{border-radius:8px;font-weight:550}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:12px}
.st-key-queue-list .stButton button{justify-content:flex-start;text-align:left;border:1px solid var(--line);background:var(--card);padding:.65rem .8rem;line-height:1.4}
.st-key-queue-list .stButton button>div{justify-content:flex-start;width:100%}.st-key-queue-list .stButton button p{text-align:left;width:100%}
.st-key-queue-list .stButton button[kind="primary"]{border:2px solid var(--accent);background:var(--accent-soft)}
.st-key-queue-list .stButton button[kind="primary"] *{color:var(--ink)}
.st-key-decision,.st-key-box-demo,.st-key-box-own{background:var(--card);border-radius:12px}
.hint{display:inline-block;background:var(--accent-soft);color:var(--accent);border-radius:7px;padding:.3rem .7rem;font-size:.86rem}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.35rem}[data-testid="stSidebarHeader"]{height:1.2rem;padding:0}
.step span{font-variant-numeric:tabular-nums}[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:1.1rem}
@media(max-width:1100px){.tiles{grid-template-columns:1fr 1fr}.checks{grid-template-columns:repeat(3,1fr)}.row{grid-template-columns:60px 1fr 110px}.row>:nth-child(3),.row>:nth-child(4){display:none}}
@media(max-width:640px){.tiles,.checks{grid-template-columns:1fr}.pohead{flex-direction:column}.pohead .amt{text-align:left}.match{grid-template-columns:1fr}[data-testid="stMainBlockContainer"]{padding:1.5rem 1rem}}
</style>
"""


def esc(value) -> str:
    return html.escape(" ".join(str(value if value is not None else "").split()))


def put(markup: str) -> None:
    st.html(markup)


def title(text: str, lede: str = "") -> None:
    put(f'<h1 class="page">{esc(text)}</h1>' + (f'<p class="lede">{esc(lede)}</p>' if lede else ""))


def money(value: Optional[float], currency: Optional[str] = "USD") -> str:
    if value is None:
        return "no total"
    return f"${value:,.2f}" if (currency or "USD") == "USD" else f"{currency} {value:,.2f}"


def chip(text: str, tone: str = "mute") -> str:
    icon = {"good": "✓", "warn": "!", "bad": "✕", "critical": "✕", "serious": "!", "info": "⟳"}.get(tone, "")
    return f'<span class="chip {tone}">{icon} {esc(text)}</span>'


def decision_chip(status: Optional[str]) -> str:
    label, tone = DECISIONS.get(status or "", (status or "unknown", "mute"))
    return chip(label, tone)


def flag_info(code: str) -> tuple[str, str, str, str]:
    return FLAGS.get(code, (code.replace("_", " ").capitalize(), "document", "serious", "Look at the details below"))


def ago(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    s = max(0, int((datetime.now(timezone.utc) - dt).total_seconds()))
    if s < 10:
        return "just now"
    for size, unit in ((86400, "d"), (3600, "h"), (60, "min")):
        if s >= size:
            return f"{s // size} {unit} ago"
    return f"{s} s ago"


def conf_meter(value: Optional[float], threshold: float) -> str:
    if value is None:
        return ""
    low = " low" if value < threshold else ""
    return f'<span class="conf{low}"><i><u style="width:{value * 100:.0f}%"></u></i>{value:.2f}</span>'


def gauge(label: str, value: Optional[float], need: float, top: float) -> str:
    if value is None:
        return f'<div class="gauge no"><div class="top"><span>{esc(label)}</span><b>not available</b></div></div>'
    fmt = "{:.0f}" if top > 1 else "{:.2f}"
    ok = "" if value >= need else " no"
    return (f'<div class="gauge{ok}"><div class="top"><span>{esc(label)}</span><b>{fmt.format(value)} '
            f'<span style="font-weight:400;color:var(--ink3)">needs {fmt.format(need)}</span></b></div>'
            f'<div class="tr"><i style="width:{min(value / top, 1) * 100:.0f}%"></i>'
            f'<u style="left:{need / top * 100:.0f}%"></u></div></div>')


def empty(head: str, body: str) -> None:
    put(f'<div class="card empty"><b>{esc(head)}</b>{esc(body)}</div>')

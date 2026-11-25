"""The numbers the policy is enforced with are read out of the policy text itself.

Policy owners edit data/seed/policy.md; the limits the agent enforces follow. `.env` values act as caps:
the stricter of the two always wins, so an edit on one side can tighten the other but never loosen it.
A rule that can no longer be found in the text is reported in `missing` and sends every PO to review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import Settings
from app.rag.store import load_policy_chunks

_AMOUNT = r"([\d,]+(?:\.\d+)?)"
_PATTERNS = {
    "auto_approval_limit": rf"grand total (?:of|above) USD {_AMOUNT}",
    "single_po_hard_limit": rf"No single PO may exceed USD {_AMOUNT}",
    "allowed_terms": r"Allowed payment terms are ([^.]+)\.",
    "price_tolerance_pct": r"more than (\d+(?:\.\d+)?)\s?% above",
    "po_split_window_days": r"within (\d+) days",
    "base_currency": r"POs are issued in ([A-Z]{3})",
}


@dataclass
class PolicyRules:
    auto_approval_limit: float
    single_po_hard_limit: float
    allowed_terms: list[str]
    price_tolerance_pct: float
    po_split_window_days: int
    base_currency: str
    clause: dict[str, str] = field(default_factory=dict)    # rule -> heading of the clause it was read from
    missing: list[str] = field(default_factory=list)        # rules not found anywhere in the policy text
    not_retrieved: list[str] = field(default_factory=list)  # found, but retrieval did not surface the clause


def _find(rule: str, clauses: list[dict]):
    for c in clauses:
        found = re.findall(_PATTERNS[rule], " ".join(c["text"].split()), re.I)
        if found:
            return found, c["heading"]
    return None, None


def rules_from_clauses(retrieved: list[dict] | None, settings: Settings) -> PolicyRules:
    """Retrieved clauses first; the full policy text backs them up, so a retrieval miss can't drop a rule."""
    full = [{"heading": c.heading, "text": c.text} for c in load_policy_chunks(settings.seed_path)]
    raw: dict[str, list[str]] = {}
    clause: dict[str, str] = {}
    missing, not_retrieved = [], []
    for rule in _PATTERNS:
        found, heading = _find(rule, retrieved or [])
        if not found:
            found, heading = _find(rule, full)
            if found and retrieved is not None:
                not_retrieved.append(rule)
        if found:
            raw[rule], clause[rule] = found, heading
        else:
            missing.append(rule)

    def number(rule: str, cap: float) -> float:
        return min([*(float(v.replace(",", "")) for v in raw.get(rule, [])), cap])

    terms = settings.allowed_terms_list
    if "allowed_terms" in raw:
        in_policy = {f"net {d}" for d in re.findall(r"net\s*(\d+)", raw["allowed_terms"][0], re.I)}
        terms = [t for t in terms if t.lower() in in_policy]
    return PolicyRules(
        auto_approval_limit=number("auto_approval_limit", settings.auto_approval_limit),
        single_po_hard_limit=number("single_po_hard_limit", settings.single_po_hard_limit),
        allowed_terms=terms,
        price_tolerance_pct=number("price_tolerance_pct", settings.price_tolerance_pct),
        # the wider window is the stricter one
        po_split_window_days=max([*(int(v) for v in raw.get("po_split_window_days", [])),
                                  settings.po_split_window_days]),
        base_currency=raw["base_currency"][0].upper() if "base_currency" in raw else settings.base_currency,
        clause=clause, missing=missing, not_retrieved=not_retrieved,
    )


def load_rules(settings: Settings) -> PolicyRules:
    return rules_from_clauses(None, settings)

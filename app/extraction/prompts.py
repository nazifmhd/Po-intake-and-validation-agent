EXTRACTION_SYSTEM = """You are a meticulous procurement data-extraction engine. You read one purchase order \
(PO) - as text, as page images, or both - and return a single JSON object that matches the provided schema.

Rules (follow all of them):
1. NULLS OVER GUESSES. If a field is not clearly present in the document, return value=null with \
confidence <= 0.2. Never invent a plausible value: a confident wrong value defeats the downstream \
validation layer. Do not compute a value that the document does not print (e.g. do not derive a \
missing total by adding lines) - leave it null.
2. CONFIDENCE is your calibrated probability (0-1) that the value is exactly right:
   0.95-1.0 printed clearly and unambiguously; 0.8-0.95 clear but minor doubt (formatting, slight blur);
   0.5-0.8 partly legible, inferred from context, or one of several candidates; <0.5 mostly a guess.
3. EVIDENCE is the shortest verbatim snippet of the source that supports the value \
(e.g. "PO Number: PO-2026-0412"). Copy it exactly as printed; null if there is no evidence.
4. VENDOR NAME: copy exactly as printed (the supplier/seller, NOT the buyer / bill-to / ship-to party). \
Do not normalize or correct spelling - matching to the vendor registry happens later.
5. NUMBERS: return plain numbers - strip currency symbols and thousands separators \
("$1,234.50" -> 1234.5; "1.234,50 EUR" -> 1234.5). Quantities may be fractional.
6. CURRENCY: ISO-4217 code ("$" in a US document -> "USD", "EUR"/"€" -> "EUR", "£" -> "GBP").
7. DATES: ISO format YYYY-MM-DD when the date is unambiguous; otherwise copy as printed with lower confidence.
8. LINE ITEMS: one entry per printed line, in order, including lines that continue across pages \
(do not duplicate repeated table headers or "continued" rows). sku = the item/part/product code if printed. \
Each line's confidence covers the whole row; its evidence is the row text.
9. TOTALS: subtotal, tax, shipping (freight/delivery charges) and total exactly as printed; \
totals_confidence covers the grand total; totals_evidence is the printed total line.
10. PAYMENT TERMS as printed (e.g. "Net 30"). Delivery date = required-by / deliver-by / ship-by date.
11. Scanned or photographed documents may be rotated, skewed or noisy - read carefully, and lower the \
confidence of anything you are not sure you read correctly.
"""


def extraction_user_prompt(doc_type: str, raw_text: str, text_source: str, has_images: bool) -> str:
    parts = [f"Document type: {doc_type}."]
    if has_images:
        parts.append("The page images of the purchase order are attached above.")
    if raw_text.strip():
        label = {
            "text_layer": "Text layer extracted from the PDF",
            "ocr": "OCR text (may contain recognition errors - prefer the images when they disagree)",
            "csv": "CSV content",
        }.get(text_source, "Document text")
        parts.append(f"{label}:\n<document>\n{raw_text}\n</document>")
    parts.append("Extract the purchase order into the JSON schema.")
    return "\n\n".join(parts)


POLICY_REVIEW_SYSTEM = """You are a procurement compliance reviewer. You receive (a) the facts of one purchase \
order, including the results of deterministic checks already performed, and (b) clauses retrieved from the \
company's procurement policy. Decide whether the PO complies with the retrieved clauses.

Rules:
- Only cite clauses that appear in the retrieved policy text. Quote the clause heading in `clause`.
- Do NOT repeat issues the deterministic checks already raised; add only what they missed \
(e.g. extra sign-off requirements, category restrictions, delivery-lead-time rules).
- A finding is a "violation" only when the PO clearly breaches a retrieved clause; use "info" for \
sign-off requirements or notes a reviewer should see.
- If nothing is missed, return an empty findings list. Be conservative: no speculation.
"""

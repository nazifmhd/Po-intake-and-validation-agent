"""Generate the demo attachments, extra labelled eval POs, edge-case files, and ground-truth labels.

    python scripts/make_samples.py [--out data/samples]

Demo files (the 3-file recording):
  01_clean_northwind.pdf        clean text PDF, known vendor, catalog prices, $4,200     -> auto_approve
  02_scanned_globex.jpg         photographed/scanned PO, laptops +28% vs catalog, $13,500 -> price_anomaly, over_threshold
  03_unknown_vendor.csv         CSV from an unregistered vendor, delivery date blank     -> vendor_unknown, missing_fields
Eval extras:
  04_acme_alias_twocol.pdf      vendor printed as an alias, two-column header, lines split across pages -> auto_approve
  05_blocked_vendor.pdf         vendor is on the registry but BLOCKED                    -> vendor_blocked
  06_math_error.pdf             printed total doesn't add up                             -> math_mismatch
  07_bad_terms.pdf              Net 90 payment terms                                     -> noncompliant_terms
  08_csv_batch.csv              two POs in one CSV (batch split at intake)               -> auto_approve x2
Edge cases (error handling):
  edge_corrupt.pdf, edge_locked.pdf (password-protected), edge_unsupported.zip
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import zipfile
from pathlib import Path

BUYER = ["Northbridge Manufacturing Ltd", "400 Harbor Way, Suite 12", "Portland, OR 97209, USA",
         "procurement@northbridge.example"]

SAMPLES = [
    {
        "id": "01_clean_northwind", "kind": "pdf", "po_number": "PO-2026-0412", "po_date": "2026-10-20",
        "vendor": "Northwind Office Products LLC", "vendor_addr": ["88 Commerce Blvd", "Seattle, WA 98101"],
        "tax_id": "US-23-4567890", "dept": "Operations", "currency": "USD", "terms": "Net 30",
        "delivery": "2026-11-05",
        "lines": [("OFF-CHR-ERG", "Ergonomic mesh office chair with lumbar support", 10, "EA", 285.00),
                  ("OFF-LMP-LED", "LED desk lamp with dimmer", 20, "EA", 42.50)],
        "tax": 296.00, "tax_label": "Sales Tax (8%)", "shipping": 204.00,
        "expected": {"decision": "auto_approve", "flags": []},
    },
    {
        "id": "02_scanned_globex", "kind": "scan", "po_number": "PO-2026-0419", "po_date": "2026-10-22",
        "vendor": "GLOBEX TECH SOLUTIONS INC", "vendor_addr": ["1200 Innovation Pkwy", "Austin, TX 78701"],
        "tax_id": "US-34-5678901", "dept": "IT", "currency": "USD", "terms": "Net 30", "delivery": "2026-11-12",
        "lines": [("IT-LAP-14P", "14-inch business laptop 16GB RAM 512GB SSD", 5, "EA", 2368.00),
                  ("IT-DCK-USC", "USB-C docking station dual display", 5, "EA", 245.00)],
        "tax": 0.00, "tax_label": "Tax (exempt)", "shipping": 435.00,
        "expected": {"decision": "needs_review", "flags": ["price_anomaly", "over_threshold"]},
    },
    {
        "id": "03_unknown_vendor", "kind": "csv", "po_number": "PO-2026-0423", "po_date": "2026-10-24",
        "vendor": "Pinnacle Promo Goods Ltd", "tax_id": "", "dept": "Marketing", "currency": "USD",
        "terms": "Net 30", "delivery": "",
        "lines": [("OFF-PPR-A4", "Copy paper A4 80gsm box of 5 reams", 40, "BOX", 38.50),
                  ("OFF-TNR-BLK", "Black toner cartridge high yield", 6, "EA", 89.00)],
        "tax": 165.92, "shipping": 60.00,
        "expected": {"decision": "needs_review", "flags": ["vendor_unknown", "missing_fields"]},
    },
    {
        "id": "04_acme_alias_twocol", "kind": "pdf", "two_column": True, "page_break_after": 6,
        "po_number": "PO-2026-0398", "po_date": "2026-10-18", "vendor": "ACME Corporation (Pvt) Ltd",
        "vendor_addr": ["5 Foundry Road", "Cleveland, OH 44113"], "tax_id": "US-12-3456789",
        "dept": "Plant Maintenance", "currency": "USD", "terms": "Net 30", "delivery": "2026-11-01",
        "lines": [("SAF-GLV-NIT", "Nitrile gloves box of 100", 40, "BOX", 14.20),
                  ("SAF-HLM-HRD", "Hard hat type II with ratchet", 25, "EA", 32.00),
                  ("SAF-VST-HV", "High-visibility safety vest class 2", 60, "EA", 11.50),
                  ("SAF-GOG-AF", "Anti-fog safety goggles", 50, "EA", 9.75),
                  ("TLS-DRL-18V", "18V cordless drill driver kit", 6, "EA", 165.00),
                  ("TLS-SET-SKT", "Socket set 94-piece metric/SAE", 4, "SET", 119.00),
                  ("ELE-LED-T8", "LED T8 tube 4ft 18W", 100, "EA", 8.90),
                  ("ELE-BRK-20A", "20A single-pole circuit breaker", 30, "EA", 12.40),
                  ("PKG-TAP-48", "Packing tape 48mm x 100m pack of 6", 20, "PK", 15.60),
                  ("JAN-CLN-MP", "Multi-purpose cleaner concentrate 5L", 10, "EA", 22.00)],
        "tax": 406.39, "tax_label": "Sales Tax (7%)", "shipping": 150.00,
        "expected": {"decision": "auto_approve", "flags": []},
    },
    {
        "id": "05_blocked_vendor", "kind": "pdf", "po_number": "PO-2026-0427", "po_date": "2026-10-25",
        "vendor": "Shadow Supply Traders", "vendor_addr": ["PO Box 9921", "Wilmington, DE 19801"],
        "tax_id": "US-55-6677889", "dept": "Warehouse", "currency": "USD", "terms": "Net 30",
        "delivery": "2026-11-08",
        "lines": [("PKG-BOX-M", "Corrugated shipping box medium 18x14x12 bundle of 25", 20, "BDL", 36.00),
                  ("PKG-PLT-WRP", "Pallet stretch wrap 18in x 1500ft", 30, "RL", 24.00)],
        "tax": 0.00, "tax_label": "Tax", "shipping": 80.00,
        "expected": {"decision": "needs_review", "flags": ["vendor_blocked"]},
    },
    {
        "id": "06_math_error", "kind": "pdf", "po_number": "PO-2026-0430", "po_date": "2026-10-26",
        "vendor": "Harbor Packaging Solutions", "vendor_addr": ["17 Dock Street", "Tacoma, WA 98402"],
        "tax_id": "US-67-8901234", "dept": "Warehouse", "currency": "USD", "terms": "Net 30",
        "delivery": "2026-11-10",
        "lines": [("PKG-BOX-M", "Corrugated shipping box medium 18x14x12 bundle of 25", 40, "BDL", 36.00),
                  ("PKG-TAP-48", "Packing tape 48mm x 100m pack of 6", 25, "PK", 15.60)],
        "tax": 146.40, "tax_label": "Sales Tax (8%)", "shipping": 45.00, "total_override": 2121.40,
        "expected": {"decision": "needs_review", "flags": ["math_mismatch"]},
    },
    {
        "id": "07_bad_terms", "kind": "pdf", "po_number": "PO-2026-0433", "po_date": "2026-10-27",
        "vendor": "Umbra Safety Equipment Co.", "vendor_addr": ["310 Shield Ave", "Denver, CO 80202"],
        "tax_id": "US-45-6789012", "dept": "Plant Maintenance", "currency": "USD", "terms": "Net 90",
        "delivery": "2026-11-15",
        "lines": [("SAF-GLV-NIT", "Nitrile gloves box of 100", 100, "BOX", 14.20),
                  ("SAF-VST-HV", "High-visibility safety vest class 2", 80, "EA", 11.50)],
        "tax": 187.20, "tax_label": "Sales Tax (8%)", "shipping": 0.00,
        "expected": {"decision": "needs_review", "flags": ["noncompliant_terms"]},
    },
]

BATCH = {
    "id": "08_csv_batch",
    "pos": [
        {"po_number": "PO-2026-0450", "po_date": "2026-10-27", "vendor": "Blue Yonder Paper Co.",
         "tax_id": "US-11-2233445", "dept": "Administration", "currency": "USD", "terms": "Net 30",
         "delivery": "2026-11-03",
         "lines": [("OFF-PPR-LTR", "Copy paper US Letter 20lb case of 10 reams", 30, "CS", 54.00),
                   ("OFF-TNR-BLK", "Black toner cartridge high yield", 10, "EA", 89.00)],
         "tax": 200.80, "shipping": 0.00, "expected": {"decision": "auto_approve", "flags": []}},
        {"po_number": "PO-2026-0451", "po_date": "2026-10-27", "vendor": "Litware IT Services",
         "tax_id": "US-22-3344556", "dept": "IT", "currency": "USD", "terms": "Net 30", "delivery": "2026-11-04",
         "lines": [("IT-KBM-WLS", "Wireless keyboard and mouse combo", 25, "EA", 59.00),
                   ("IT-HDS-ANC", "Noise-cancelling USB headset", 15, "EA", 129.00)],
         "tax": 272.80, "shipping": 25.00, "expected": {"decision": "auto_approve", "flags": []}},
    ],
}


def totals(spec: dict) -> tuple[float, float]:
    subtotal = round(sum(q * p for _, _, q, _, p in spec["lines"]), 2)
    total = spec.get("total_override", round(subtotal + spec["tax"] + spec["shipping"], 2))
    return subtotal, total


def money(v: float) -> str:
    return f"${v:,.2f}"


# --------------------------------------------------------------------------- PDF rendering
def render_po_pdf(spec: dict, path: Path) -> None:
    import pymupdf as fitz

    doc = fitz.open()
    W, H = 595, 842
    font = "helv"
    bold = "hebo"

    def text(page, x, y, s, size=9.5, f=font, right=False):
        if right:
            x -= fitz.get_text_length(s, fontname=f, fontsize=size)
        page.insert_text((x, y), s, fontsize=size, fontname=f)

    cols = {"line": 50, "sku": 74, "desc": 146, "qty": 392, "unit": 422, "price": 490, "amount": 548}

    def table_header(page, y):
        page.draw_rect(fitz.Rect(45, y - 12, 550, y + 5), color=(0.2, 0.2, 0.2), fill=(0.9, 0.9, 0.92))
        for key, label in (("line", "Line"), ("sku", "SKU"), ("desc", "Description"), ("qty", "Qty"),
                           ("unit", "Unit")):
            text(page, cols[key], y, label, 9, bold)
        text(page, cols["price"], y, "Unit Price", 9, bold, right=True)
        text(page, cols["amount"], y, "Amount", 9, bold, right=True)
        return y + 18

    def new_page(first: bool):
        page = doc.new_page(width=W, height=H)
        text(page, 50, 60, "PURCHASE ORDER", 20, bold)
        text(page, 545, 52, BUYER[0], 10, bold, right=True)
        for i, ln in enumerate(BUYER[1:]):
            text(page, 545, 66 + i * 12, ln, 8.5, right=True)
        page.draw_line((45, 105), (550, 105), color=(0.3, 0.3, 0.3), width=1)
        return page

    page = new_page(True)
    y = 128
    hdr_left = [("PO Number", spec["po_number"]), ("PO Date", spec["po_date"]), ("Department", spec["dept"]),
                ("Currency", spec["currency"]), ("Payment Terms", spec["terms"]),
                ("Delivery Date", spec["delivery"])]
    hdr_vendor = [("Vendor", spec["vendor"]), ("Tax ID", spec["tax_id"])] + \
                 [("", a) for a in spec.get("vendor_addr", [])]
    if spec.get("two_column"):
        for i in range(max(len(hdr_left), len(hdr_vendor))):
            if i < len(hdr_left):
                text(page, 50, y + i * 15, f"{hdr_left[i][0]}: {hdr_left[i][1]}")
            if i < len(hdr_vendor):
                k, v = hdr_vendor[i]
                text(page, 320, y + i * 15, f"{k}: {v}" if k else v)
        y += max(len(hdr_left), len(hdr_vendor)) * 15 + 12
    else:
        for k, v in hdr_left:
            text(page, 50, y, f"{k}: {v}")
            y += 15
        y += 6
        text(page, 50, y, "Supplier details", 10, bold)
        y += 15
        for k, v in hdr_vendor:
            text(page, 50, y, f"{k}: {v}" if k else v)
            y += 15
        y += 4
    text(page, 50, y, "Ship To: Northbridge Manufacturing Ltd - Receiving Dock B, 400 Harbor Way, Portland, OR", 8.5)
    y += 24

    y = table_header(page, y)
    brk = spec.get("page_break_after")
    for i, (sku, desc, qty, unit, price) in enumerate(spec["lines"], start=1):
        if brk and i == brk + 1:
            text(page, 545, y + 6, "continued on next page ...", 8, right=True)
            text(page, 297, 815, "Page 1 of 2", 8)
            page = new_page(False)
            y = table_header(page, 140)
        text(page, cols["line"], y, str(i))
        text(page, cols["sku"], y, sku, 8.5)
        text(page, cols["desc"], y, desc, 8)
        text(page, cols["qty"], y, f"{qty:g}")
        text(page, cols["unit"], y, unit)
        text(page, cols["price"], y, money(price), right=True)
        text(page, cols["amount"], y, money(round(qty * price, 2)), right=True)
        y += 17
    page.draw_line((45, y - 8), (550, y - 8), color=(0.5, 0.5, 0.5), width=0.6)

    subtotal, total = totals(spec)
    y += 8
    for label, val, f in (("Subtotal:", subtotal, font), (f"{spec.get('tax_label', 'Tax')}:", spec["tax"], font),
                          ("Shipping:", spec["shipping"], font), (f"Total ({spec['currency']}):", total, bold)):
        text(page, 400, y, label, 10, f)
        text(page, 548, y, money(val), 10, f, right=True)
        y += 16
    y += 18
    text(page, 50, y, "Please reference the PO number on all invoices and packing slips.", 8.5)
    text(page, 50, y + 12, "Authorized by: J. Alvarez, Procurement Manager", 8.5)
    if brk:
        text(page, 297, 815, "Page 2 of 2", 8)
    doc.save(path)
    doc.close()


def render_scan(spec: dict, path: Path, tmp_pdf: Path) -> None:
    """Render the PO, then make it look photographed: low DPI, rotation, noise, blur, JPEG artefacts."""
    import pymupdf as fitz
    from PIL import Image, ImageEnhance, ImageFilter

    render_po_pdf(spec, tmp_pdf)
    with fitz.open(tmp_pdf) as doc:
        pix = doc[0].get_pixmap(dpi=130)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    tmp_pdf.unlink(missing_ok=True)
    img = img.convert("L")
    img = img.rotate(-1.7, expand=True, fillcolor=235, resample=Image.BICUBIC)
    noise = Image.effect_noise(img.size, 28).point(lambda p: 255 - (255 - p) // 3)
    img = Image.blend(img, noise, 0.10)
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    img = ImageEnhance.Contrast(img).enhance(0.85)
    img = ImageEnhance.Brightness(img).enhance(0.96)
    img.save(path, "JPEG", quality=55)


CSV_FIELDS = ["po_number", "po_date", "vendor_name", "vendor_tax_id", "buyer_department", "currency",
              "payment_terms", "delivery_date", "line_no", "sku", "description", "quantity", "unit", "unit_price",
              "line_total", "subtotal", "tax", "shipping", "total"]


def csv_rows(spec: dict) -> list[dict]:
    subtotal, total = totals(spec)
    rows = []
    for i, (sku, desc, qty, unit, price) in enumerate(spec["lines"], start=1):
        first = i == 1
        rows.append({
            "po_number": spec["po_number"], "po_date": spec["po_date"], "vendor_name": spec["vendor"],
            "vendor_tax_id": spec["tax_id"], "buyer_department": spec["dept"], "currency": spec["currency"],
            "payment_terms": spec["terms"], "delivery_date": spec["delivery"], "line_no": i, "sku": sku,
            "description": desc, "quantity": qty, "unit": unit, "unit_price": f"{price:.2f}",
            "line_total": f"{qty * price:.2f}",
            "subtotal": f"{subtotal:.2f}" if first else "", "tax": f"{spec['tax']:.2f}" if first else "",
            "shipping": f"{spec['shipping']:.2f}" if first else "", "total": f"{total:.2f}" if first else "",
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def label_for(spec: dict, file: str) -> dict:
    subtotal, total = totals(spec)
    return {
        "file": file, "expected": spec["expected"],
        "fields": {
            "po_number": spec["po_number"], "po_date": spec["po_date"], "vendor_name": spec["vendor"],
            "vendor_tax_id": spec["tax_id"] or None, "buyer_department": spec["dept"], "currency": spec["currency"],
            "payment_terms": spec["terms"], "delivery_date": spec["delivery"] or None,
            "subtotal": subtotal, "tax": spec["tax"], "shipping": spec["shipping"], "total": total,
        },
        "line_items": [{"sku": s, "quantity": q, "unit_price": p, "line_total": round(q * p, 2)}
                       for s, _, q, _, p in spec["lines"]],
    }


def make_samples(out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    labels_dir = out / "labels"
    labels_dir.mkdir(exist_ok=True)
    random.seed(7)
    produced: dict[str, Path] = {}

    for spec in SAMPLES:
        if spec["kind"] == "pdf":
            path = out / f"{spec['id']}.pdf"
            render_po_pdf(spec, path)
        elif spec["kind"] == "scan":
            path = out / f"{spec['id']}.jpg"
            render_scan(spec, path, out / f"_{spec['id']}.pdf")
        else:
            path = out / f"{spec['id']}.csv"
            write_csv(path, csv_rows(spec))
        produced[spec["id"]] = path
        (labels_dir / f"{spec['id']}.json").write_text(json.dumps(label_for(spec, path.name), indent=2))

    batch_path = out / f"{BATCH['id']}.csv"
    write_csv(batch_path, [r for po in BATCH["pos"] for r in csv_rows(po)])
    produced[BATCH["id"]] = batch_path
    (labels_dir / f"{BATCH['id']}.json").write_text(json.dumps(
        {"file": batch_path.name, "batch": [label_for(po, batch_path.name) for po in BATCH["pos"]]}, indent=2))

    # --- edge cases ---
    corrupt = out / "edge_corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.7\n" + os.urandom(2048))
    produced["edge_corrupt"] = corrupt

    import pymupdf as fitz
    locked = out / "edge_locked.pdf"
    tmp = out / "_locked_src.pdf"
    render_po_pdf(SAMPLES[0], tmp)
    with fitz.open(tmp) as d:
        d.save(locked, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="s3cret", owner_pw="owner-s3cret")
    tmp.unlink(missing_ok=True)
    produced["edge_locked"] = locked

    zpath = out / "edge_unsupported.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("po.txt", "not a supported attachment")
    produced["edge_unsupported"] = zpath
    return produced


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "samples"))
    args = ap.parse_args()
    for name, p in make_samples(Path(args.out)).items():
        print(f"{name:28s} {p}")

import json

import pytest

from app.extraction.extractor import (
    extract, extract_heuristic, finalize, normalize_date, normalize_terms, parse_number,
)
from app.extraction.schema import Confidenced, LineItem, PurchaseOrderExtraction
from app.ingestion.loaders import IngestionError, RawDocument, load_document
from app.llm import LLMClient, LLMError, strict_json_schema


@pytest.mark.parametrize("raw,expected", [
    ("$1,234.50", 1234.5), ("1.234,50", 1234.5), ("1,234", 1234.0), ("12,5", 12.5),
    ("(12.00)", -12.0), ("", None), (None, None), ("USD 4,200.00", 4200.0), (7, 7.0),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def test_normalizers():
    assert normalize_date("Oct 20, 2026") == ("2026-10-20", True)
    assert normalize_date("2026-10-20") == ("2026-10-20", True)
    assert normalize_date("sometime soon")[1] is False
    assert normalize_terms("Net 30 days") == "Net 30"
    assert normalize_terms("NET-60") == "Net 60"


def test_text_pdf_heuristic_extraction(samples, tmp_path):
    doc = load_document(samples["01_clean_northwind"], tmp_path / "pages")
    assert doc.doc_type == "pdf_text" and doc.page_images
    po = extract(doc)
    assert po.po_number.value == "PO-2026-0412"
    assert po.vendor_name.value == "Northwind Office Products LLC"
    assert po.po_date.value == "2026-10-20" and po.delivery_date.value == "2026-11-05"
    assert po.payment_terms.value == "Net 30" and po.currency.value == "USD"
    assert [li.sku for li in po.line_items] == ["OFF-CHR-ERG", "OFF-LMP-LED"]
    assert (po.subtotal, po.tax, po.shipping, po.total) == (3700.0, 296.0, 204.0, 4200.0)
    assert po.extraction_method == "heuristic"
    assert not [w for w in po.extraction_warnings if "evidence" in w]


def test_two_column_multipage_pdf(samples, tmp_path):
    """Header split over two columns and line items continuing on page 2."""
    doc = load_document(samples["04_acme_alias_twocol"], tmp_path / "pages")
    assert len(doc.page_images) == 2
    po = extract(doc)
    assert po.vendor_name.value == "ACME Corporation (Pvt) Ltd"  # NOT normalized at extraction
    assert po.po_date.value == "2026-10-18"
    assert po.vendor_tax_id.value == "US-12-3456789"
    assert len(po.line_items) == 10
    assert po.total == pytest.approx(6361.89)


def test_csv_extraction_keeps_missing_field_null(samples, tmp_path):
    doc = load_document(samples["03_unknown_vendor"], tmp_path / "pages")
    po = extract(doc)
    assert po.extraction_method == "csv_columns"
    assert po.vendor_name.value == "Pinnacle Promo Goods Ltd"
    assert not po.delivery_date.present and po.delivery_date.confidence == 0.0   # null, not a guess
    assert len(po.line_items) == 2 and po.total == pytest.approx(2299.92)


def test_evidence_not_in_source_cuts_confidence():
    """Anti-hallucination: a value whose evidence isn't in the document loses confidence."""
    text = "PURCHASE ORDER\nPO Number: PO-1\nVendor: Real Vendor Inc\nTotal: $100.00\n" + "x" * 60
    ext = PurchaseOrderExtraction(
        po_number=Confidenced(value="PO-1", confidence=0.95, evidence="PO Number: PO-1"),
        po_date=Confidenced.missing(),
        vendor_name=Confidenced(value="Imaginary Corp", confidence=0.95, evidence="Vendor: Imaginary Corp"),
        vendor_tax_id=Confidenced.missing(), buyer_department=Confidenced.missing(),
        currency=Confidenced.missing(), payment_terms=Confidenced.missing(), delivery_date=Confidenced.missing(),
        line_items=[LineItem(line_no=1, quantity=1, unit_price=100, line_total=100, confidence=0.9)],
        total=100.0, totals_confidence=0.95, totals_evidence="Total: $100.00",
    )
    po = finalize(ext, RawDocument("pdf_text", text, "text_layer"), "test")
    assert po.po_number.confidence == 0.95
    assert po.vendor_name.confidence < 0.5
    assert any("vendor_name: evidence not found" in w for w in po.extraction_warnings)


def test_null_with_high_confidence_is_reset():
    ext = PurchaseOrderExtraction(
        po_number=Confidenced(value=None, confidence=0.9), po_date=Confidenced.missing(),
        vendor_name=Confidenced.missing(), vendor_tax_id=Confidenced.missing(),
        buyer_department=Confidenced.missing(), currency=Confidenced.missing(),
        payment_terms=Confidenced.missing(), delivery_date=Confidenced.missing(), line_items=[],
    )
    po = finalize(ext, RawDocument("pdf_text", "", "text_layer"), "test")
    assert po.po_number.confidence == 0.0
    assert "No line items were extracted" in po.extraction_warnings


def test_confidence_is_clamped():
    assert Confidenced(value="x", confidence=1.7).confidence == 1.0
    assert Confidenced(value="x", confidence="bad").confidence == 0.0


@pytest.mark.parametrize("key,code", [("edge_locked", "locked_file"), ("edge_corrupt", "corrupt_file"),
                                      ("edge_unsupported", "unsupported_type")])
def test_ingestion_errors_have_codes(samples, tmp_path, key, code):
    with pytest.raises(IngestionError) as exc:
        load_document(samples[key], tmp_path / "pages")
    assert exc.value.code == code


def test_scan_without_ocr_or_llm_fails_with_reason(samples, tmp_path, monkeypatch):
    monkeypatch.setattr("app.ingestion.loaders.ocr_available", lambda: False)
    doc = load_document(samples["02_scanned_globex"], tmp_path / "pages")
    assert doc.doc_type == "image" and doc.page_images
    from app.extraction.extractor import ExtractionError
    with pytest.raises(ExtractionError) as exc:
        extract(doc)
    assert exc.value.code == "no_text_for_offline_extraction"


def test_strict_schema_shape():
    schema = strict_json_schema(PurchaseOrderExtraction)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    conf = schema["$defs"]["Confidenced"]
    assert set(conf["required"]) == {"value", "confidence", "evidence"}
    assert "maximum" not in json.dumps(schema) and '"default"' not in json.dumps(schema)


def _fake_client(responses):
    from app.config import get_settings

    client = object.__new__(LLMClient)
    client.settings = get_settings()
    client.provider = "fake"
    calls = []

    def fake_call(system, text, schema, name, images, max_tokens):
        calls.append(text)
        return responses[len(calls) - 1]

    client._call_with_backoff = fake_call
    return client, calls


def test_llm_invalid_json_is_reprompted_with_validation_error():
    good = PurchaseOrderExtraction(
        po_number=Confidenced(value="PO-9", confidence=0.9), po_date=Confidenced.missing(),
        vendor_name=Confidenced.missing(), vendor_tax_id=Confidenced.missing(),
        buyer_department=Confidenced.missing(), currency=Confidenced.missing(),
        payment_terms=Confidenced.missing(), delivery_date=Confidenced.missing(), line_items=[],
    ).model_dump_json()
    client, calls = _fake_client(['{"po_number": "not an object"}', good])
    out = client.structured(system="s", text="t", response_model=PurchaseOrderExtraction)
    assert out.po_number.value == "PO-9"
    assert len(calls) == 2 and "rejected by schema validation" in calls[1]


def test_llm_gives_up_after_bounded_retries():
    client, calls = _fake_client(["{}", "{}", "{}"])
    with pytest.raises(LLMError):
        client.structured(system="s", text="t", response_model=PurchaseOrderExtraction)
    assert len(calls) == 3


def test_heuristic_on_ocr_text_has_lower_confidence():
    text = "PO Number: PO-77\nVendor: Acme Corp\n1 OFF-LMP-LED LED desk lamp 2 EA $42.50 $85.00\nTotal: $85.00"
    ext = extract_heuristic(RawDocument("image", text, "ocr"))
    assert ext.po_number.confidence < 0.8 and ext.line_items[0].confidence < 0.8

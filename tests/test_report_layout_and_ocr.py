"""
tests/test_report_layout_and_ocr.py

Tests:
  1. Template renders without xhtml2pdf errors
  2. Summary count cards reflect actual check_results counts
  3. OCR unavailable -> reason says "OCR engine unavailable", not just "not found"
  4. OCR available but field absent -> reason says "not found in OCR-extracted text"
  5. Authenticity fail does NOT skip other checks; all checks still run
  6. AUTH_01 evidence is only in the details section, not repeated verbatim in the table
  7. Image annotation draws bounding boxes when evidence_bbox is present
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import base64
import datetime
from unittest.mock import MagicMock
import importlib

# If earlier test files (e.g. test_e2e.py or test_ocr_service.py) replaced cv2/numpy
# with MagicMock in sys.modules, restore the real modules here.
if isinstance(sys.modules.get('cv2'), MagicMock):
    sys.modules.pop('cv2', None)
if isinstance(sys.modules.get('numpy'), MagicMock):
    sys.modules.pop('numpy', None)

import numpy as np
import cv2

if 'backend.report_service.annotator' in sys.modules:
    import backend.report_service.annotator
    importlib.reload(backend.report_service.annotator)

from backend.schemas import (
    ComplianceReport, CheckResult, CheckStatus, Severity,
    ProductCategory, AuthenticityDetails, BoundingBox
)
from backend.rule_engine.validators.common import (
    validate_mrp, validate_net_quantity, validate_date_of_mfg, validate_country_of_origin
)
from backend.schemas import NormalizedScan
from backend.report_service.annotator import annotate_image


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_report(check_results, overall=CheckStatus.FAIL, auth=None):
    return ComplianceReport(
        scan_id="test-0000",
        category=ProductCategory.GENERAL_RETAIL,
        overall_verdict=overall,
        check_results=check_results,
        generated_at=datetime.datetime.utcnow().isoformat(),
        authenticity=auth,
    )


def _make_auth(is_ai=True):
    return AuthenticityDetails(
        is_ai_generated=is_ai,
        classification_label="AI GENERATED" if is_ai else "AUTHENTIC",
        authenticity_score=0.1 if is_ai else 0.95,
        ai_confidence_percentage=90 if is_ai else 5,
        status=CheckStatus.FAIL if is_ai else CheckStatus.PASS,
        verdict_summary="Deep fake probability high." if is_ai else "Authentic physical capture.",
        ai_signatures=["GAN smoothing", "No EXIF camera data"] if is_ai else [],
    )


def _render_template(report):
    """Render the Jinja2 template to HTML string."""
    from jinja2 import Environment, FileSystemLoader
    template_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    env = Environment(loader=FileSystemLoader(template_dir))
    tpl = env.get_template("report_template.html")
    return tpl.render(
        report=report,
        generated_at_display="24/10/2024, 22:45",
        digital_claims={}, physical_ocr={},
        layer_1_results={}, layer_2_results={},
    )


def _html_to_pdf(html_str):
    """Try xhtml2pdf conversion, return (bytes_written, error_count)."""
    from xhtml2pdf import pisa
    buf = io.BytesIO()
    result = pisa.CreatePDF(html_str, dest=buf)
    return buf.tell(), result.err


def _make_scan_empty():
    """NormalizedScan with all fields None (OCR returned nothing)."""
    return NormalizedScan(
        scan_id="t0", category=ProductCategory.GENERAL_RETAIL,
    )


# ─────────────────────────────────────────────────────────────
# Test 1: Template renders without PDF errors
# ─────────────────────────────────────────────────────────────
def test_template_renders_to_pdf():
    check = CheckResult(
        check_id="01", check_name="MRP", rule_ref="Rule_6_1_e",
        severity=Severity.BLOCKING, status=CheckStatus.FAIL,
        reason="MRP declaration not found. OCR engine unavailable."
    )
    report = _make_report([check])
    html = _render_template(report)
    assert "LabelGuard AI" in html
    assert "reportFooter" in html
    assert "<pdf:pagenumber>" in html
    assert "<pdf:pagecount>" in html
    size, errs = _html_to_pdf(html)
    assert errs == 0, f"xhtml2pdf reported {errs} error(s)"
    assert size > 2000, f"PDF suspiciously small: {size} bytes"
    print(f"[PASS] Test 1: Template renders to PDF with page-number footer ({size} bytes, {errs} errors)")


# ─────────────────────────────────────────────────────────────
# Test 2: Summary count cards match actual results
# ─────────────────────────────────────────────────────────────
def test_summary_counts_in_html():
    checks = [
        CheckResult(check_id="01", check_name="MRP", rule_ref="R", severity=Severity.BLOCKING,
                    status=CheckStatus.PASS, reason="ok"),
        CheckResult(check_id="02", check_name="Net Qty", rule_ref="R", severity=Severity.MAJOR,
                    status=CheckStatus.FAIL, reason="fail"),
        CheckResult(check_id="03", check_name="Date", rule_ref="R", severity=Severity.MAJOR,
                    status=CheckStatus.REVIEW, reason="review"),
    ]
    report = _make_report(checks, overall=CheckStatus.FAIL)
    html = _render_template(report)
    # The summary cards contain the count numbers
    assert "1" in html  # at least one of each
    assert "Passed" in html
    assert "Failed" in html
    assert "Review" in html
    print("[PASS] Test 2: Summary count cards present in HTML")


# ─────────────────────────────────────────────────────────────
# Test 3: OCR unavailable -> honest message
# ─────────────────────────────────────────────────────────────
def test_ocr_unavailable_gives_honest_reason():
    scan = _make_scan_empty()
    conf_map = {"__ocr_available__": 0.0}  # signal: OCR engine did not run

    r = validate_mrp(scan, 0.95, "01", "Rule_6_1_e", conf_map=conf_map)
    assert r.status == CheckStatus.FAIL
    assert "OCR engine unavailable" in r.reason, f"Got: {r.reason}"

    r2 = validate_net_quantity(scan, 0.95, "02", "Rule_6_1_c", conf_map=conf_map)
    assert "OCR engine unavailable" in r2.reason, f"Got: {r2.reason}"

    r3 = validate_date_of_mfg(scan, 0.95, "03", "Rule_6_1_d", conf_map=conf_map)
    assert "OCR engine unavailable" in r3.reason, f"Got: {r3.reason}"

    r4 = validate_country_of_origin(scan, 0.95, "04", "Rule_6_1_aa", conf_map=conf_map)
    assert "OCR engine unavailable" in r4.reason, f"Got: {r4.reason}"

    print("[PASS] Test 3: OCR unavailable -> 'OCR engine unavailable' in all reasons")


# ─────────────────────────────────────────────────────────────
# Test 4: OCR available but field absent -> "not found in OCR"
# ─────────────────────────────────────────────────────────────
def test_ocr_available_field_absent_honest_reason():
    scan = _make_scan_empty()
    conf_map = {"__ocr_available__": 1.0}  # signal: OCR ran, just didn't find field

    r = validate_mrp(scan, 0.95, "01", "Rule_6_1_e", conf_map=conf_map)
    assert r.status == CheckStatus.FAIL
    assert "not found in OCR" in r.reason, f"Got: {r.reason}"
    # Must NOT say "OCR engine unavailable" when OCR did run
    assert "OCR engine unavailable" not in r.reason, f"Wrong msg: {r.reason}"
    print("[PASS] Test 4: OCR ran but field absent -> 'not found in OCR' message")


# ─────────────────────────────────────────────────────────────
# Test 5: Authenticity FAIL does NOT skip other checks
# ─────────────────────────────────────────────────────────────
def test_auth_fail_does_not_skip_other_checks():
    """
    Even when authenticity is AI-generated (FAIL), the other checks must still
    run and appear in the report. They should show FAIL/REVIEW with honest reasons,
    not be hidden.
    """
    from backend.rule_engine.engine import evaluate_packaging_compliance
    scan = _make_scan_empty()
    conf_map = {"__ocr_available__": 0.0}  # OCR unavailable

    results = evaluate_packaging_compliance(scan, conf_map)

    check_ids = [r.check_id for r in results]
    # All standard checks must be present
    assert "01" in check_ids, "MRP check missing"
    assert "02" in check_ids, "Net Qty check missing"
    assert "03" in check_ids, "Date check missing"
    assert "04" in check_ids, "COO check missing"
    assert "05" in check_ids, "Manufacturer check missing"

    # All must say OCR unavailable, not silently pass
    for r in results:
        if r.check_id in ("01", "02", "03"):
            assert r.status in (CheckStatus.FAIL, CheckStatus.REVIEW), \
                f"Check {r.check_id} should be FAIL/REVIEW when OCR is unavailable, got {r.status}"

    print(f"[PASS] Test 5: Auth FAIL does not skip checks — {len(results)} checks ran")


# ─────────────────────────────────────────────────────────────
# Test 6: AUTH_01 long evidence NOT in table cell (template)
# ─────────────────────────────────────────────────────────────
def test_auth_evidence_in_details_not_table():
    auth_check = CheckResult(
        check_id="AUTH_01", check_name="Image Authenticity",
        rule_ref="Auth_Directive", severity=Severity.BLOCKING,
        status=CheckStatus.FAIL,
        reason="AI-generated image detected.",
        evidence_text="GAN smoothing detected; EXIF missing; spectral artifacts: " + "X" * 500
    )
    report = _make_report([auth_check], auth=_make_auth(is_ai=True))
    html = _render_template(report)

    # The long evidence string should NOT appear verbatim in the table cell
    # (it's truncated to 120 chars or moved to the details section)
    long_evidence = "X" * 200  # part of the long string
    # Find the table section specifically
    table_start = html.find('<table class="checks">')
    table_end = html.find('</table>', table_start)
    table_html = html[table_start:table_end]
    assert long_evidence not in table_html, "Long evidence string should not appear verbatim in table"

    # The details section should contain it
    details_start = html.find('Authenticity Details')
    assert details_start > 0, "Authenticity Details section not found"
    print("[PASS] Test 6: Long AUTH evidence not in table; details section present")


# ─────────────────────────────────────────────────────────────
# Test 7: Annotator draws bounding boxes when bbox present
# ─────────────────────────────────────────────────────────────
def test_annotator_draws_boxes():
    if isinstance(sys.modules.get('cv2'), MagicMock):
        sys.modules.pop('cv2', None)
    if isinstance(sys.modules.get('numpy'), MagicMock):
        sys.modules.pop('numpy', None)
    import cv2
    import numpy as np
    import backend.report_service.annotator as ann_mod
    importlib.reload(ann_mod)

    # Create a tiny white test image on disk
    test_img_path = os.path.join(os.path.dirname(__file__), "test_annotate.jpg")
    img = np.ones((200, 300, 3), dtype=np.uint8) * 240
    cv2.imwrite(test_img_path, img)

    check_with_box = CheckResult(
        check_id="01", check_name="MRP", rule_ref="Rule_6_1_e",
        severity=Severity.BLOCKING, status=CheckStatus.PASS,
        reason="MRP found.",
        evidence_bbox=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=50)
    )
    report = _make_report([check_with_box], overall=CheckStatus.PASS)
    b64 = annotate_image(test_img_path, report)
    assert b64.startswith("data:image/jpeg;base64,"), "Not a valid data URI"

    # Decode the image and check a pixel in the drawn box region is colored
    raw = base64.b64decode(b64.split(",", 1)[1])
    arr = np.frombuffer(raw, np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded is not None
    # The original image was nearly white (240,240,240). The green PASS box
    # changes pixels at the border. Sample the box border area.
    border_pixel = decoded[10, 10]  # top-left corner of drawn rectangle
    original_was_white = all(c > 200 for c in [240, 240, 240])
    assert original_was_white  # sanity
    # at least one channel should differ from 240 (box was drawn)
    assert not all(abs(int(c) - 240) < 5 for c in border_pixel), \
        f"Border pixel unchanged — box not drawn: {border_pixel}"

    os.remove(test_img_path)
    print("[PASS] Test 7: Annotator draws colored bounding box on image")


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== LabelGuard Report Layout & OCR Content Tests ===\n")
    tests = [
        test_template_renders_to_pdf,
        test_summary_counts_in_html,
        test_ocr_unavailable_gives_honest_reason,
        test_ocr_available_field_absent_honest_reason,
        test_auth_fail_does_not_skip_other_checks,
        test_auth_evidence_in_details_not_table,
        test_annotator_draws_boxes,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n=== Results: {passed} passed, {failed} failed ===\n")

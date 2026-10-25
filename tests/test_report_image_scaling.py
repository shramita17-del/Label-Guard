import os
import io
import base64
import pytest
from PIL import Image
import pypdf

from backend.report_service.pdf_generator import (
    compute_image_display_dimensions,
    prepare_report_image,
    generate_pdf_report,
    MAX_W,
    MAX_H,
)
from backend.schemas import (
    ComplianceReport, CheckResult, CheckStatus, Severity, ProductCategory, BoundingBox
)


def _create_synthetic_image_b64(width: int, height: int, color=(200, 220, 240)) -> str:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def test_display_size_tall_image():
    """Asserts computed display size for tall image (1200x1600) is within MAX_W x MAX_H with <1% aspect ratio error."""
    w, h = 1200, 1600
    disp_w, disp_h = compute_image_display_dimensions(w, h, max_w_mm=MAX_W, max_h_mm=MAX_H)
    assert disp_w <= MAX_W + 1e-3, f"disp_w {disp_w} exceeds MAX_W {MAX_W}"
    assert disp_h <= MAX_H + 1e-3, f"disp_h {disp_h} exceeds MAX_H {MAX_H}"

    orig_aspect = w / h
    computed_aspect = disp_w / disp_h
    rel_error = abs(orig_aspect - computed_aspect) / orig_aspect
    assert rel_error < 0.01, f"Aspect ratio error {rel_error:.4f} >= 1%"


def test_display_size_wide_image():
    """Asserts computed display size for wide image (1600x600) is within MAX_W x MAX_H with <1% aspect ratio error."""
    w, h = 1600, 600
    disp_w, disp_h = compute_image_display_dimensions(w, h, max_w_mm=MAX_W, max_h_mm=MAX_H)
    assert disp_w <= MAX_W + 1e-3, f"disp_w {disp_w} exceeds MAX_W {MAX_W}"
    assert disp_h <= MAX_H + 1e-3, f"disp_h {disp_h} exceeds MAX_H {MAX_H}"

    orig_aspect = w / h
    computed_aspect = disp_w / disp_h
    rel_error = abs(orig_aspect - computed_aspect) / orig_aspect
    assert rel_error < 0.01, f"Aspect ratio error {rel_error:.4f} >= 1%"


def test_generate_pdf_for_tall_and_wide_images(tmp_path):
    """Generates PDF for tall and wide images and verifies proper rendering and scaling."""
    template_dir = os.path.join(os.path.dirname(__file__), "..", "reports")

    for name, w, h in [("tall", 1200, 1600), ("wide", 1600, 600)]:
        img_b64 = _create_synthetic_image_b64(w, h)
        report = ComplianceReport(
            scan_id=f"test_{name}",
            category=ProductCategory.GENERAL_RETAIL,
            overall_verdict=CheckStatus.PASS,
            check_results=[
                CheckResult(
                    check_id="01", check_name="MRP", rule_ref="Rule_6_1_e",
                    severity=Severity.BLOCKING, status=CheckStatus.PASS,
                    evidence_text="Rs. 100",
                    evidence_bbox=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=50),
                    confidence=0.95, reason="Valid."
                )
            ],
            annotated_image_base64=img_b64,
            generated_at="2026-10-25T12:00:00Z"
        )
        out_pdf = str(tmp_path / f"test_{name}.pdf")
        pdf_path = generate_pdf_report(report, template_dir, out_pdf)
        assert os.path.exists(pdf_path), f"PDF not generated for {name}"

        # Verify PDF is valid and readable
        reader = pypdf.PdfReader(pdf_path)
        assert len(reader.pages) >= 1

        # Verify downscaling produced an image with longest side <= 1000 px
        processed_b64, dw, dh = prepare_report_image(img_b64, MAX_W, MAX_H)
        raw = base64.b64decode(processed_b64.split(",", 1)[1])
        downscaled_img = Image.open(io.BytesIO(raw))
        assert max(downscaled_img.size) <= 1000

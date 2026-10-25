import sys
import os
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.schemas import (
    ComplianceReport, CheckResult, CheckStatus, Severity,
    ProductCategory, AuthenticityDetails, BoundingBox
)
from backend.report_service.pdf_generator import generate_pdf_report
from backend.report_service.annotator import annotate_image

# 1. Setup sample authenticity details
auth = AuthenticityDetails(
    is_ai_generated=True,
    classification_label="AI GENERATED (SYNTHETIC)",
    authenticity_score=0.08,
    ai_confidence_percentage=92,
    status=CheckStatus.FAIL,
    verdict_summary="High-confidence AI/GAN synthetic image. EXIF data absent. Spectral artifacts detected.",
    ai_signatures=["GAN texture smoothing", "No EXIF camera data", "Unnatural noise pattern"],
    mockup_disclaimers=["Product render detected", "Studio background"],
    spectral_artifact_score=0.87,
    noise_residual_score=0.91,
    barcode_status="Barcode region detected but not scannable (likely rendered)",
)

# 2. Setup check results with real bounding boxes for detected fields
checks = [
    CheckResult(
        check_id="AUTH_01",
        check_name="Image Authenticity",
        rule_ref="Auth_Directive",
        severity=Severity.BLOCKING,
        status=CheckStatus.FAIL,
        reason="AI-generated image detected. See Authenticity Details below.",
        evidence_text="GAN smoothing; No EXIF; Spectral artifacts: 0.87; Noise residual: 0.91; Barcode rendered.",
    ),
    CheckResult(
        check_id="01",
        check_name="MRP",
        rule_ref="Rule_6_1_e",
        severity=Severity.BLOCKING,
        status=CheckStatus.FAIL,
        reason="MRP declaration not found. OCR engine unavailable — text could not be read from this image.",
        evidence_bbox=BoundingBox(x_min=60, y_min=120, x_max=220, y_max=160),
    ),
    CheckResult(
        check_id="02",
        check_name="Net Quantity",
        rule_ref="Rule_6_1_c",
        severity=Severity.MAJOR,
        status=CheckStatus.FAIL,
        reason="Net quantity not found. OCR engine unavailable — text could not be read from this image.",
        evidence_bbox=BoundingBox(x_min=60, y_min=180, x_max=220, y_max=215),
    ),
    CheckResult(
        check_id="03",
        check_name="Date of Mfg",
        rule_ref="Rule_6_1_d",
        severity=Severity.MAJOR,
        status=CheckStatus.FAIL,
        reason="Manufacturing date not found. OCR engine unavailable — text could not be read from this image.",
        evidence_bbox=BoundingBox(x_min=60, y_min=230, x_max=240, y_max=265),
    ),
    CheckResult(
        check_id="04",
        check_name="Country of Origin",
        rule_ref="Rule_6_1_aa",
        severity=Severity.BLOCKING,
        status=CheckStatus.FAIL,
        reason="Country of origin not found. OCR engine unavailable — text could not be read from this image.",
        evidence_bbox=BoundingBox(x_min=60, y_min=280, x_max=250, y_max=315),
    ),
    CheckResult(
        check_id="05",
        check_name="Manufacturer/Packer",
        rule_ref="Rule_6_1_a",
        severity=Severity.MAJOR,
        status=CheckStatus.FAIL,
        reason="Manufacturer name or address not found. OCR engine unavailable — text could not be read from this image.",
        evidence_bbox=BoundingBox(x_min=60, y_min=330, x_max=320, y_max=380),
    ),
    CheckResult(
        check_id="06",
        check_name="Consumer Care",
        rule_ref="Rule_6_1_f",
        severity=Severity.MAJOR,
        status=CheckStatus.PASS,
        reason="Consumer care contact details verified.",
        evidence_text="care@brand.com, Ph: 1800-123-456",
        evidence_bbox=BoundingBox(x_min=60, y_min=400, x_max=300, y_max=440),
    ),
]

# 3. Check if scan image exists or find uploaded scan images
scan_img_path = None
for candidate in [
    "reports/generated/f090b8a1_extracted_1_0.jpg",
    "reports/generated/f090b8a1.jpg",
    "reports/generated/f090b8a1.png",
    "data/samples/sample_label.jpg",
    "data/samples/sample_label.png",
]:
    if os.path.exists(candidate):
        scan_img_path = candidate
        break

report = ComplianceReport(
    scan_id="f090b8a1",
    category=ProductCategory.GENERAL_RETAIL,
    overall_verdict=CheckStatus.FAIL,
    check_results=checks,
    generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    authenticity=auth,
)

if scan_img_path and os.path.exists(scan_img_path):
    report.annotated_image_base64 = annotate_image(scan_img_path, report)
else:
    # Use fallback canvas annotator
    report.annotated_image_base64 = annotate_image("nonexistent.jpg", report)

out_pdf = "reports/generated/LabelGuard_Report_f090b8a1_redesigned.pdf"
pdf_path = generate_pdf_report(
    report=report,
    template_dir="reports",
    output_path=out_pdf,
)

print(f"Sample PDF successfully generated at: {pdf_path} (Size: {os.path.getsize(pdf_path)} bytes)")

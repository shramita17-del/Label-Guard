import os
import io
import base64
import logging
from PIL import Image
from jinja2 import Environment, FileSystemLoader
from backend.schemas import ComplianceReport

logger = logging.getLogger(__name__)

# ── Compliance report image display constraints (in mm) ──
MAX_W = 85.0  # mm: maximum bounding box width
MAX_H = 60.0  # mm: maximum bounding box height
DEFAULT_DPI = 96.0  # Standard reference DPI for mm/pixel conversion
MM_PER_INCH = 25.4


def compute_image_display_dimensions(
    img_w_px: int,
    img_h_px: int,
    max_w_mm: float = MAX_W,
    max_h_mm: float = MAX_H,
    dpi: float = DEFAULT_DPI,
) -> tuple[float, float]:
    """
    Computes display size (width_mm, height_mm) fitting inside max_w_mm x max_h_mm,
    strictly preserving the image aspect ratio:
        scale = min(max_w_mm / img_w_mm, max_h_mm / img_h_mm, 1.0)
    """
    if img_w_px <= 0 or img_h_px <= 0:
        return max_w_mm, max_h_mm

    img_w_mm = (img_w_px / dpi) * MM_PER_INCH
    img_h_mm = (img_h_px / dpi) * MM_PER_INCH

    scale = min(max_w_mm / img_w_mm, max_h_mm / img_h_mm, 1.0)
    disp_w_mm = img_w_mm * scale
    disp_h_mm = img_h_mm * scale
    return disp_w_mm, disp_h_mm


def prepare_report_image(
    image_src_or_base64: str,
    max_w_mm: float = MAX_W,
    max_h_mm: float = MAX_H,
    dpi: float = DEFAULT_DPI,
) -> tuple[str, str, str]:
    """
    Opens the image with Pillow, downscales so longest side <= 1000 px (LANCZOS),
    saves as JPEG quality 85, and calculates display width and height in mm.
    Returns (image_data_uri, display_width_str, display_height_str).
    """
    if not image_src_or_base64:
        return "", f"{max_w_mm:.1f}mm", f"{max_h_mm:.1f}mm"

    pil_img = None
    try:
        if image_src_or_base64.startswith("data:image"):
            b64_part = image_src_or_base64.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_part)
            pil_img = Image.open(io.BytesIO(raw_bytes))
        elif os.path.exists(image_src_or_base64):
            pil_img = Image.open(image_src_or_base64)
        else:
            try:
                raw_bytes = base64.b64decode(image_src_or_base64)
                pil_img = Image.open(io.BytesIO(raw_bytes))
            except Exception:
                pass
    except Exception as exc:
        logger.warning(f"Failed to open image with Pillow: {exc}")

    if pil_img is None:
        return image_src_or_base64, f"{max_w_mm:.1f}mm", f"{max_h_mm:.1f}mm"

    cur_w, cur_h = pil_img.size
    longest = max(cur_w, cur_h)
    if longest > 1000:
        scale_factor = 1000.0 / longest
        new_w = max(1, int(round(cur_w * scale_factor)))
        new_h = max(1, int(round(cur_h * scale_factor)))
        resample_filter = getattr(Image, "Resampling", Image).LANCZOS
        pil_img = pil_img.resize((new_w, new_h), resample_filter)
        cur_w, cur_h = pil_img.size

    buf = io.BytesIO()
    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buf, format="JPEG", quality=85)
    processed_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    disp_w_mm, disp_h_mm = compute_image_display_dimensions(cur_w, cur_h, max_w_mm, max_h_mm, dpi)
    display_width = f"{disp_w_mm:.1f}mm"
    display_height = f"{disp_h_mm:.1f}mm"

    return processed_b64, display_width, display_height


# ── PDF engine: xhtml2pdf (pure-Python, works on Windows without GTK/Pango) ──
try:
    from xhtml2pdf import pisa
    _HAS_XHTML2PDF = True
    logger.info("PDF engine: xhtml2pdf loaded successfully.")
except ImportError:
    pisa = None
    _HAS_XHTML2PDF = False
    logger.warning("xhtml2pdf not installed. PDF output will use HTML fallback.")


def _convert_html_to_pdf(html_string: str, output_path: str) -> bool:
    """
    Convert an HTML string to a PDF file using xhtml2pdf.
    Returns True on success, False on failure.
    """
    if not _HAS_XHTML2PDF:
        return False

    try:
        with open(output_path, "wb") as pdf_file:
            result = pisa.CreatePDF(
                src=html_string,
                dest=pdf_file,
                encoding="utf-8",
            )
            if result.err:
                logger.error(f"xhtml2pdf reported {result.err} error(s) during conversion.")
                return False
            return True
    except Exception as exc:
        logger.error(f"xhtml2pdf conversion failed: {exc}")
        return False


def generate_pdf_report(
    report: ComplianceReport,
    template_dir: str,
    output_path: str,
    digital_claims: dict = None,
    physical_ocr: dict = None,
    layer_1_results: dict = None,
    layer_2_results: dict = None,
) -> str:
    """
    Renders the compliance report to HTML via Jinja2, then converts it to
    a real PDF file using xhtml2pdf.  Falls back to an HTML file only if
    the PDF engine is unavailable.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Force the output path to end with .pdf
    if not output_path.lower().endswith(".pdf"):
        output_path = os.path.splitext(output_path)[0] + ".pdf"

    # ── Jinja2 rendering ──
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html")

    # Format the generated_at timestamp for the report
    generated_at_display = report.generated_at
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(report.generated_at.replace("Z", "+00:00"))
        generated_at_display = dt.strftime("%d/%m/%Y, %H:%M")
    except Exception:
        pass

    # Process scanned/annotated image display sizing for PDF
    raw_img = getattr(report, "annotated_image_base64", None)
    image_src = None
    display_width = None
    display_height = None
    image_caption = None

    if raw_img:
        image_src, display_width, display_height = prepare_report_image(raw_img)
        has_bbox = any(getattr(c, "evidence_bbox", None) is not None for c in (report.check_results or []))
        image_caption = (
            "Annotated Package Scan (OpenCV Bounding-Box Detection)"
            if has_bbox
            else "Scanned Package Image"
        )
        report.annotated_image_base64 = image_src

    html_out = template.render(
        report=report,
        generated_at_display=generated_at_display,
        image_src=image_src,
        display_width=display_width,
        display_height=display_height,
        image_caption=image_caption,
        digital_claims=digital_claims or {},
        physical_ocr=physical_ocr or {},
        layer_1_results=layer_1_results or {},
        layer_2_results=layer_2_results or {},
    )

    # ── PDF conversion ──
    if _convert_html_to_pdf(html_out, output_path):
        logger.info(f"PDF report generated: {output_path}")
        return output_path

    # ── Fallback: save raw HTML (last resort) ──
    logger.warning("PDF engine failed or unavailable. Saving HTML fallback.")
    html_path = output_path.replace(".pdf", ".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_out)
    return html_path

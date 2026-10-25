import io
import cv2
import base64
import numpy as np
from PIL import Image
from backend.schemas import ComplianceReport, CheckStatus

# BGR color palette per status
_STATUS_COLORS = {
    CheckStatus.PASS:           (34, 197, 94),    # green
    CheckStatus.FAIL:           (59,  59, 220),   # red (BGR)
    CheckStatus.REVIEW:         (37, 150, 255),   # amber-orange
    CheckStatus.EXEMPT:         (200, 200, 200),
    CheckStatus.NOT_APPLICABLE: (180, 180, 180),
}

# Friendly field labels used as box captions
_CHECK_LABELS = {
    "MRP":                 "MRP",
    "Net Quantity":        "Net Qty",
    "Date of Mfg":        "Mfg Date",
    "Country of Origin":   "Country",
    "Manufacturer/Packer": "Manufacturer",
    "Veg/Non-Veg Mark":   "Veg Mark",
    "FSSAI License":       "FSSAI",
    "Fiber Composition":   "Fiber",
    "Image Authenticity":  "Auth",
}


def annotate_image(image_path: str, report: ComplianceReport) -> str:
    """
    Draws colored bounding boxes and label text on the scan image for every
    CheckResult that has an evidence_bbox.

    - Green  = PASS
    - Red    = FAIL
    - Orange = REVIEW

    Returns a data-URI base64 JPEG string ready to embed in HTML/PDF.
    """
    img = cv2.imread(image_path)
    if img is None:
        # Blank canvas fallback (no image on disk, e.g. temp file cleaned up)
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        img[:] = (245, 245, 245)

    h, w = img.shape[:2]

    # ── draw an overlay for each check that has a bbox ──────────────────────
    boxes_drawn = 0
    for check in report.check_results:
        bbox = check.evidence_bbox
        if not bbox:
            continue

        # clamp to image bounds
        x1 = max(0, min(bbox.x_min, w - 1))
        y1 = max(0, min(bbox.y_min, h - 1))
        x2 = max(0, min(bbox.x_max, w - 1))
        y2 = max(0, min(bbox.y_max, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color = _STATUS_COLORS.get(check.status, (200, 200, 200))
        label = _CHECK_LABELS.get(check.check_name, check.check_name)
        caption = f"{label}: {check.status.value}"

        # Thicker box for FAIL/REVIEW, thinner for PASS
        thickness = 3 if check.status == CheckStatus.FAIL else 2

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label background pill above the box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thick = 1
        (tw, th), baseline = cv2.getTextSize(caption, font, font_scale, font_thick)
        label_y = max(th + 4, y1 - 2)
        cv2.rectangle(img,
                       (x1, label_y - th - 4),
                       (x1 + tw + 6, label_y + baseline),
                       color, cv2.FILLED)
        cv2.putText(img, caption,
                    (x1 + 3, label_y - 2),
                    font, font_scale, (255, 255, 255), font_thick,
                    cv2.LINE_AA)
        boxes_drawn += 1

    # ── if NO boxes were drawn, add a subtle "No regions detected" watermark ─
    if boxes_drawn == 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "No region boxes detected"
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 1)
        cx, cy = w // 2 - tw // 2, h // 2
        cv2.putText(img, text, (cx, cy), font, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

    # ── downscale so longest side <= 1000 px (LANCZOS) & encode to JPEG q=85 ──
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    cur_w, cur_h = pil_img.size
    longest = max(cur_w, cur_h)
    if longest > 1000:
        scale_factor = 1000.0 / longest
        new_w = max(1, int(round(cur_w * scale_factor)))
        new_h = max(1, int(round(cur_h * scale_factor)))
        resample_filter = getattr(Image, "Resampling", Image).LANCZOS
        pil_img = pil_img.resize((new_w, new_h), resample_filter)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_b64}"

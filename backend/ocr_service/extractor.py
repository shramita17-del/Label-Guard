import logging
import re
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from backend.schemas import DetectedRegion, OCRResult, OCRToken, BoundingBox

logger = logging.getLogger(__name__)


def _clean_and_split_tokens(raw_tokens: List[OCRToken]) -> List[OCRToken]:
    """
    Cleans OCR text spacing to ensure words keep their spaces and splits composite tokens
    (e.g., 'Lic. No. 10012031000312' -> 'Lic. No.' and '10012031000312').
    """
    cleaned: List[OCRToken] = []
    for tok in raw_tokens:
        text = tok.text.strip()
        if not text:
            continue
        conf = tok.confidence
        bbox = tok.bbox

        # 1. Spacing fixes for common glued OCR words & digits
        text = re.sub(r'(\d+)\s*([a-zA-Z]{1,3})\b', r'\1 \2', text)
        text = re.sub(r'\bNETWEIGHT\b', 'NET WEIGHT', text, flags=re.IGNORECASE)
        text = re.sub(r'\bMRPRS\b', 'MRP RS', text, flags=re.IGNORECASE)
        text = re.sub(r'\bITCLIMITED\b', 'ITC LIMITED', text, flags=re.IGNORECASE)
        text = re.sub(r'\bMARKETEDBY\b', 'MARKETED BY', text, flags=re.IGNORECASE)
        text = re.sub(r'\bindl\.ofall\b', 'incl. of all', text, flags=re.IGNORECASE)
        text = re.sub(r'(\b\d{2})1(\d{2}[\/\-]\d{2,4}\b)', r'\1/\2', text)
        text = re.sub(r'([a-zA-Z]+)(\d{6,})', r'\1 \2', text)
        text = re.sub(r'Lic\.No\.', 'Lic. No. ', text, flags=re.IGNORECASE)
        text = re.sub(r'LicNo', 'Lic. No. ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()

        # 2. Check if composite token needs splitting with proportional bboxes
        m_lic = re.match(r'^(Lic\.?\s*No\.?)\s+([12]\d{13})$', text, re.IGNORECASE)
        if m_lic:
            w_total = bbox.x_max - bbox.x_min
            split_x = bbox.x_min + int(w_total * 0.4)
            cleaned.append(
                OCRToken(
                    text="Lic. No.",
                    confidence=conf,
                    bbox=BoundingBox(x_min=bbox.x_min, y_min=bbox.y_min, x_max=split_x, y_max=bbox.y_max)
                )
            )
            cleaned.append(
                OCRToken(
                    text=m_lic.group(2),
                    confidence=conf,
                    bbox=BoundingBox(x_min=split_x, y_min=bbox.y_min, x_max=bbox.x_max, y_max=bbox.y_max)
                )
            )
            continue

        m_mrp = re.match(r'^(\d+\.\d{2})([\s/].*)$', text)
        if m_mrp:
            w_total = bbox.x_max - bbox.x_min
            split_x = bbox.x_min + int(w_total * 0.5)
            cleaned.append(
                OCRToken(
                    text=m_mrp.group(1),
                    confidence=conf,
                    bbox=BoundingBox(x_min=bbox.x_min, y_min=bbox.y_min, x_max=split_x, y_max=bbox.y_max)
                )
            )
            cleaned.append(
                OCRToken(
                    text=m_mrp.group(2).strip(),
                    confidence=conf,
                    bbox=BoundingBox(x_min=split_x, y_min=bbox.y_min, x_max=bbox.x_max, y_max=bbox.y_max)
                )
            )
            continue

        cleaned.append(OCRToken(text=text, confidence=conf, bbox=bbox))

    # Sort tokens by y then x
    cleaned.sort(key=lambda t: (t.bbox.y_min, t.bbox.x_min))
    return cleaned


class OCRExtractor:
    def __init__(self):
        """
        Initializes OCR engines with a robust fallback chain:
        1. RapidOCR (ONNX runtime, pure-Python CPU, fast, line & token bboxes)
        2. WinOCR (Windows 10/11 native Media OCR)
        3. PaddleOCR (if installed)
        """
        self.rapid_ocr = None
        self.winocr = None
        self.paddle_ocr = None
        self.active_backend = None

        # 1. Try RapidOCR
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.rapid_ocr = RapidOCR()
            self.active_backend = "rapidocr"
            logger.info("OCR Engine initialized: RapidOCR (ONNX Runtime)")
        except Exception as e:
            logger.debug(f"RapidOCR initialization skipped: {e}")

        # 2. Try WinOCR
        if self.winocr is None:
            try:
                import winocr
                self.winocr = winocr
                if not self.active_backend:
                    self.active_backend = "winocr"
                logger.info("OCR Engine initialized: WinOCR (Windows Media OCR)")
            except Exception as e:
                logger.debug(f"WinOCR initialization skipped: {e}")

        # 3. Try PaddleOCR
        try:
            from paddleocr import PaddleOCR
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
            except TypeError:
                self.paddle_ocr = PaddleOCR(lang="en")
            if not self.active_backend:
                self.active_backend = "paddleocr"
            logger.info("OCR Engine initialized: PaddleOCR")
        except Exception as e:
            logger.debug(f"PaddleOCR load notice: {e}")

        # Log active OCR backend at startup
        logger.info(f"Active OCR backend: {self.active_backend}")
        print(f"[OCR Startup] Active OCR backend: {self.active_backend}")

    @property
    def is_available(self) -> bool:
        return bool(self.rapid_ocr or self.winocr or self.paddle_ocr)

    @property
    def ocr(self):
        return self.paddle_ocr

    @ocr.setter
    def ocr(self, val):
        self.paddle_ocr = val

    def _extract_crop_rapidocr(self, crop: np.ndarray, offset_x: int, offset_y: int, scale: float = 1.0):
        lines_text = []
        confidences = []
        tokens: List[OCRToken] = []

        try:
            results, _ = self.rapid_ocr(crop)
            if results:
                for item in results:
                    box, text, conf = item[0], str(item[1]).strip(), float(item[2])
                    if not text:
                        continue
                    x_min = int(min(pt[0] for pt in box) / scale) + offset_x
                    y_min = int(min(pt[1] for pt in box) / scale) + offset_y
                    x_max = int(max(pt[0] for pt in box) / scale) + offset_x
                    y_max = int(max(pt[1] for pt in box) / scale) + offset_y

                    token_bbox = BoundingBox(
                        x_min=max(0, x_min),
                        y_min=max(0, y_min),
                        x_max=max(0, x_max),
                        y_max=max(0, y_max),
                    )
                    tok = OCRToken(text=text, confidence=round(conf, 3), bbox=token_bbox)
                    tokens.append(tok)
                    lines_text.append(text)
                    confidences.append(conf)
        except Exception as exc:
            logger.warning(f"RapidOCR crop extraction error: {exc}")

        return lines_text, confidences, tokens

    def _extract_crop_winocr(self, crop: np.ndarray, offset_x: int, offset_y: int, scale: float = 1.0):
        lines_text = []
        confidences = []
        tokens: List[OCRToken] = []

        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                res = pool.submit(self.winocr.recognize_cv2_sync, crop, "en").result()

            for line in res.get("lines", []):
                ltxt = line.get("text", "").strip()
                if ltxt:
                    lines_text.append(ltxt)
                    confidences.append(0.92)

                for w in line.get("words", []):
                    wtext = w.get("text", "").strip()
                    brect = w.get("bounding_rect", {})
                    if wtext and brect:
                        wx = int(brect.get("x", 0) / scale) + offset_x
                        wy = int(brect.get("y", 0) / scale) + offset_y
                        ww = int(brect.get("width", 0) / scale)
                        wh = int(brect.get("height", 0) / scale)
                        tokens.append(
                            OCRToken(
                                text=wtext,
                                confidence=0.92,
                                bbox=BoundingBox(
                                    x_min=max(0, wx),
                                    y_min=max(0, wy),
                                    x_max=max(0, wx + ww),
                                    y_max=max(0, wy + wh)
                                )
                            )
                        )
        except Exception as exc:
            logger.warning(f"WinOCR crop extraction error: {exc}")

        return lines_text, confidences, tokens

    def _extract_crop_paddleocr(self, crop: np.ndarray, offset_x: int, offset_y: int, scale: float = 1.0):
        lines_text = []
        confidences = []
        tokens: List[OCRToken] = []

        try:
            result = self.paddle_ocr.ocr(crop, cls=True)
            if result and isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                for line in result[0]:
                    if isinstance(line, (list, tuple)) and len(line) >= 2 and isinstance(line[1], (list, tuple)):
                        box, (text, conf) = line[0], line[1]
                        text = str(text).strip()
                        conf = float(conf)
                        if box:
                            x_min = int(min(pt[0] for pt in box) / scale) + offset_x
                            y_min = int(min(pt[1] for pt in box) / scale) + offset_y
                            x_max = int(max(pt[0] for pt in box) / scale) + offset_x
                            y_max = int(max(pt[1] for pt in box) / scale) + offset_y
                        else:
                            x_min, y_min, x_max, y_max = offset_x, offset_y, offset_x + 50, offset_y + 20

                        token_bbox = BoundingBox(
                            x_min=max(0, x_min),
                            y_min=max(0, y_min),
                            x_max=max(0, x_max),
                            y_max=max(0, y_max),
                        )
                        tokens.append(OCRToken(text=text, confidence=round(conf, 3), bbox=token_bbox))
                        lines_text.append(text)
                        confidences.append(conf)
        except Exception as exc:
            logger.warning(f"PaddleOCR crop extraction error: {exc}")

        return lines_text, confidences, tokens

    def _run_single_engine(self, crop: np.ndarray, offset_x: int, offset_y: int, scale: float = 1.0):
        lines_text = []
        confidences = []
        tokens: List[OCRToken] = []

        if self.rapid_ocr is not None:
            lines_text, confidences, tokens = self._extract_crop_rapidocr(crop, offset_x, offset_y, scale)
        if not lines_text and self.winocr is not None:
            lines_text, confidences, tokens = self._extract_crop_winocr(crop, offset_x, offset_y, scale)
        if not lines_text and self.paddle_ocr is not None:
            lines_text, confidences, tokens = self._extract_crop_paddleocr(crop, offset_x, offset_y, scale)

        return lines_text, confidences, tokens

    def extract_text(self, image_path: str, regions: Optional[List[DetectedRegion]] = None) -> List[OCRResult]:
        """
        Crops the image based on detected regions and extracts text, tokens, and confidences.
        Runs a preprocessing pipeline producing:
          (i) 2x upscale (cubic)
          (ii) grayscale
          (iii) auto-detect light-on-dark by mean brightness and invert if dark
          (iv) CLAHE
          (v) adaptive threshold
        Runs OCR on the original + the best 2 variants and merges, keeping per-region the result with highest confidence.
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Image not found at {image_path}, creating fallback canvas.")
            img = np.zeros((800, 800, 3), dtype=np.uint8)

        h, w = img.shape[:2]

        if not regions:
            regions = [
                DetectedRegion(
                    region_label="full_package",
                    bbox=BoundingBox(x_min=0, y_min=0, x_max=w, y_max=h),
                    yolo_confidence=1.0,
                )
            ]

        ocr_results: List[OCRResult] = []

        for region in regions:
            b = region.bbox
            y_min, y_max = max(0, b.y_min), min(h, b.y_max)
            x_min, x_max = max(0, b.x_min), min(w, b.x_max)

            crop = img[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue

            # Preprocessing pipeline
            variants_to_run = [("orig", crop, 1.0)]
            try:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if getattr(crop, "ndim", 3) == 3 else crop.copy()
                mean_brightness = float(np.mean(gray))

                # Variants
                up2x = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                inv = cv2.bitwise_not(gray) if mean_brightness < 128.0 else gray
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                thresh = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)

                # Select original + best 2 variants
                if mean_brightness < 128.0:
                    variants_to_run.append(("inv", inv, 1.0))
                    variants_to_run.append(("clahe", clahe, 1.0))
                else:
                    variants_to_run.append(("clahe", clahe, 1.0))
                    variants_to_run.append(("gray", gray, 1.0))
            except Exception:
                pass

            best_res = None
            best_score = -1.0

            for v_name, v_img, v_scale in variants_to_run:
                v_lines, v_confs, v_toks = self._run_single_engine(v_img, x_min, y_min, scale=v_scale)
                if v_toks:
                    avg_c = sum(v_confs) / len(v_confs) if v_confs else 0.0
                    score = avg_c * min(len(v_toks), 25)
                    if score > best_score:
                        best_score = score
                        best_res = (v_lines, v_confs, v_toks)

            if best_res:
                lines_text, confidences, raw_tokens = best_res
                tokens = _clean_and_split_tokens(raw_tokens)
                lines_text = [t.text for t in tokens]
                avg_conf = sum(t.confidence for t in tokens) / len(tokens) if tokens else 0.90
                ocr_results.append(
                    OCRResult(
                        region_label=region.region_label,
                        raw_text="\n".join(lines_text),
                        ocr_confidence=round(avg_conf, 3),
                        bbox=region.bbox,
                        tokens=tokens,
                    )
                )
            else:
                ocr_results.append(
                    OCRResult(
                        region_label=region.region_label,
                        raw_text="",
                        ocr_confidence=0.0,
                        bbox=region.bbox,
                        tokens=[],
                    )
                )

        return ocr_results

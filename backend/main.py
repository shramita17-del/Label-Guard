import os
import shutil
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from typing import Optional

from backend.schemas import ProductCategory, NormalizedScan, ComplianceReport, CheckStatus
from backend.ocr_service.detector import YoloDetector
from backend.ocr_service.extractor import OCRExtractor
from backend.parser_service.normalizers import normalize_mrp, normalize_net_quantity, normalize_dates, normalize_fiber_composition
from backend.parser_service.geometry import analyze_veg_mark
from backend.rule_engine.validators.common import validate_mrp, validate_net_quantity, validate_date_of_mfg, validate_country_of_origin
from backend.rule_engine.validators.apparel_textile import validate_fiber_composition as val_fiber
from backend.rule_engine.validators.food_beverage import validate_veg_nonveg_mark as val_veg
from backend.rule_engine.fssai_verifier import verify_fssai_license
from backend.report_service.annotator import annotate_image
from backend.report_service.pdf_generator import generate_pdf_report

app = FastAPI(title="LabelGuard AI", description="Automated Packaging Compliance Verification Engine (SIH 26034)")

# Lazy ML load to prevent app crash on startup if weights are missing/downloading
ml_components = {"detector": None, "extractor": None}

def get_ml():
    if ml_components["detector"] is None:
        try:
            ml_components["detector"] = YoloDetector("yolov8n.pt")
        except Exception as e:
            print(f"YOLO detector note: {e}")
            fallback_det = YoloDetector.__new__(YoloDetector)
            fallback_det.model = None
            ml_components["detector"] = fallback_det
            
    if ml_components["extractor"] is None:
        try:
            ml_components["extractor"] = OCRExtractor()
        except Exception as e:
            print(f"OCR extractor note: {e}")
            fallback_ext = OCRExtractor.__new__(OCRExtractor)
            fallback_ext.ocr = None
            ml_components["extractor"] = fallback_ext
            
    return ml_components["detector"], ml_components["extractor"]
    
def auto_route_category(ocr_results) -> ProductCategory:
    """Flow E: Lightweight Heuristic Classifier based on checklist_matrix.json hints"""
    raw_text_concat = " ".join([r.raw_text.lower() for r in ocr_results])
    labels_concat = " ".join([r.region_label.lower() for r in ocr_results])
    
    if "fssai" in raw_text_concat or "fssai" in labels_concat or "veg_mark" in labels_concat or "nutrition_table" in labels_concat:
        return ProductCategory.FOOD_BEVERAGE
    if "cotton" in raw_text_concat or "polyester" in raw_text_concat or "%" in raw_text_concat or "size" in labels_concat:
        return ProductCategory.APPAREL_TEXTILE
        
    return ProductCategory.GENERAL_RETAIL

@app.post("/scan", response_model=ComplianceReport)
async def process_scan(file: UploadFile = File(...)):
    """Flow A: Full Physical Package Scan Pipeline"""
    detector, extractor = get_ml()
        
    scan_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(os.getcwd(), "scratch")
    os.makedirs(temp_dir, exist_ok=True)
    img_name = file.filename if file.filename else f"scan_{scan_id}.jpg"
    img_path = os.path.join(temp_dir, img_name)
    
    with open(img_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    # 1. Image -> Regions -> OCR (Phase 2, Steps 1-2)
    regions = detector.detect_regions(img_path) if detector else []
    ocr_results = extractor.extract_text(img_path, regions) if extractor else []
    
    # 2. Auto Routing (Flow E)
    category = auto_route_category(ocr_results)
    
    # 3. Pydantic Normalization Mapping (Phase 2, Step 3)
    # Using region labels and raw text to extract specific fields
    text_map = {r.region_label: r.raw_text for r in ocr_results}
    conf_map = {r.region_label: r.ocr_confidence for r in ocr_results}
    
    # Fallback to full text search if specific labeled regions were not distinct
    all_text = " ".join([r.raw_text for r in ocr_results])
    
    mrp_text = text_map.get("mrp_region", all_text)
    net_qty_text = text_map.get("net_quantity_region", all_text)
    fssai_text = text_map.get("fssai_region", all_text)
    coo_text = text_map.get("coo_region", all_text)
    fiber_text = text_map.get("fiber_region", all_text)
    date_text = text_map.get("date_region", all_text)
    
    scan = NormalizedScan(
        scan_id=scan_id, 
        category=category,
        mrp=normalize_mrp(mrp_text),
        net_quantity=normalize_net_quantity(net_qty_text),
        fssai_license_number=fssai_text if fssai_text != all_text else None,
        country_of_origin=coo_text if coo_text != all_text else None,
        fiber_composition=normalize_fiber_composition(fiber_text) if category == ProductCategory.APPAREL_TEXTILE else None
    )
    
    dates = normalize_dates(date_text)
    if dates:
        scan.dates.append(dates)
        
    if category == ProductCategory.FOOD_BEVERAGE:
        # Pass bbox geometry for OpenCV analysis if region detected
        veg_region = next((r for r in regions if r.region_label == "veg_mark"), None)
        if veg_region:
            b = veg_region.bbox
            scan.veg_nonveg_mark = analyze_veg_mark(img_path, {"x_min": b.x_min, "y_min": b.y_min, "x_max": b.x_max, "y_max": b.y_max})
            
    # 4. Rule Engine Execution (Phase 2, Step 4 + Phase 4 FSSAI)
    results = []
    
    # Tier 1 Common Checks
    results.append(validate_mrp(scan, conf_map.get("mrp_region", 0.95), "01", "Rule_6_1_e"))
    results.append(validate_net_quantity(scan, conf_map.get("net_quantity_region", 0.95), "02", "Rule_6_1_c"))
    results.append(validate_date_of_mfg(scan, conf_map.get("date_region", 0.95), "03", "Rule_6_1_d"))
    results.append(validate_country_of_origin(scan, conf_map.get("coo_region", 0.95), "04", "Rule_6_1_aa"))
    
    # Tier 1 Category Specific Checks + Tier 2 FSSAI
    if category == ProductCategory.APPAREL_TEXTILE:
        results.append(val_fiber(scan, conf_map.get("fiber_region", 0.95), "A02", "Textile_Standards"))
    elif category == ProductCategory.FOOD_BEVERAGE:
        results.append(val_veg(scan, conf_map.get("veg_mark", 0.95), "F12", "Veg_Mark_Rules"))
        results.append(verify_fssai_license(scan, conf_map.get("fssai_region", 0.95)))
        
    # 5. Determine Overall Verdict
    # If any BLOCKING is FAIL -> FAIL, else if any REVIEW -> REVIEW
    overall = CheckStatus.PASS
    for r in results:
        if r.status == CheckStatus.FAIL and r.severity.value == "blocking":
            overall = CheckStatus.FAIL
            break
        elif r.status == CheckStatus.REVIEW:
            if overall != CheckStatus.FAIL:
                overall = CheckStatus.REVIEW
                
    report = ComplianceReport(
        scan_id=scan.scan_id,
        category=scan.category,
        overall_verdict=overall,
        check_results=results,
        generated_at=datetime.utcnow().isoformat()
    )
    
    # 6. Generate Outputs (Phase 3)
    report.annotated_image_base64 = annotate_image(img_path, report)
    
    pdf_dir = os.path.join(os.getcwd(), "reports", "generated")
    os.makedirs(pdf_dir, exist_ok=True)
    report_out_path = os.path.join(pdf_dir, f"{scan_id}.pdf")
    generate_pdf_report(report, os.path.join(os.getcwd(), "reports"), report_out_path)
    
    # Clean up upload
    if os.path.exists(img_path):
        try:
            os.remove(img_path)
        except Exception:
            pass
            
    return report

# Mount the frontend directory to serve the UI at the root url
app.mount("/", StaticFiles(directory=os.path.join(os.getcwd(), "frontend"), html=True), name="frontend")

import os
import shutil
import uuid
import re
from datetime import datetime, timezone
from typing import Optional, Dict, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.schemas import (
    ProductCategory, NormalizedScan, ComplianceReport,
    CheckStatus, CheckResult, Severity, AuthenticityDetails,
    UserRole, LoginRequest, OTPRequest, OTPVerifyRequest,
    UserProfile, AuthResponse
)
from backend.rule_engine.engine import evaluate_packaging_compliance
from backend.rule_engine.authenticity import evaluate_image_authenticity
from backend.ocr_service.detector import YoloDetector
from backend.ocr_service.extractor import OCRExtractor
from backend.parser_service.normalizers import (
    normalize_mrp, normalize_net_quantity, normalize_dates, normalize_fiber_composition
)
from backend.parser_service.geometry import analyze_veg_mark
from backend.report_service.annotator import annotate_image
from backend.report_service.pdf_generator import generate_pdf_report

app = FastAPI(
    title="LabelGuard AI",
    description="Automated Packaging Compliance Verification Engine (SIH 26034)"
)

# ─────────────────────────────────────────────────────────────
# IN-MEMORY AUTHENTICATION & ROLE ENFORCEMENT
# ─────────────────────────────────────────────────────────────

ACTIVE_SESSIONS: Dict[str, UserProfile] = {}
ACTIVE_OTPS: Dict[str, str] = {
    "customer@labelguard.com": "482190",
    "gov-lm-4821": "482190"
}

AUTHORIZED_USERS = {
    UserRole.CONSUMER: {
        "customer@labelguard.com": {
            "password": "Customer@123",
            "name": "Priya Sharma",
            "id": "CUST-8831",
            "email": "customer@labelguard.com",
            "department": "Consumer Protection Portal",
            "jurisdiction": "National"
        }
    },
    UserRole.INSPECTOR: {
        "gov-lm-4821": {
            "password": "Inspector@2026",
            "name": "Insp. Rajesh Kumar",
            "id": "GOV-LM-4821",
            "email": "rajesh.kumar@gov.in",
            "department": "Legal Metrology (Dept of Consumer Affairs)",
            "jurisdiction": "Northern Circle - Zone 4 (Delhi NCR)"
        }
    }
}


def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[UserProfile]:
    """Helper to extract and validate active user from Authorization header"""
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "").strip()
    return ACTIVE_SESSIONS.get(token)


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """
    Role-enforced authentication endpoint.
    Verifies user credentials strictly matching their selected role on the backend.
    """
    role = request.role
    username_clean = request.username.strip().lower()

    role_users = AUTHORIZED_USERS.get(role, {})
    user_record = role_users.get(username_clean)

    if not user_record or user_record["password"] != request.password:
        # Check if they attempted to log in with an ID belonging to the other role
        other_role = UserRole.INSPECTOR if role == UserRole.CONSUMER else UserRole.CONSUMER
        other_users = AUTHORIZED_USERS.get(other_role, {})
        if username_clean in other_users:
            raise HTTPException(
                status_code=403,
                detail=f"Access Denied: This credential is registered for the '{other_role.value}' role, not '{role.value}'."
            )
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials. Please verify your username and password."
        )

    # Generate session token
    token = f"lg_sess_{uuid.uuid4().hex}"
    profile = UserProfile(
        id=user_record["id"],
        name=user_record["name"],
        email=user_record.get("email"),
        role=role,
        department=user_record.get("department"),
        jurisdiction=user_record.get("jurisdiction")
    )
    ACTIVE_SESSIONS[token] = profile

    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user=profile,
        message=f"Successfully authenticated as {profile.name} ({role.value.capitalize()})"
    )


@app.get("/api/auth/login")
async def login_get_not_allowed():
    """Explicitly reject GET requests on authentication endpoint with 405 Method Not Allowed"""
    raise HTTPException(
        status_code=405,
        detail="Method Not Allowed: Authentication requires HTTP POST with valid JSON credentials."
    )


@app.post("/api/auth/otp/request")
async def request_otp(request: OTPRequest):
    """Generates and delivers a one-time passcode for mobile/email sign-in"""
    ident = request.identifier.strip().lower()
    otp_code = "482190"  # Deterministic test code for seamless demonstration
    ACTIVE_OTPS[ident] = otp_code
    return {
        "status": "success",
        "message": f"One-Time Passcode sent to {request.identifier}",
        "demo_otp": otp_code
    }


@app.post("/api/auth/otp/verify", response_model=AuthResponse)
async def verify_otp(request: OTPVerifyRequest):
    """Verifies OTP and generates an authenticated session token"""
    ident = request.identifier.strip().lower()
    stored_otp = ACTIVE_OTPS.get(ident, "482190")

    if request.otp.strip() != stored_otp:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP code.")

    role = request.role or UserRole.CONSUMER
    name = "Inspector (OTP Verified)" if role == UserRole.INSPECTOR else "Consumer (OTP Verified)"
    user_id = f"OTP-{uuid.uuid4().hex[:6].upper()}"

    token = f"lg_sess_{uuid.uuid4().hex}"
    profile = UserProfile(
        id=user_id,
        name=name,
        email=request.identifier if "@" in request.identifier else None,
        role=role,
        department="Enforcement Wing" if role == UserRole.INSPECTOR else "Citizen Safety",
        jurisdiction="Northern Circle - Zone 4 (Delhi NCR)" if role == UserRole.INSPECTOR else "National"
    )
    ACTIVE_SESSIONS[token] = profile

    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user=profile,
        message="OTP Verified successfully."
    )


@app.get("/api/auth/me", response_model=UserProfile)
async def get_me(authorization: Optional[str] = Header(None)):
    """Fetches currently active authenticated user profile"""
    user = get_current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or not authenticated.")
    return user


@app.post("/api/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    """Invalidates active session token"""
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        ACTIVE_SESSIONS.pop(token, None)
    return {"status": "success", "message": "Logged out successfully."}


# ─────────────────────────────────────────────────────────────
# ML PIPELINE & COMPLIANCE VERIFICATION
# ─────────────────────────────────────────────────────────────

# Lazy ML load to prevent app crash on startup if weights are missing/downloading
ml_components = {"detector": None, "extractor": None}


def get_ml():
    if ml_components["detector"] is None:
        try:
            ml_components["detector"] = YoloDetector("yolov8n.pt")
        except Exception as e:
            print(f"YOLO detector notice: {e}")
            fallback_det = YoloDetector.__new__(YoloDetector)
            fallback_det.model = None
            ml_components["detector"] = fallback_det

    if ml_components["extractor"] is None:
        try:
            ml_components["extractor"] = OCRExtractor()
        except Exception as e:
            print(f"OCR extractor notice: {e}")
            fallback_ext = OCRExtractor.__new__(OCRExtractor)
            fallback_ext.ocr = None
            fallback_ext.winocr = None
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
async def process_scan(file: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    """
    Flow A: Full Physical Package Scan Pipeline:
    1. Upload & save image to scratch
    2. Detect packaging layout regions (YOLO)
    3. OCR text extraction & confidence scoring
    4. Auto-route category (Food/Beverage, Apparel/Textile, General Retail)
    5. Multi-signal AI Image Authenticity detection (Method 1 + Semantic + Barcode + DL Arbiter)
    6. Normalize key Legal Metrology attributes (MRP, Net Qty, Dates, COO, Manufacturer)
    7. Execute shared compliance rule engine with dynamic Supabase / SQLite metadata
    8. Calculate overall compliance verdict including AI authenticity directive
    9. Generate annotated visual & downloadable PDF report
    """
    detector, extractor = get_ml()

    scan_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(os.getcwd(), "scratch")
    os.makedirs(temp_dir, exist_ok=True)
    img_name = file.filename if file.filename else f"scan_{scan_id}.jpg"
    img_path = os.path.join(temp_dir, f"{scan_id}_{img_name}")

    try:
        with open(img_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 1. Image -> Regions -> OCR (Phase 2, Steps 1-2)
        regions = detector.detect_regions(img_path) if detector else []
        ocr_results = extractor.extract_text(img_path, regions) if extractor else []

        # 2. Auto Routing (Flow E)
        category = auto_route_category(ocr_results)

        # 3. Pydantic Normalization Mapping (Phase 2, Step 3)
        text_map = {r.region_label: r.raw_text for r in ocr_results}
        conf_map = {r.region_label: r.ocr_confidence for r in ocr_results}
        all_text = " ".join([r.raw_text for r in ocr_results])

        # 4. Multi-Signal Image Authenticity Verification (Machine C: Method 1 + Fixes)
        auth_check, auth_details = evaluate_image_authenticity(img_path, extracted_text=all_text)

        # 5. Extract fields from labeled regions and full text
        mrp_text = text_map.get("mrp_region", all_text)
        net_qty_text = text_map.get("net_quantity_region", all_text)

        fssai_val = text_map.get("fssai_region")
        if not fssai_val:
            f_match = re.search(r'\b([12]\d{13})\b', all_text)
            if f_match:
                fssai_val = f_match.group(1)

        coo_val = text_map.get("coo_region")
        if not coo_val:
            c_match = re.search(r'(?:country of origin|made in|origin)\s*[:\-]?\s*([a-zA-Z\s]+)', all_text, re.IGNORECASE)
            if c_match:
                coo_val = c_match.group(1).strip()
            elif "india" in all_text.lower():
                coo_val = "India"

        # Manufacturer name extraction heuristic
        mfg_name = text_map.get("manufacturer_region")
        if not mfg_name:
            m_match = re.search(r'(?:mfg by|manufactured by|packed by|marketed by|mfd by)\s*[:\-]?\s*([a-zA-Z0-9\s,\.]+)', all_text, re.IGNORECASE)
            if m_match:
                mfg_name = m_match.group(1).strip()[:80]

        # Pincode extraction heuristic
        pincode_val = None
        pin_match = re.search(r'\b([1-9][0-9]{5})\b', all_text)
        if pin_match:
            pincode_val = pin_match.group(1)

        fiber_text = text_map.get("fiber_region", all_text)
        date_text = text_map.get("date_region", all_text)

        scan = NormalizedScan(
            scan_id=scan_id,
            category=category,
            manufacturer_name=mfg_name,
            pincode=pincode_val,
            mrp=normalize_mrp(mrp_text),
            net_quantity=normalize_net_quantity(net_qty_text),
            fssai_license_number=fssai_val,
            country_of_origin=coo_val,
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
            else:
                scan.veg_nonveg_mark = analyze_veg_mark(img_path)

        # 6. Shared rule engine execution (Machine B: Supabase / SQLite metadata)
        rule_results = evaluate_packaging_compliance(scan, conf_map)
        results = [auth_check] + rule_results

        # 7. Determine overall verdict
        overall = CheckStatus.PASS
        for result in results:
            if (
                result.status == CheckStatus.FAIL
                and result.severity == Severity.BLOCKING
            ):
                overall = CheckStatus.FAIL
                break
            elif result.status == CheckStatus.REVIEW:
                if overall != CheckStatus.FAIL:
                    overall = CheckStatus.REVIEW

        # Mandatory Compliance Directive: AI-generated/synthetic media is non-compliant by law
        if auth_details and auth_details.is_ai_generated:
            overall = CheckStatus.FAIL
        elif auth_details and (
            (auth_details.ai_confidence_percentage >= 40 and auth_details.ai_confidence_percentage <= 60)
            or auth_details.status == CheckStatus.REVIEW
            or (auth_details.classification_label and "MAYBE AI" in auth_details.classification_label)
        ):
            if overall != CheckStatus.FAIL:
                overall = CheckStatus.REVIEW

        report = ComplianceReport(
            scan_id=scan.scan_id,
            category=scan.category,
            overall_verdict=overall,
            check_results=results,
            generated_at=datetime.now(timezone.utc).isoformat(),
            authenticity=auth_details
        )

        # 8. Generate Outputs (Phase 3)
        report.annotated_image_base64 = annotate_image(img_path, report)

        pdf_dir = os.path.join(os.getcwd(), "reports", "generated")
        os.makedirs(pdf_dir, exist_ok=True)
        report_out_path = os.path.join(pdf_dir, f"{scan_id}.pdf")
        final_report_path = generate_pdf_report(report, os.path.join(os.getcwd(), "reports"), report_out_path)
        if final_report_path and os.path.exists(final_report_path):
            report.report_url = f"/reports/generated/{os.path.basename(final_report_path)}"

        return report

    finally:
        # Clean up temporary uploaded file safely
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception:
                pass


@app.get("/reports/generated/{filename}")
async def get_generated_report(filename: str):
    file_path = os.path.join(os.getcwd(), "reports", "generated", filename)
    if os.path.exists(file_path):
        media_type = "application/pdf" if filename.endswith(".pdf") else "text/html"
        return FileResponse(file_path, media_type=media_type)
    raise HTTPException(status_code=404, detail="Report not found")


# Mount the frontend directory to serve the UI at the root url
app.mount("/", StaticFiles(directory=os.path.join(os.getcwd(), "frontend"), html=True), name="frontend")

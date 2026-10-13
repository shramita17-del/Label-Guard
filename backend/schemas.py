from __future__ import annotations
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────

class ProductCategory(str, Enum):
    FOOD_BEVERAGE = "food_beverage"
    APPAREL_TEXTILE = "apparel_textile"
    GENERAL_RETAIL = "general_retail"

class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"
    EXEMPT = "EXEMPT"
    NOT_APPLICABLE = "N/A"

class Severity(str, Enum):
    BLOCKING = "blocking"
    MAJOR = "major"
    MINOR = "minor"

# ─────────────────────────────────────────────────────────────
# STAGE 2/3 OUTPUT — Region Detection + OCR
# ─────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class DetectedRegion(BaseModel):
    region_label: str
    bbox: BoundingBox
    yolo_confidence: float = Field(ge=0, le=1)

class OCRResult(BaseModel):
    region_label: str
    raw_text: str
    ocr_confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox

class OCRPayload(BaseModel):
    image_id: str
    detected_regions: List[DetectedRegion]
    ocr_results: List[OCRResult]
    overall_confidence: float = Field(ge=0, le=1)

# ─────────────────────────────────────────────────────────────
# STAGE 4 OUTPUT — Normalized structured data
# ─────────────────────────────────────────────────────────────

class NetQuantity(BaseModel):
    value: float
    unit: str
    raw_text: str

class MRPDeclaration(BaseModel):
    amount: float
    currency_symbol_found: bool
    tax_inclusive_phrase_found: bool
    raw_text: str

class DateDeclaration(BaseModel):
    date_type: str
    month: Optional[int] = None
    year: Optional[int] = None
    raw_text: str
    is_valid_format: bool

class ConsumerCare(BaseModel):
    phone_found: bool
    email_found: bool
    address_found: bool
    pincode_found: bool
    raw_text: str

class FiberComposition(BaseModel):
    components: dict[str, float]
    sums_to_100: bool

class VegNonVegMark(BaseModel):
    shape_detected: Optional[str] = None
    color_detected: Optional[str] = None
    measured_mm: Optional[float] = None
    required_mm: Optional[float] = None
    pdp_area_cm2: Optional[float] = None

class NormalizedScan(BaseModel):
    scan_id: str
    category: ProductCategory
    manufacturer_name: Optional[str] = None
    manufacturer_address: Optional[str] = None
    pincode: Optional[str] = None
    country_of_origin: Optional[str] = None
    generic_name: Optional[str] = None
    net_quantity: Optional[NetQuantity] = None
    mrp: Optional[MRPDeclaration] = None
    dates: List[DateDeclaration] = []
    consumer_care: Optional[ConsumerCare] = None
    fiber_composition: Optional[FiberComposition] = None
    veg_nonveg_mark: Optional[VegNonVegMark] = None
    fssai_license_number: Optional[str] = None
    pdp_area_cm2: Optional[float] = None

# ─────────────────────────────────────────────────────────────
# STAGE 5 OUTPUT — Rule Engine results
# ─────────────────────────────────────────────────────────────

class CheckResult(BaseModel):
    check_id: str
    check_name: str
    rule_ref: str
    severity: Severity
    status: CheckStatus
    evidence_text: Optional[str] = None
    evidence_bbox: Optional[BoundingBox] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None

class AuthenticityDetails(BaseModel):
    is_ai_generated: bool = False
    classification_label: str = "AUTHENTIC PHYSICAL CAPTURE" # "AI GENERATED (SYNTHETIC MEDIA)", "AUTHENTIC PHYSICAL CAPTURE", "UNVERIFIED / DIGITAL RENDER"
    authenticity_score: float = 1.0 # 0.0 to 1.0 (1.0 = authentic photo, 0.0 = AI generated)
    ai_confidence_percentage: int = 0 # 0% to 100%
    status: CheckStatus # PASS, REVIEW, FAIL
    camera_hardware: Optional[str] = None # e.g. "Apple iPhone 14"
    ai_signatures: List[str] = [] # detected generator signatures or prompts
    spectral_artifact_score: float = 0.0
    noise_residual_score: float = 0.0
    deep_learning_score: Optional[float] = None # 0.0 to 1.0 (Deep Learning Fake / AI Probability)
    deep_learning_model: Optional[str] = None # e.g. "ViT (umm-maybe/AI-image-detector)"
    mockup_disclaimers: List[str] = [] # detected packaging mockup/demo disclaimers
    barcode_status: Optional[str] = None # e.g. "VALID_EAN13", "INVALID_CHECKSUM", "HALLUCINATED_PATTERN"
    barcode_value: Optional[str] = None # extracted barcode digits
    verdict_summary: str

class ComplianceReport(BaseModel):
    scan_id: str
    category: ProductCategory
    overall_verdict: CheckStatus
    check_results: List[CheckResult]
    annotated_image_base64: Optional[str] = None
    generated_at: str
    report_url: Optional[str] = None
    authenticity: Optional[AuthenticityDetails] = None

# ─────────────────────────────────────────────────────────────
# E-COMMERCE MISMATCH CHECK — dual input comparison
# ─────────────────────────────────────────────────────────────

class DigitalListingJSON(BaseModel):
    source_url: str
    manufacturer_name: Optional[str] = None
    country_of_origin: Optional[str] = None
    generic_name: Optional[str] = None
    net_quantity: Optional[NetQuantity] = None
    mrp: Optional[MRPDeclaration] = None
    expiry_or_best_before: Optional[DateDeclaration] = None
    consumer_care: Optional[ConsumerCare] = None

class MismatchField(BaseModel):
    field_name: str
    digital_value: Optional[str] = None
    physical_value: Optional[str] = None
    match_method: str
    match_score: Optional[float] = None
    is_match: bool

class MismatchReport(BaseModel):
    scan_id: str
    digital_listing_url: str
    fields_compared: List[MismatchField]
    overall_mismatch_detected: bool

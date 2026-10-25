from typing import Optional, Dict, Any
from backend.schemas import NormalizedScan, CheckResult, CheckStatus, Severity

CONFIDENCE_THRESHOLD = 0.70
_OCR_FLAG = "__ocr_available__"

def _ocr_status(conf_map: dict) -> str:
    if conf_map.get(_OCR_FLAG, 1.0) == 0.0:
        return "OCR engine unavailable — text could not be read from this image."
    return "Field not found in OCR-extracted text. Verify the label is clearly visible."

def validate_mrp(
    scan: NormalizedScan,
    field_confidence: float,
    check_id: str,
    rule_ref: str,
    conf_map: Optional[dict] = None,
    digital_claims: Optional[dict] = None,
) -> CheckResult:
    conf_map = conf_map or {}
    mrp = scan.mrp

    # 1. OCR confidence check
    if field_confidence < 0.60:
        return CheckResult(
            check_id=check_id,
            check_name="MRP",
            rule_ref=rule_ref,
            severity=Severity.BLOCKING,
            status=CheckStatus.UNCERTAIN,
            confidence=field_confidence,
            reason="OCR confidence below threshold, field ambiguous.",
        )
    elif field_confidence < 0.85:
        return CheckResult(
            check_id=check_id,
            check_name="MRP",
            rule_ref=rule_ref,
            severity=Severity.BLOCKING,
            status=CheckStatus.REVIEW,
            confidence=field_confidence,
            reason="OCR confidence below 85%. Manual review required.",
        )

    # 2. Field absence
    if not mrp:
        return CheckResult(
            check_id=check_id,
            check_name="MRP",
            rule_ref=rule_ref,
            severity=Severity.BLOCKING,
            status=CheckStatus.FAIL,
            confidence=field_confidence,
            reason=f"MRP declaration not found on package. {_ocr_status(conf_map)}",
        )

    # 3. Cross-field check vs digital listing claims
    if digital_claims and digital_claims.get("mrp") is not None:
        try:
            d_mrp = float(digital_claims["mrp"])
            if abs(d_mrp - mrp.amount) > 0.01:
                return CheckResult(
                    check_id=check_id,
                    check_name="MRP",
                    rule_ref=rule_ref,
                    severity=Severity.BLOCKING,
                    status=CheckStatus.FAIL,
                    evidence_text=f"Physical package MRP: Rs. {mrp.amount:.2f} | Digital listing MRP: Rs. {d_mrp:.2f}",
                    confidence=field_confidence,
                    matched_span={"text": mrp.raw_text},
                    reason=f"Cross-field Mismatch: Physical label MRP (Rs. {mrp.amount:.2f}) differs from digital claim (Rs. {d_mrp:.2f}).",
                )
        except (ValueError, TypeError):
            pass

    # 4. Deterministic Legal Metrology Rule 6(1)(e)
    matched_span = {"text": mrp.raw_text}
    if mrp.currency_symbol_found and mrp.tax_inclusive_phrase_found:
        return CheckResult(
            check_id=check_id,
            check_name="MRP",
            rule_ref=rule_ref,
            severity=Severity.BLOCKING,
            status=CheckStatus.PASS,
            evidence_text=mrp.raw_text,
            confidence=field_confidence,
            matched_span=matched_span,
            reason=f"MRP declaration verified: Rs. {mrp.amount:.2f} with tax-inclusive phrase.",
        )
    else:
        missing = []
        if not mrp.currency_symbol_found:
            missing.append("currency symbol (Rs./₹)")
        if not mrp.tax_inclusive_phrase_found:
            missing.append("'inclusive of all taxes' phrase")
        return CheckResult(
            check_id=check_id,
            check_name="MRP",
            rule_ref=rule_ref,
            severity=Severity.BLOCKING,
            status=CheckStatus.FAIL,
            evidence_text=mrp.raw_text,
            confidence=field_confidence,
            matched_span=matched_span,
            reason=f"MRP found but missing mandatory declaration: {', '.join(missing)}.",
        )


def validate_net_quantity(
    scan: NormalizedScan,
    field_confidence: float,
    check_id: str,
    rule_ref: str,
    conf_map: Optional[dict] = None,
    digital_claims: Optional[dict] = None,
) -> CheckResult:
    conf_map = conf_map or {}
    qty = scan.net_quantity

    if field_confidence < 0.60:
        return CheckResult(
            check_id=check_id,
            check_name="Net Quantity",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.UNCERTAIN,
            confidence=field_confidence,
            reason="OCR confidence below threshold, field ambiguous.",
        )
    elif field_confidence < 0.85:
        return CheckResult(
            check_id=check_id,
            check_name="Net Quantity",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.REVIEW,
            confidence=field_confidence,
            reason=f"OCR confidence ({field_confidence:.2f}) below 85%.",
        )

    if not qty:
        return CheckResult(
            check_id=check_id,
            check_name="Net Quantity",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.FAIL,
            confidence=field_confidence,
            reason=f"Net quantity not found on package. {_ocr_status(conf_map)}",
        )

    # Cross-field check vs digital claims
    if digital_claims and digital_claims.get("net_quantity") is not None:
        try:
            d_qty = float(digital_claims["net_quantity"])
            if abs(d_qty - qty.value) > 0.01:
                return CheckResult(
                    check_id=check_id,
                    check_name="Net Quantity",
                    rule_ref=rule_ref,
                    severity=Severity.MAJOR,
                    status=CheckStatus.FAIL,
                    evidence_text=f"Physical package: {qty.value} {qty.unit} | Digital claim: {d_qty}",
                    confidence=field_confidence,
                    matched_span={"text": qty.raw_text},
                    reason=f"Cross-field Mismatch: Physical quantity ({qty.value} {qty.unit}) differs from digital claim ({d_qty}).",
                )
        except (ValueError, TypeError):
            pass

    # Deterministic Rule 6(1)(c): Standard metric units (g, kg, ml, l, number)
    valid_units = ["g", "kg", "ml", "l", "number", "pcs", "m", "cm"]
    matched_span = {"text": qty.raw_text}
    if qty.unit.lower() in valid_units:
        return CheckResult(
            check_id=check_id,
            check_name="Net Quantity",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.PASS,
            evidence_text=qty.raw_text,
            confidence=field_confidence,
            matched_span=matched_span,
            reason=f"Valid standard unit '{qty.unit}' detected ({qty.value:g} {qty.unit}).",
        )
    else:
        return CheckResult(
            check_id=check_id,
            check_name="Net Quantity",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.FAIL,
            evidence_text=qty.raw_text,
            confidence=field_confidence,
            matched_span=matched_span,
            reason=f"Non-standard unit '{qty.unit}' used. Legal Metrology requires g, kg, ml, l, or number.",
        )


def validate_date_of_mfg(
    scan: NormalizedScan,
    field_confidence: float,
    check_id: str,
    rule_ref: str,
    conf_map: Optional[dict] = None,
    digital_claims: Optional[dict] = None,
) -> CheckResult:
    conf_map = conf_map or {}
    dates = [d for d in scan.dates if d.date_type == "mfg"]

    if field_confidence < 0.60:
        return CheckResult(
            check_id=check_id,
            check_name="Date of Mfg",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.UNCERTAIN,
            confidence=field_confidence,
            reason="OCR confidence below threshold, field ambiguous.",
        )
    elif field_confidence < 0.85:
        return CheckResult(
            check_id=check_id,
            check_name="Date of Mfg",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.REVIEW,
            confidence=field_confidence,
            reason=f"OCR confidence ({field_confidence:.2f}) below 85%.",
        )

    if not dates:
        return CheckResult(
            check_id=check_id,
            check_name="Date of Mfg",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.FAIL,
            confidence=field_confidence,
            reason=f"Manufacturing date not found on package. {_ocr_status(conf_map)}",
        )

    date_obj = dates[0]
    matched_span = {"text": date_obj.raw_text}
    if date_obj.is_valid_format:
        return CheckResult(
            check_id=check_id,
            check_name="Date of Mfg",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.PASS,
            evidence_text=date_obj.raw_text,
            confidence=field_confidence,
            matched_span=matched_span,
            reason=f"Valid manufacturing date format detected: {date_obj.raw_text}",
        )
    else:
        return CheckResult(
            check_id=check_id,
            check_name="Date of Mfg",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.FAIL,
            evidence_text=date_obj.raw_text,
            confidence=field_confidence,
            matched_span=matched_span,
            reason=f"Date text found but format is invalid: '{date_obj.raw_text}'. Expected MM/YYYY or DD-MM-YYYY.",
        )


def validate_country_of_origin(
    scan: NormalizedScan,
    field_confidence: float,
    check_id: str,
    rule_ref: str,
    conf_map: Optional[dict] = None,
    digital_claims: Optional[dict] = None,
) -> CheckResult:
    conf_map = conf_map or {}
    coo = scan.country_of_origin

    if field_confidence < 0.60:
        return CheckResult(
            check_id=check_id,
            check_name="Country of Origin",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.UNCERTAIN,
            confidence=field_confidence,
            reason="OCR confidence below threshold, field ambiguous.",
        )
    elif field_confidence < 0.85:
        return CheckResult(
            check_id=check_id,
            check_name="Country of Origin",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.REVIEW,
            confidence=field_confidence,
            reason="OCR confidence below threshold.",
        )

    if not coo:
        if conf_map.get(_OCR_FLAG, 1.0) == 0.0:
            return CheckResult(
                check_id=check_id,
                check_name="Country of Origin",
                rule_ref=rule_ref,
                severity=Severity.MAJOR,
                status=CheckStatus.FAIL,
                confidence=field_confidence,
                reason=f"Country of origin missing. {_ocr_status(conf_map)}",
            )
        return CheckResult(
            check_id=check_id,
            check_name="Country of Origin",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.REVIEW,
            confidence=field_confidence,
            reason=f"Country of origin missing. {_ocr_status(conf_map)} Requires manual verification if product is imported.",
        )

    # Cross-field check vs digital claims
    if digital_claims and digital_claims.get("country_of_origin"):
        d_coo = digital_claims["country_of_origin"].strip().lower()
        if d_coo not in coo.lower() and coo.lower() not in d_coo:
            return CheckResult(
                check_id=check_id,
                check_name="Country of Origin",
                rule_ref=rule_ref,
                severity=Severity.BLOCKING,
                status=CheckStatus.FAIL,
                evidence_text=f"Physical: {coo} | Digital: {digital_claims['country_of_origin']}",
                confidence=field_confidence,
                matched_span={"text": coo},
                reason=f"Cross-field Mismatch: Physical country ({coo}) differs from digital claim ({digital_claims['country_of_origin']}).",
            )

    return CheckResult(
        check_id=check_id,
        check_name="Country of Origin",
        rule_ref=rule_ref,
        severity=Severity.MAJOR,
        status=CheckStatus.PASS,
        evidence_text=coo,
        confidence=field_confidence,
        matched_span={"text": coo},
        reason=f"Country of origin declared: {coo}",
    )


def validate_manufacturer(
    scan: NormalizedScan,
    field_confidence: float,
    check_id: str,
    rule_ref: str,
    conf_map: Optional[dict] = None,
    digital_claims: Optional[dict] = None,
) -> CheckResult:
    conf_map = conf_map or {}

    if field_confidence < 0.60:
        return CheckResult(
            check_id=check_id,
            check_name="Manufacturer/Packer",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.UNCERTAIN,
            confidence=field_confidence,
            reason="OCR confidence below threshold, field ambiguous.",
        )

    if not scan.manufacturer_name:
        if "manufacturer_region" in conf_map and field_confidence < 0.85:
            return CheckResult(
                check_id=check_id,
                check_name="Manufacturer/Packer",
                rule_ref=rule_ref,
                severity=Severity.MAJOR,
                status=CheckStatus.REVIEW,
                confidence=field_confidence,
                reason="Manufacturer region detected but OCR confidence below 85%. Manual review required.",
            )
        return CheckResult(
            check_id=check_id,
            check_name="Manufacturer/Packer",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.FAIL,
            confidence=field_confidence,
            reason=f"Manufacturer/Packer name and address not found on packaging. {_ocr_status(conf_map)}",
        )

    addr_suffix = f", {scan.manufacturer_address}" if scan.manufacturer_address else ""
    pin_suffix = f" - {scan.pincode}" if scan.pincode else ""
    full_evidence = f"{scan.manufacturer_name}{addr_suffix}{pin_suffix}"

    # Cross-field check vs digital claims
    if digital_claims and digital_claims.get("manufacturer_address") and scan.manufacturer_address:
        from rapidfuzz import fuzz
        sim = fuzz.token_set_ratio(digital_claims["manufacturer_address"], scan.manufacturer_address)
        if sim < 55:
            return CheckResult(
                check_id=check_id,
                check_name="Manufacturer/Packer",
                rule_ref=rule_ref,
                severity=Severity.MAJOR,
                status=CheckStatus.FAIL,
                evidence_text=f"Physical: {scan.manufacturer_address} | Digital: {digital_claims['manufacturer_address']}",
                confidence=field_confidence,
                matched_span={"text": full_evidence},
                reason=f"Cross-field Mismatch: Physical address differs from digital listing ({sim:.0f}% match).",
            )

    return CheckResult(
        check_id=check_id,
        check_name="Manufacturer/Packer",
        rule_ref=rule_ref,
        severity=Severity.MAJOR,
        status=CheckStatus.PASS,
        evidence_text=full_evidence,
        confidence=field_confidence,
        matched_span={"text": full_evidence},
        reason=f"Manufacturer declared: {full_evidence}",
    )



def validate_consumer_care(
    scan: NormalizedScan,
    field_confidence: float,
    check_id: str,
    rule_ref: str,
    conf_map: Optional[dict] = None,
    digital_claims: Optional[dict] = None,
) -> CheckResult:
    conf_map = conf_map or {}
    cc = scan.consumer_care

    if field_confidence < CONFIDENCE_THRESHOLD:
        return CheckResult(
            check_id=check_id,
            check_name="Consumer Care Details",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.UNCERTAIN,
            confidence=field_confidence,
            reason="OCR confidence below threshold. Field ambiguous.",
        )

    if not cc:
        ocr_flag = conf_map.get(_OCR_FLAG, 1.0)
        status = CheckStatus.UNCERTAIN if ocr_flag != 0.0 else CheckStatus.FAIL
        return CheckResult(
            check_id=check_id,
            check_name="Consumer Care Details",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=status,
            confidence=field_confidence,
            reason=f"Consumer care details not found. {_ocr_status(conf_map)} Marked UNCERTAIN for ML recovery.",
        )

    if cc.phone_found or cc.email_found:
        return CheckResult(
            check_id=check_id,
            check_name="Consumer Care Details",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.PASS,
            evidence_text=cc.raw_text,
            confidence=field_confidence,
            matched_span={"text": cc.raw_text},
            reason="Consumer care contact details verified.",
        )
    else:
        return CheckResult(
            check_id=check_id,
            check_name="Consumer Care Details",
            rule_ref=rule_ref,
            severity=Severity.MAJOR,
            status=CheckStatus.FAIL,
            evidence_text=cc.raw_text,
            confidence=field_confidence,
            matched_span={"text": cc.raw_text},
            reason="Consumer care section present but missing valid phone number or email.",
        )

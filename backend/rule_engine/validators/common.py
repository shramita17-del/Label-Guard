from backend.schemas import NormalizedScan, CheckResult, CheckStatus, Severity

CONFIDENCE_THRESHOLD = 0.85

def validate_mrp(scan: NormalizedScan, field_confidence: float, check_id: str, rule_ref: str) -> CheckResult:
    if field_confidence < CONFIDENCE_THRESHOLD:
        return CheckResult(
            check_id=check_id, check_name="MRP", rule_ref=rule_ref, severity=Severity.BLOCKING,
            status=CheckStatus.REVIEW, confidence=field_confidence, 
            reason="OCR confidence below 85%. Manual review required."
        )
        
    mrp = scan.mrp
    if not mrp:
        return CheckResult(
            check_id=check_id, check_name="MRP", rule_ref=rule_ref, severity=Severity.BLOCKING,
            status=CheckStatus.FAIL, confidence=field_confidence,
            reason="MRP declaration not found on package."
        )
        
    if mrp.currency_symbol_found and mrp.tax_inclusive_phrase_found:
        return CheckResult(
            check_id=check_id, check_name="MRP", rule_ref=rule_ref, severity=Severity.BLOCKING,
            status=CheckStatus.PASS, evidence_text=mrp.raw_text, confidence=field_confidence,
            reason="MRP value and tax inclusive phrase found."
        )
    else:
        return CheckResult(
            check_id=check_id, check_name="MRP", rule_ref=rule_ref, severity=Severity.BLOCKING,
            status=CheckStatus.FAIL, evidence_text=mrp.raw_text, confidence=field_confidence,
            reason="Missing currency symbol or 'inclusive of all taxes' phrase."
        )


def validate_net_quantity(scan: NormalizedScan, field_confidence: float, check_id: str, rule_ref: str) -> CheckResult:
    if field_confidence < CONFIDENCE_THRESHOLD:
        return CheckResult(
            check_id=check_id, check_name="Net Quantity", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.REVIEW, confidence=field_confidence,
            reason="OCR confidence below 85%."
        )
        
    qty = scan.net_quantity
    if not qty:
        return CheckResult(
            check_id=check_id, check_name="Net Quantity", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.FAIL, confidence=field_confidence, reason="Net quantity not found."
        )
        
    valid_units = ['g', 'kg', 'ml', 'l', 'number']
    if qty.unit in valid_units:
        return CheckResult(
            check_id=check_id, check_name="Net Quantity", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.PASS, evidence_text=qty.raw_text, confidence=field_confidence,
            reason=f"Valid standard unit '{qty.unit}' used."
        )
    else:
        return CheckResult(
            check_id=check_id, check_name="Net Quantity", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.FAIL, evidence_text=qty.raw_text, confidence=field_confidence,
            reason=f"Non-standard unit '{qty.unit}' used."
        )


def validate_date_of_mfg(scan: NormalizedScan, field_confidence: float, check_id: str, rule_ref: str) -> CheckResult:
    if field_confidence < CONFIDENCE_THRESHOLD:
        return CheckResult(
            check_id=check_id, check_name="Date of Mfg", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.REVIEW, confidence=field_confidence, reason="OCR confidence below 85%."
        )
        
    dates = [d for d in scan.dates if d.date_type == "mfg"]
    if not dates:
        return CheckResult(
            check_id=check_id, check_name="Date of Mfg", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.FAIL, confidence=field_confidence, reason="Manufacturing date not found."
        )
        
    date_obj = dates[0]
    if date_obj.is_valid_format:
        return CheckResult(
            check_id=check_id, check_name="Date of Mfg", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.PASS, evidence_text=date_obj.raw_text, confidence=field_confidence,
            reason="Valid manufacturing date format."
        )
    else:
        return CheckResult(
            check_id=check_id, check_name="Date of Mfg", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.FAIL, evidence_text=date_obj.raw_text, confidence=field_confidence,
            reason="Invalid date format."
        )


def validate_country_of_origin(scan: NormalizedScan, field_confidence: float, check_id: str, rule_ref: str) -> CheckResult:
    if field_confidence < CONFIDENCE_THRESHOLD:
        return CheckResult(
            check_id=check_id, check_name="Country of Origin", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.REVIEW, confidence=field_confidence, reason="OCR confidence below threshold."
        )
        
    coo = scan.country_of_origin
    if not coo:
        return CheckResult(
            check_id=check_id, check_name="Country of Origin", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.REVIEW, confidence=field_confidence, 
            reason="Country of origin missing. Requires manual verification if product is imported."
        )
        
    return CheckResult(
        check_id=check_id, check_name="Country of Origin", rule_ref=rule_ref, severity=Severity.MAJOR,
        status=CheckStatus.PASS, evidence_text=coo, confidence=field_confidence,
        reason=f"Country of origin declared: {coo}"
    )

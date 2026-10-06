from backend.schemas import NormalizedScan, CheckResult, CheckStatus, Severity

CONFIDENCE_THRESHOLD = 0.85

def validate_fiber_composition(scan: NormalizedScan, field_confidence: float, check_id: str, rule_ref: str) -> CheckResult:
    if field_confidence < CONFIDENCE_THRESHOLD:
        return CheckResult(
            check_id=check_id, check_name="Fiber Composition", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.REVIEW, confidence=field_confidence, reason="OCR confidence below 85%."
        )
        
    fiber = scan.fiber_composition
    if not fiber:
        return CheckResult(
            check_id=check_id, check_name="Fiber Composition", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.FAIL, confidence=field_confidence, reason="Fiber composition not found."
        )
        
    if fiber.sums_to_100:
        return CheckResult(
            check_id=check_id, check_name="Fiber Composition", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.PASS, confidence=field_confidence, 
            reason="Fiber components sum to exactly 100%."
        )
    else:
        return CheckResult(
            check_id=check_id, check_name="Fiber Composition", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.FAIL, confidence=field_confidence, 
            reason="Fiber components math violation (does not sum to 100%)."
        )

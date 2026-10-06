from backend.schemas import NormalizedScan, CheckResult, CheckStatus, Severity

CONFIDENCE_THRESHOLD = 0.85

def validate_veg_nonveg_mark(scan: NormalizedScan, field_confidence: float, check_id: str, rule_ref: str) -> CheckResult:
    if field_confidence < CONFIDENCE_THRESHOLD:
        return CheckResult(
            check_id=check_id, check_name="Veg/Non-Veg Mark", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.REVIEW, confidence=field_confidence, reason="CV detection confidence low."
        )
        
    mark = scan.veg_nonveg_mark
    if not mark:
        return CheckResult(
            check_id=check_id, check_name="Veg/Non-Veg Mark", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.FAIL, confidence=field_confidence, reason="Mark not detected on packaging."
        )
        
    if mark.shape_detected == "circle" and mark.color_detected in ["green", "brown"]:
        return CheckResult(
            check_id=check_id, check_name="Veg/Non-Veg Mark", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.PASS, confidence=field_confidence, 
            reason=f"Valid {mark.color_detected} {mark.shape_detected} mark detected."
        )
    else:
        return CheckResult(
            check_id=check_id, check_name="Veg/Non-Veg Mark", rule_ref=rule_ref, severity=Severity.MAJOR,
            status=CheckStatus.FAIL, confidence=field_confidence, 
            reason=f"Invalid mark geometry or color: {mark.color_detected} {mark.shape_detected}."
        )

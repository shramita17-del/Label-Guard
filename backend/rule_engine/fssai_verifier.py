import os
import json
import asyncio
from backend.schemas import NormalizedScan, CheckResult, CheckStatus, Severity
from backend.scraper_service.foscos_scraper import scrape_foscos_license

def verify_fssai_license(scan: NormalizedScan, field_confidence: float = 0.95) -> CheckResult:
    """
    Executes the 3-Tier FSSAI verification pipeline (Check F14).
    """
    license_no = scan.fssai_license_number
    check_id = "F14"
    rule_ref = "FSSAI_Licensing"
    
    if not license_no:
        return CheckResult(
            check_id=check_id, check_name="FSSAI License No.", rule_ref=rule_ref,
            severity=Severity.BLOCKING, status=CheckStatus.FAIL, confidence=field_confidence,
            reason="FSSAI license number not found on packaging."
        )

    # Remove spaces/hyphens
    clean_license = "".join(filter(str.isdigit, license_no))

    # ==========================================
    # TIER 1: Offline Syntax Validator
    # ==========================================
    if len(clean_license) != 14:
        return CheckResult(
            check_id=check_id, check_name="FSSAI License No.", rule_ref=rule_ref,
            severity=Severity.BLOCKING, status=CheckStatus.FAIL, confidence=field_confidence,
            evidence_text=license_no, reason="Tier 1 (Syntax): FSSAI number must be exactly 14 digits."
        )
        
    if clean_license[0] not in ['1', '2']:
        return CheckResult(
            check_id=check_id, check_name="FSSAI License No.", rule_ref=rule_ref,
            severity=Severity.BLOCKING, status=CheckStatus.FAIL, confidence=field_confidence,
            evidence_text=license_no, reason="Tier 1 (Syntax): First digit must be 1 or 2."
        )

    # ==========================================
    # TIER 2: Live FoSCoS Scraper (Async)
    # ==========================================
    live_result = None
    try:
        # Run the async scraper. If it fails, live_result remains None.
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            live_result = loop.run_until_complete(scrape_foscos_license(clean_license))
    except Exception:
        live_result = None

    if live_result:
        return CheckResult(
            check_id=check_id, check_name="FSSAI License No.", rule_ref=rule_ref,
            severity=Severity.BLOCKING, status=CheckStatus.PASS, confidence=field_confidence,
            evidence_text=license_no, reason=f"Tier 2 (Live): Verified active for {live_result.get('company_name')}"
        )

    # ==========================================
    # TIER 3: Local Cache Fallback
    # ==========================================
    cache_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fssai_cache.json")
    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    cached_data = cache.get(clean_license)
    if cached_data:
        if cached_data.get("status") == "Active":
            return CheckResult(
                check_id=check_id, check_name="FSSAI License No.", rule_ref=rule_ref,
                severity=Severity.BLOCKING, status=CheckStatus.PASS, confidence=field_confidence,
                evidence_text=license_no, 
                reason=f"Tier 3 (Cache): Verified active for {cached_data.get('company_name')}"
            )
        else:
            return CheckResult(
                check_id=check_id, check_name="FSSAI License No.", rule_ref=rule_ref,
                severity=Severity.BLOCKING, status=CheckStatus.FAIL, confidence=field_confidence,
                evidence_text=license_no, 
                reason=f"Tier 3 (Cache): License is {cached_data.get('status')}"
            )

    # Default Answer Principle (Rule 4)
    return CheckResult(
        check_id=check_id, check_name="FSSAI License No.", rule_ref=rule_ref,
        severity=Severity.BLOCKING, status=CheckStatus.REVIEW, confidence=field_confidence,
        evidence_text=license_no, 
        reason="Passed Tier 1 syntax, but FoSCoS unavailable and not in local cache. Manual review required."
    )

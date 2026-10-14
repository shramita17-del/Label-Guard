from typing import List, Dict, Optional
from datetime import datetime
from backend.schemas import (
    NormalizedScan,
    ProductCategory,
    CheckResult,
    CheckStatus,
    Severity,
    ComplianceReport,
    DigitalListingJSON,
    MismatchReport,
    MismatchField
)
from backend.parser_service.fuzzy_match import fuzzy_match_text
from backend.rule_engine.dynamic_checker import fetch_rule_metadata
from backend.rule_engine.validators.common import (
    validate_mrp,
    validate_net_quantity,
    validate_date_of_mfg,
    validate_country_of_origin
)
from backend.rule_engine.validators.apparel_textile import validate_fiber_composition
from backend.rule_engine.validators.food_beverage import validate_veg_nonveg_mark
from backend.rule_engine.fssai_verifier import verify_fssai_license


def evaluate_packaging_compliance(
    scan: NormalizedScan,
    conf_map: Optional[Dict[str, float]] = None
) -> List[CheckResult]:
    """
    Executes categorized compliance evaluation as per ProductCategory schema.
    Takes NormalizedScan (from OCR pipeline) and produces List[CheckResult]
    enriched with Supabase Legal Metrology citations.
    """
    if conf_map is None:
        conf_map = {}

    results: List[CheckResult] = []

    # -------------------------------------------------------------
    # 1. TIER 1: COMMON LEGAL METROLOGY CHECKS (All Categories)
    # -------------------------------------------------------------
    # MRP Check (Rule 6(1)(e))
    r_mrp = fetch_rule_metadata("R6_1_e")
    mrp_res = validate_mrp(
        scan,
        conf_map.get("mrp_region", 0.95),
        check_id="01",
        rule_ref=r_mrp.get("source_rule", "Rule_6_1_e")
    )
    results.append(mrp_res)

    # Net Quantity Check (Rule 6(1)(c))
    r_net_qty = fetch_rule_metadata("R6_1_c")
    net_qty_res = validate_net_quantity(
        scan,
        conf_map.get("net_quantity_region", 0.95),
        check_id="02",
        rule_ref=r_net_qty.get("source_rule", "Rule_6_1_c")
    )
    results.append(net_qty_res)

    # Date of Mfg Check (Rule 6(1)(d))
    r_mfg = fetch_rule_metadata("R6_1_d")
    mfg_res = validate_date_of_mfg(
        scan,
        conf_map.get("date_region", 0.95),
        check_id="03",
        rule_ref=r_mfg.get("source_rule", "Rule_6_1_d")
    )
    results.append(mfg_res)

    # Country of Origin Check (Rule 6(1)(aa))
    r_coo = fetch_rule_metadata("R6_1_aa")
    coo_res = validate_country_of_origin(
        scan,
        conf_map.get("coo_region", 0.95),
        check_id="04",
        rule_ref=r_coo.get("source_rule", "Rule_6_1_aa")
    )
    results.append(coo_res)

    # Manufacturer Details Check (Rule 6(1)(a))
    r_mfg_name = fetch_rule_metadata("R6_1_a")
    mfg_conf = conf_map.get("manufacturer_region", 0.95)
    if scan.manufacturer_name:
        addr_suffix = f", {scan.manufacturer_address}" if scan.manufacturer_address else ""
        pin_suffix = f" - {scan.pincode}" if scan.pincode else ""
        full_evidence = f"{scan.manufacturer_name}{addr_suffix}{pin_suffix}"
        results.append(
            CheckResult(
                check_id="05",
                check_name="Manufacturer/Packer",
                rule_ref=r_mfg_name.get("source_rule", "Rule_6_1_a"),
                severity=Severity.MAJOR,
                status=CheckStatus.PASS,
                evidence_text=full_evidence,
                confidence=mfg_conf,
                reason=f"Manufacturer declared: {full_evidence}"
            )
        )
    elif "manufacturer_region" in conf_map and mfg_conf < 0.85:
        results.append(
            CheckResult(
                check_id="05",
                check_name="Manufacturer/Packer",
                rule_ref=r_mfg_name.get("source_rule", "Rule_6_1_a"),
                severity=Severity.MAJOR,
                status=CheckStatus.REVIEW,
                confidence=mfg_conf,
                reason="Manufacturer region detected but OCR confidence below 85%. Manual review required."
            )
        )
    else:
        results.append(
            CheckResult(
                check_id="05",
                check_name="Manufacturer/Packer",
                rule_ref=r_mfg_name.get("source_rule", "Rule_6_1_a"),
                severity=Severity.MAJOR,
                status=CheckStatus.FAIL,
                confidence=mfg_conf,
                reason="Manufacturer/Packer name and address not found on packaging."
            )
        )

    # -------------------------------------------------------------
    # 2. TIER 2: CATEGORY-SPECIFIC CHECKS
    # -------------------------------------------------------------
    if scan.category == ProductCategory.FOOD_BEVERAGE:
        # Veg/Non-Veg Mark
        r_veg = fetch_rule_metadata("R6_8")
        results.append(
            validate_veg_nonveg_mark(
                scan,
                conf_map.get("veg_mark", 0.95),
                check_id="F12",
                rule_ref=r_veg.get("source_rule", "Rule_6_8")
            )
        )
        # FSSAI License verification
        results.append(
            verify_fssai_license(
                scan,
                conf_map.get("fssai_region", 0.95)
            )
        )

    elif scan.category == ProductCategory.APPAREL_TEXTILE:
        # Fiber Composition math validation
        results.append(
            validate_fiber_composition(
                scan,
                conf_map.get("fiber_region", 0.95),
                check_id="A02",
                rule_ref="Textile_Standards"
            )
        )

    elif scan.category == ProductCategory.GENERAL_RETAIL:
        # General Retail specific checks
        pass

    return results


def build_compliance_report(
    scan: NormalizedScan,
    conf_map: Optional[Dict[str, float]] = None
) -> ComplianceReport:
    """
    Convenience method: runs evaluation and wraps into final ComplianceReport schema.
    """
    check_results = evaluate_packaging_compliance(scan, conf_map)

    overall = CheckStatus.PASS
    for r in check_results:
        if r.status == CheckStatus.FAIL and r.severity == Severity.BLOCKING:
            overall = CheckStatus.FAIL
            break
        elif r.status == CheckStatus.REVIEW:
            if overall != CheckStatus.FAIL:
                overall = CheckStatus.REVIEW

    return ComplianceReport(
        scan_id=scan.scan_id,
        category=scan.category,
        overall_verdict=overall,
        check_results=check_results,
        generated_at=datetime.utcnow().isoformat()
    )


def compare_scans(physical: NormalizedScan, digital: DigitalListingJSON) -> MismatchReport:
    """
    Implements Flow C: Mismatch Fraud Check between physical packaging and e-commerce listings.
    """
    fields_compared = []
    
    # 1. Compare Manufacturer (Fuzzy text match)
    if physical.manufacturer_name and digital.manufacturer_name:
        is_match = fuzzy_match_text(physical.manufacturer_name, digital.manufacturer_name)
        fields_compared.append(
            MismatchField(
                field_name="manufacturer_name",
                physical_value=physical.manufacturer_name,
                digital_value=digital.manufacturer_name,
                match_method="fuzzy",
                is_match=is_match
            )
        )
        
    # 2. Compare MRP (Exact numeric match)
    if physical.mrp and digital.mrp:
        is_match = (physical.mrp.amount == digital.mrp.amount)
        fields_compared.append(
            MismatchField(
                field_name="mrp",
                physical_value=str(physical.mrp.amount),
                digital_value=str(digital.mrp.amount),
                match_method="exact",
                is_match=is_match
            )
        )
        
    # 3. Compare Net Quantity (Exact numeric/unit match)
    if physical.net_quantity and digital.net_quantity:
        is_match = (physical.net_quantity.value == digital.net_quantity.value and 
                    physical.net_quantity.unit == digital.net_quantity.unit)
        fields_compared.append(
            MismatchField(
                field_name="net_quantity",
                physical_value=f"{physical.net_quantity.value}{physical.net_quantity.unit}",
                digital_value=f"{digital.net_quantity.value}{digital.net_quantity.unit}",
                match_method="exact",
                is_match=is_match
            )
        )

    # 4. Compare Country of Origin (Fuzzy)
    if physical.country_of_origin and digital.country_of_origin:
        is_match = fuzzy_match_text(physical.country_of_origin, digital.country_of_origin)
        fields_compared.append(
            MismatchField(
                field_name="country_of_origin",
                physical_value=physical.country_of_origin,
                digital_value=digital.country_of_origin,
                match_method="fuzzy",
                is_match=is_match
            )
        )

    # Determine overall fraud flag (True if ANY field is NOT a match)
    overall_mismatch = any(not f.is_match for f in fields_compared)
    
    return MismatchReport(
        scan_id=physical.scan_id,
        digital_listing_url=digital.source_url,
        fields_compared=fields_compared,
        overall_mismatch_detected=overall_mismatch
    )

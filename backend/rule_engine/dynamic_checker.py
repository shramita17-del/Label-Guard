from typing import Dict, Any, List, Optional
from backend.db_service.query_rules import get_rule_by_id

# 1. FAKE INPUT (stand-in for vision pipeline's extracted fields)
sample_extracted_data = {
    "category": "food",
    "is_imported": True,
    "manufacture_date": "2024-03-15",
    "mrp_text": "MRP Rs. 45.00 incl. of all taxes",
    "net_quantity_text": "200g",
    "manufacturer_name": "ABC Foods Pvt Ltd",
    "manufacturer_address": None,      # missing on purpose, should FAIL
    "country_of_origin": None,         # missing on purpose, should FAIL (product is imported)
    "consumer_care_email": "care@abcfoods.com",
    "consumer_care_phone": "9876543210",
}

# 2. COMPARISON LOGIC
def check_field(extracted_value: Optional[str], rule_id: str, field_description: str) -> Dict[str, Any]:
    """
    Validates field presence and decorates the result with the official Legal Metrology rule citation
    and amendment history stored in Supabase.
    """
    rule = get_rule_by_id(rule_id)

    citation = rule.get("source_rule", rule_id) if rule else rule_id
    requirement = rule.get("requirement") if rule else ""

    if rule is None:
        return {
            "field": field_description,
            "verdict": "REVIEW",
            "reason": f"Rule {rule_id} not available in database (or offline) - verifying presence only.",
            "rule_citation": citation,
            "requirement": requirement
        }

    if extracted_value is None or str(extracted_value).strip() == "":
        return {
            "field": field_description,
            "verdict": "FAIL",
            "reason": f"{field_description} is missing - required by {rule_id}",
            "rule_citation": citation,
            "requirement": requirement
        }

    return {
        "field": field_description,
        "verdict": "PASS",
        "reason": f"{field_description} is present",
        "rule_citation": citation,
        "requirement": requirement
    }


def run_checklist(extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    results.append(check_field(extracted_data.get("manufacturer_name"), "R6_1_a", "Manufacturer Name"))
    results.append(check_field(extracted_data.get("manufacturer_address"), "R6_1_a", "Manufacturer Address"))
    results.append(check_field(extracted_data.get("net_quantity_text"), "R6_1_c", "Net Quantity"))
    results.append(check_field(extracted_data.get("mrp_text"), "R6_1_e", "MRP"))
    results.append(check_field(extracted_data.get("consumer_care_email"), "R6_2", "Consumer Care Email"))
    results.append(check_field(extracted_data.get("consumer_care_phone"), "R6_2", "Consumer Care Phone"))

    # Conditional check: country of origin is only required for IMPORTED products
    if extracted_data.get("is_imported"):
        results.append(check_field(extracted_data.get("country_of_origin"), "R6_1_aa", "Country of Origin"))

    return results


def print_report(results: List[Dict[str, Any]]):
    print("\n COMPLIANCE REPORT")
    for r in results:
        print(f"[{r['verdict']}] {r['field']} — {r['reason']} (Citation: {r.get('rule_citation')})")
    print("\n")


# 3. RUN
if __name__ == "__main__":
    results = run_checklist(sample_extracted_data)
    print_report(results)

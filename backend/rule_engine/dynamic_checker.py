from typing import Dict, Any, List, Optional
from backend.db_service.query_rules import get_rule_by_id

# In-memory fallback if Supabase is offline, so pipelines never crash
OFFLINE_RULES_CACHE: Dict[str, Dict[str, Any]] = {
    "R6_1_a": {
        "source_rule": "Rule 6(1)(a)",
        "field": "Manufacturer/Packer/Importer Name & Address",
        "requirement": "Name and address of manufacturer, packer or importer.",
        "mandatory": True
    },
    "R6_1_aa": {
        "source_rule": "Rule 6(1)(aa)",
        "field": "Country of Origin",
        "requirement": "For imported products, country of origin/manufacture/assembly.",
        "mandatory": "conditional"
    },
    "R6_1_b": {
        "source_rule": "Rule 6(1)(b)",
        "field": "Common/Generic Name",
        "requirement": "Common or generic name of the commodity.",
        "mandatory": True
    },
    "R6_1_c": {
        "source_rule": "Rule 6(1)(c)",
        "field": "Net Quantity",
        "requirement": "Net quantity in standard SI unit (g, kg, ml, l, or number).",
        "mandatory": True
    },
    "R6_1_d": {
        "source_rule": "Rule 6(1)(d)",
        "field": "Month and Year of Manufacture",
        "requirement": "Month and year commodity was manufactured.",
        "mandatory": True
    },
    "R6_1_da": {
        "source_rule": "Rule 6(1)(da)",
        "field": "Best Before / Use By Date",
        "requirement": "Best before or use by date for time-sensitive commodities.",
        "mandatory": "conditional"
    },
    "R6_1_e": {
        "source_rule": "Rule 6(1)(e)",
        "field": "Retail Sale Price (MRP)",
        "requirement": "MRP inclusive of all taxes in Indian currency.",
        "mandatory": True
    },
    "R6_2": {
        "source_rule": "Rule 6(2)",
        "field": "Consumer Care Details",
        "requirement": "Name, address, phone number and email for consumer complaints.",
        "mandatory": True
    },
    "R6_8": {
        "source_rule": "Rule 6(8)",
        "field": "Veg/Non-Veg Dot Declaration",
        "requirement": "Green dot for vegetarian or Brown/Red dot for non-vegetarian.",
        "mandatory": "conditional"
    },
    "R26_f": {
        "source_rule": "Rule 26(f)",
        "field": "Loose Garment Reduced Checklist",
        "requirement": "Reduced 4-field checklist for loose/open inspection garments.",
        "mandatory": "conditional"
    }
}


def fetch_rule_metadata(rule_id: str) -> Dict[str, Any]:
    """
    Fetches rule metadata from Supabase with automatic local fallback.
    """
    rule = get_rule_by_id(rule_id)
    if rule:
        return rule
    return OFFLINE_RULES_CACHE.get(rule_id, {
        "source_rule": rule_id,
        "requirement": "Legal Metrology Packaged Commodities Rules 2011"
    })

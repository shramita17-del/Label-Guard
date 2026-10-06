from typing import List
from backend.schemas import NormalizedScan, DigitalListingJSON, MismatchReport, MismatchField
from backend.parser_service.fuzzy_match import fuzzy_match_text

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

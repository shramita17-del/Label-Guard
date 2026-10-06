import re
from typing import Optional
from backend.schemas import NetQuantity, MRPDeclaration, DateDeclaration, FiberComposition

def normalize_mrp(raw_text: str) -> Optional[MRPDeclaration]:
    """Parses text like 'Rs. 199.50 (inclusive of all taxes)'"""
    if not raw_text: return None
    text_lower = raw_text.lower()
    
    # Extract numerical price
    match = re.search(r'(?:rs\.?|₹|inr)\s*([\d\.,]+)', text_lower)
    if not match:
        return None
        
    amount_str = match.group(1).replace(',', '')
    try:
        amount = float(amount_str)
    except ValueError:
        return None
        
    currency_found = 'rs' in text_lower or '₹' in text_lower or 'inr' in text_lower
    tax_inclusive = 'incl' in text_lower and 'tax' in text_lower
    
    return MRPDeclaration(
        amount=amount,
        currency_symbol_found=currency_found,
        tax_inclusive_phrase_found=tax_inclusive,
        raw_text=raw_text
    )

def normalize_net_quantity(raw_text: str) -> Optional[NetQuantity]:
    """Parses text like 'Net Wt: 500g', '1.5 Litres', '5 pcs'"""
    if not raw_text: return None
    text_lower = raw_text.lower()
    
    # Regex to find number and standard units
    match = re.search(r'([\d\.]+)\s*(g|kg|ml|l|litre|litres|grams|gram|pieces|pcs|number|no|u)\b', text_lower)
    if not match:
        return None
        
    val_str, unit_raw = match.groups()
    try:
        val = float(val_str)
    except ValueError:
        return None
        
    # Standardize unit mappings per schemas.py
    unit = unit_raw
    if unit in ['grams', 'gram']: unit = 'g'
    elif unit in ['litre', 'litres']: unit = 'l'
    elif unit in ['pieces', 'pcs', 'no', 'u']: unit = 'number'
    
    return NetQuantity(
        value=val,
        unit=unit,
        raw_text=raw_text
    )

def normalize_dates(raw_text: str, date_type: str = "mfg") -> Optional[DateDeclaration]:
    """Parses dates in formats like MM/YYYY, DD-MM-YYYY"""
    if not raw_text: return None
    
    match = re.search(r'(\d{1,2})[/\-](\d{2,4}(?!\d))', raw_text)
    
    year = None
    month = None
    is_valid = False
    
    if match:
        p1, p2 = match.groups()
        if len(p2) == 4:
            year = int(p2)
            month = int(p1) if 1 <= int(p1) <= 12 else None
        elif len(p2) == 2 and len(p1) <= 2:
            year = 2000 + int(p2) # hackathon assumption
            month = int(p1)
            
        if year and month and 1 <= month <= 12:
            is_valid = True
            
    return DateDeclaration(
        date_type=date_type,
        month=month,
        year=year,
        raw_text=raw_text,
        is_valid_format=is_valid
    )

def normalize_fiber_composition(raw_text: str) -> Optional[FiberComposition]:
    """Parses '60% cotton, 40% polyester' and validates math"""
    if not raw_text: return None
    text_lower = raw_text.lower()
    
    matches = re.findall(r'([\d\.]+)\s*%\s*([a-z]+)', text_lower)
    
    components = {}
    total = 0.0
    for pct_str, material in matches:
        try:
            pct = float(pct_str)
            components[material] = pct
            total += pct
        except ValueError:
            continue
            
    if not components:
        return None
        
    return FiberComposition(
        components=components,
        sums_to_100=abs(total - 100.0) < 0.01
    )

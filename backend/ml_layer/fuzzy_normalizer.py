import re
from typing import Optional, Tuple, Dict, Any, List
from rapidfuzz import fuzz, process

class FuzzyNormalizer:
    """
    Component 2 of Layer 2 ML Model:
    Uses RapidFuzz string matching and a rule-based OCR error correction map
    (e.g., O/0, l/1, S/5, g/9, R$/Rs, broken decimals, spacing) to recover
    canonical values from noisy or corrupted OCR tokens.
    """

    # Canonical Legal Metrology tax phrase variants
    TAX_PHRASES = [
        "inclusive of all taxes",
        "incl of all taxes",
        "incl. of all taxes",
        "incl of taxes",
        "all taxes inclusive",
        "incl. all taxes",
        "inclusive all taxes",
    ]

    STANDARD_UNITS = ["g", "kg", "ml", "l", "number", "pcs", "pieces", "units", "m", "cm"]

    def __init__(self):
        pass

    # ─────────────────────────────────────────────────────────
    # Low-level OCR Noise Cleaners
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def fix_ocr_digits(text: str) -> str:
        """
        Replaces common OCR letter misrecognitions within numeric strings:
        O/o -> 0, l/I/| -> 1, S/s -> 5, B -> 8, g/q -> 9, Z/z -> 2
        """
        if not text:
            return ""

        # Normalize common broken decimals: '299 . 00' or '299 , 00' -> '299.00'
        text = re.sub(r'(\d+)\s*[\.,]\s*(\d{2})\b', r'\1.\2', text)

        # Fix currency variants: R$, R5, Rs., ₹
        text = re.sub(r'\b[R|r][\$|5]\b', 'Rs.', text)
        text = re.sub(r'\b[R|r][S|s]\b', 'Rs.', text)
        text = re.sub(r'\b[R|r]\s*[\.\-]?\s*[S|s]\s*[\.\-]?', 'Rs. ', text)

        # Replace 'g' or 'q' with '9' inside numbers, e.g., '2gg.00' -> '299.00'
        def _replace_inner_digits(match):
            sub = match.group(0)
            sub = sub.replace('g', '9').replace('q', '9')
            sub = sub.replace('O', '0').replace('o', '0')
            sub = sub.replace('l', '1').replace('I', '1').replace('|', '1')
            sub = sub.replace('S', '5').replace('s', '5')
            sub = sub.replace('B', '8')
            sub = sub.replace('Z', '2').replace('z', '2')
            return sub

        # Pattern matching digit-like sequences with corrupted characters
        text = re.sub(r'\b(?=[a-zA-Z0-9\.\,]*\d)[a-zA-Z0-9\.\,]{2,}\b', _replace_inner_digits, text)
        return text

    # ─────────────────────────────────────────────────────────
    # Field Recovery Helpers
    # ─────────────────────────────────────────────────────────

    def recover_mrp(self, raw_text: str) -> Dict[str, Any]:
        """
        Recovers MRP price, currency presence, and tax-inclusive phrase
        from noisy OCR text like 'MRP: R$. 2gg.00 (inc1 of all taxes)'.
        """
        cleaned = self.fix_ocr_digits(raw_text)

        # Currency symbol detection
        currency_found = bool(
            re.search(r'(?:Rs\.?|INR|₹|MRP)', raw_text, re.IGNORECASE) or
            re.search(r'(?:Rs\.?|INR|₹)', cleaned, re.IGNORECASE)
        )

        # Tax phrase detection via RapidFuzz
        tax_phrase_found = False
        best_tax_score = 0.0
        for phrase in self.TAX_PHRASES:
            score = fuzz.partial_ratio(phrase, raw_text.lower())
            if score > best_tax_score:
                best_tax_score = score
            if score >= 75:  # Tolerance for OCR corruptions (e.g. 'inc1 of a11 taxes')
                tax_phrase_found = True
                break

        # Extract numeric amount
        amount = None
        # Look for numbers with decimal e.g. 299.00, 49.50, or standalone integers after Rs/MRP
        price_matches = re.findall(r'(?:Rs\.?|INR|₹|MRP\s*[:\-])?\s*(\d+(?:\.\d{1,2})?)', cleaned, re.IGNORECASE)
        candidates = []
        for p in price_matches:
            try:
                val = float(p)
                if 0.5 <= val <= 500000.0:  # Reasonable price range
                    candidates.append(val)
            except ValueError:
                pass

        if candidates:
            # Pick first valid price candidate
            amount = candidates[0]

        recovered_str = ""
        if amount is not None:
            tax_suffix = " (Incl. of all taxes)" if tax_phrase_found else ""
            curr_prefix = "Rs. " if currency_found else ""
            recovered_str = f"MRP: {curr_prefix}{amount:.2f}{tax_suffix}"

        return {
            "recovered_text": recovered_str or cleaned,
            "amount": amount,
            "currency_symbol_found": currency_found,
            "tax_inclusive_phrase_found": tax_phrase_found,
            "tax_match_score": round(best_tax_score / 100.0, 3),
            "is_valid": amount is not None and currency_found and tax_phrase_found,
        }

    def recover_net_quantity(self, raw_text: str) -> Dict[str, Any]:
        """
        Recovers numeric quantity and standard metric units
        from noisy OCR e.g., '200 m1' -> 200 ml, '1.5 kq' -> 1.5 kg, '5OO g' -> 500 g.
        """
        cleaned = self.fix_ocr_digits(raw_text)

        # Fix common unit typos: m1 -> ml, kq -> kg, gm -> g, gms -> g, ltr -> l, ltrs -> l
        unit_fixes = [
            (r'\bm1\b', 'ml'),
            (r'\bmL\b', 'ml'),
            (r'\bkq\b', 'kg'),
            (r'\bgms?\b', 'g'),
            (r'\bltrs?\b', 'l'),
            (r'\bnumber\b', 'number'),
            (r'\bnos?\b', 'number'),
            (r'\bpcs?\b', 'pcs'),
        ]
        fixed_text = cleaned
        for pat, rep in unit_fixes:
            fixed_text = re.sub(pat, rep, fixed_text, flags=re.IGNORECASE)

        # Match (value, unit) pairs
        match = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)', fixed_text)
        val = None
        unit = None
        if match:
            try:
                val = float(match.group(1))
                cand_unit = match.group(2).lower()
                # Best fuzzy match against canonical standard units
                best_unit, score, _ = process.extractOne(cand_unit, self.STANDARD_UNITS, scorer=fuzz.ratio)
                if score >= 80:
                    unit = best_unit
            except (ValueError, TypeError):
                pass

        # Normalize unit names to statutory LM standard units
        if unit in ["pcs", "pieces", "units"]:
            unit = "number"

        recovered_str = f"{val:g} {unit}" if (val is not None and unit is not None) else ""

        return {
            "recovered_text": recovered_str or cleaned,
            "value": val,
            "unit": unit,
            "is_valid": val is not None and unit in ["g", "kg", "ml", "l", "number", "m", "cm"],
        }

    def recover_date(self, raw_text: str) -> Dict[str, Any]:
        """
        Recovers manufacturing/packing dates e.g. 'l0/2024' -> 10/2024, '1O-2O24' -> 10/2024.
        """
        cleaned = self.fix_ocr_digits(raw_text)

        # Pattern: MM/YYYY or MM-YYYY or DD/MM/YYYY or DD-MM-YYYY
        match = re.search(r'\b(0?[1-9]|1[0-2])[\/\-](20\d{2})\b', cleaned)
        if match:
            month = int(match.group(1))
            year = int(match.group(2))
            recovered_str = f"{month:02d}/{year}"
            return {
                "recovered_text": recovered_str,
                "month": month,
                "year": year,
                "is_valid": True,
            }

        # Check full date DD/MM/YYYY
        match_full = re.search(r'\b([0-2]?[1-9]|3[01])[\/\-](0?[1-9]|1[0-2])[\/\-](20\d{2})\b', cleaned)
        if match_full:
            month = int(match_full.group(2))
            year = int(match_full.group(3))
            recovered_str = f"{month:02d}/{year}"
            return {
                "recovered_text": recovered_str,
                "month": month,
                "year": year,
                "is_valid": True,
            }

        return {
            "recovered_text": cleaned,
            "month": None,
            "year": None,
            "is_valid": False,
        }

    def recover_country(self, raw_text: str) -> Dict[str, Any]:
        """
        Recovers Country of Origin declaration from noisy OCR text.
        """
        cleaned = self.fix_ocr_digits(raw_text)

        common_countries = ["India", "United States", "China", "Germany", "Japan", "Vietnam", "Thailand", "Bangladesh", "Sri Lanka"]
        best_country = None
        best_score = 0.0

        for country in common_countries:
            score = fuzz.partial_ratio(country.lower(), cleaned.lower())
            if score > best_score:
                best_score = score
                best_country = country

        is_valid = best_score >= 80
        return {
            "recovered_text": f"Country of Origin: {best_country}" if is_valid else cleaned,
            "country": best_country if is_valid else None,
            "score": round(best_score / 100.0, 3),
            "is_valid": is_valid,
        }

    def recover_consumer_care(self, raw_text: str) -> Dict[str, Any]:
        """
        Recovers phone number and email from consumer care text.
        """
        cleaned = self.fix_ocr_digits(raw_text)
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', cleaned)
        phone_match = re.search(r'(?:1800[-\s]?\d{3}[-\s]?\d{3,4}|\+?91[-\s]?[6-9]\d{9}|\b[6-9]\d{9}\b)', cleaned)

        email = email_match.group(0) if email_match else None
        phone = phone_match.group(0) if phone_match else None

        return {
            "recovered_text": f"Phone: {phone or 'N/A'}, Email: {email or 'N/A'}",
            "email": email,
            "phone": phone,
            "is_valid": email is not None or phone is not None,
        }

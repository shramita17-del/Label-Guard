import re
import math
from typing import List, Optional
from backend.schemas import OCRToken


def _match_keyword(txt: str, kw: str) -> bool:
    """Matches keyword on whole words with word boundaries, not substrings (e.g. 'rs' does not match 'SUGARS')."""
    kw_clean = kw.strip().lower()
    txt_clean = txt.lower()
    if not re.search(r'\w', kw_clean):
        return kw_clean in txt_clean
    pattern = r'(?:\b|^)' + re.escape(kw_clean) + r'(?:\b|$|[.:])'
    return bool(re.search(pattern, txt_clean))


def _find_anchor_token(tokens: List[OCRToken], keywords: List[str]) -> Optional[OCRToken]:
    """Finds the first token matching any anchor keyword using word boundaries."""
    for token in tokens:
        for kw in keywords:
            if _match_keyword(token.text, kw):
                return token
    return None


def _find_associated_value_token(
    tokens: List[OCRToken],
    anchor: OCRToken,
    val_regex: Optional[str] = None
) -> Optional[OCRToken]:
    """
    Finds the associated value token for an anchor:
    1. Looks to the right on the same line or in the paired value column.
    2. Looks directly below within a column.
    3. Handles label column on the left and value column on the right (PKD -> 15/06/26, MRP Rs. -> 40.00).
    """
    abox = anchor.bbox
    acx = (abox.x_min + abox.x_max) / 2.0
    acy = (abox.y_min + abox.y_max) / 2.0
    ah = max(1, abox.y_max - abox.y_min)
    aw = max(1, abox.x_max - abox.x_min)

    best_tok = None
    best_cost = float("inf")

    for tok in tokens:
        if tok == anchor:
            continue
        if val_regex and not re.search(val_regex, tok.text):
            continue

        tbox = tok.bbox
        tcx = (tbox.x_min + tbox.x_max) / 2.0
        tcy = (tbox.y_min + tbox.y_max) / 2.0

        dx = tbox.x_min - abox.x_max
        dy = tcy - acy

        # Case 1: To the right on the same line or in paired value column
        if tbox.x_min >= abox.x_min - 20 and -ah * 0.8 <= dy <= ah * 3.5:
            cost = max(0, dx) + abs(dy) * 2.0
        # Case 2: Directly BELOW
        elif tbox.y_min >= abox.y_min and abs(tcx - acx) <= max(120, aw * 1.5):
            cost = max(0, tbox.y_min - abox.y_max) * 1.5 + abs(tcx - acx)
        else:
            cost = 10000 + math.hypot(tcx - acx, tcy - acy)

        if cost < best_cost:
            best_cost = cost
            best_tok = tok

    return best_tok


def extract_field(tokens: List[OCRToken], anchor_keywords: List[str], val_regex: Optional[str] = None) -> Optional[str]:
    anchor = _find_anchor_token(tokens, anchor_keywords)
    if not anchor:
        return None
    val_tok = _find_associated_value_token(tokens, anchor, val_regex=val_regex)
    if val_tok:
        return val_tok.text.strip()
    return None


def extract_mrp(tokens: List[OCRToken]) -> Optional[str]:
    keywords = ["mrp", "price", "rs", "₹", "inr"]
    anchor = _find_anchor_token(tokens, keywords)
    if not anchor:
        return None

    # Check if price is within the anchor token itself
    m = re.search(r'(?:rs|₹|inr|mrp)?\s*[:\-]?\s*(\d+(?:\.\d{1,2})?)', anchor.text, re.IGNORECASE)
    if m and float(m.group(1)) > 0:
        return m.group(1)

    val_tok = _find_associated_value_token(tokens, anchor, val_regex=r'\d+\.\d{2}')
    if val_tok:
        return val_tok.text.strip()

    # Fallback to closest numeric token
    val_tok = _find_associated_value_token(tokens, anchor, val_regex=r'\d+')
    if val_tok:
        return val_tok.text.strip()
    return None


def extract_net_quantity(tokens: List[OCRToken]) -> Optional[str]:
    keywords = ["net weight", "net wt", "net quantity", "net qty", "net"]
    anchor = _find_anchor_token(tokens, keywords)
    if not anchor:
        return None

    # Check if net quantity is inside anchor token
    m = re.search(r'\b(\d+(?:\.\d+)?\s*(?:g|kg|ml|l|gm|grams))\b', anchor.text, re.IGNORECASE)
    if m:
        return m.group(1)

    val_tok = _find_associated_value_token(tokens, anchor, val_regex=r'\d+\s*(?:g|kg|ml|l|gm|grams)\b')
    if val_tok:
        return val_tok.text.strip()
    return None


def extract_fssai(tokens: List[OCRToken]) -> Optional[str]:
    keywords = ["fssai", "license", "licence", "lic. no", "lic no"]
    anchor = _find_anchor_token(tokens, keywords)
    if anchor:
        m = re.search(r'\b([12]\d{13})\b', anchor.text)
        if m:
            return m.group(1)
        val_tok = _find_associated_value_token(tokens, anchor, val_regex=r'\b[12]\d{13}\b')
        if val_tok:
            m = re.search(r'\b([12]\d{13})\b', val_tok.text)
            if m:
                return m.group(1)

    # Fallback: search any 14-digit number
    for token in tokens:
        m = re.search(r'\b([12]\d{13})\b', token.text)
        if m:
            return m.group(1)
    return None


def extract_coo(tokens: List[OCRToken]) -> Optional[str]:
    keywords = ["country of origin", "made in", "origin"]
    anchor = _find_anchor_token(tokens, keywords)
    if not anchor:
        return None
    val_tok = _find_associated_value_token(tokens, anchor)
    if val_tok:
        return val_tok.text.strip()
    return None


def extract_manufacturer(tokens: List[OCRToken]) -> Optional[str]:
    keywords = ["marketed by", "mfg by", "manufactured by", "packed by", "mfd by"]
    anchor = _find_anchor_token(tokens, keywords)
    if not anchor:
        return None
    val_tok = _find_associated_value_token(tokens, anchor)
    if val_tok:
        return val_tok.text.strip()
    return None


def extract_date(tokens: List[OCRToken]) -> Optional[str]:
    # 1. Prioritize manufacturing / packaging date anchors
    mfg_keywords = ["pkd", "mfg", "manufacturing", "packed", "date of mfg", "date of pkd"]
    anchor = _find_anchor_token(tokens, mfg_keywords)
    if anchor:
        m = re.search(r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b', anchor.text)
        if m:
            return m.group(1)
        val_tok = _find_associated_value_token(tokens, anchor, val_regex=r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}')
        if val_tok:
            m = re.search(r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b', val_tok.text)
            if m:
                return m.group(1)
            return val_tok.text.strip()

    # 2. Fallback to expiry / use by date anchors
    exp_keywords = ["use by", "expiry", "best before", "exp", "date"]
    anchor = _find_anchor_token(tokens, exp_keywords)
    if anchor:
        m = re.search(r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b', anchor.text)
        if m:
            return m.group(1)
        val_tok = _find_associated_value_token(tokens, anchor, val_regex=r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}')
        if val_tok:
            m = re.search(r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b', val_tok.text)
            if m:
                return m.group(1)
            return val_tok.text.strip()

    return None

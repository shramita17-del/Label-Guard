import difflib

def fuzzy_match_text(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """
    Compares two strings. If match >= threshold, returns True.
    Uses difflib for a zero-dependency hackathon fallback. 
    (Can be swapped to RapidFuzz if time permits).
    """
    if not text1 or not text2:
        return False
        
    t1 = str(text1).lower().strip()
    t2 = str(text2).lower().strip()
    
    score = difflib.SequenceMatcher(None, t1, t2).ratio()
    return score >= threshold

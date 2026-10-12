import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

url: Optional[str] = os.environ.get("SUPABASE_URL")
key: Optional[str] = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

supabase = None
if url and key:
    try:
        from supabase import create_client
        supabase = create_client(url, key)
    except Exception as e:
        print(f"[Supabase Warning] Failed to initialize client: {e}")
        supabase = None


def get_rule_by_id(rule_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches a single rule by its rule_id (e.g. 'R6_1_e').
    Returns the full content (JSONB) for that rule, or None if not found or DB unavailable.
    """
    if not supabase:
        return None
    try:
        response = supabase.table("rules").select("*").eq("rule_id", rule_id).execute()
        if response.data and len(response.data) > 0:
            return response.data[0].get("content")
    except Exception as e:
        print(f"[Supabase Error] get_rule_by_id failed for {rule_id}: {e}")
    return None


def get_rules_by_section(section_name: str) -> List[Dict[str, Any]]:
    """
    Fetches ALL rules belonging to one section (e.g. 'declarations', 'exemptions').
    Returns a list of their content objects.
    """
    if not supabase:
        return []
    try:
        response = supabase.table("rules").select("*").eq("section", section_name).execute()
        if response.data:
            return [row.get("content") for row in response.data]
    except Exception as e:
        print(f"[Supabase Error] get_rules_by_section failed for {section_name}: {e}")
    return []


def get_all_sections() -> List[str]:
    """
    Utility: lists every distinct section name currently in the table.
    Useful for sanity-checking what actually got seeded.
    """
    if not supabase:
        return []
    try:
        response = supabase.table("rules").select("section").execute()
        if response.data:
            sections = set(row.get("section") for row in response.data)
            return sorted(sections)
    except Exception as e:
        print(f"[Supabase Error] get_all_sections failed: {e}")
    return []


# Quick manual test when you run this file directly
if __name__ == "__main__":
    if not supabase:
        print("Supabase credentials not configured in .env (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)")
    else:
        print("Sections in DB:", get_all_sections())
        print()
        print("Fetching R6_1_e:")
        print(get_rule_by_id("R6_1_e"))

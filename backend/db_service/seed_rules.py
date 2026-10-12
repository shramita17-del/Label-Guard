import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

if not url or not key:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in your .env file before seeding.")

from supabase import create_client
supabase = create_client(url, key)

# Load rules_base.json from data/ directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
JSON_PATH = BASE_DIR / "data" / "rules_base.json"

if not JSON_PATH.exists():
    raise FileNotFoundError(f"Rules base file not found at: {JSON_PATH}")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

def insert_row(section, rule_id, effective_from, source_gsr, content):
    """One function, reused for every insert — avoids repeating the same insert code everywhere."""
    supabase.table("rules").insert({
        "section": section,
        "rule_id": rule_id,
        "effective_from": effective_from,
        "source_gsr": source_gsr,
        "content": content
    }).execute()

def walk_section(section_name, section_data):
    """
    Handles the fact that different sections of your JSON are shaped differently:
    - some are a LIST of rule objects (e.g. declarations, exemptions, known_gaps)
    - some are a single big DICT (e.g. font_size_rules, source)
    """
    if isinstance(section_data, list):
        for item in section_data:
            if isinstance(item, dict):
                insert_row(
                    section=section_name,
                    rule_id=item.get("rule_id"),
                    effective_from=item.get("effective_from"),
                    source_gsr=item.get("inserted_by") or item.get("amended_by") or item.get("gsr"),
                    content=item
                )
    elif isinstance(section_data, dict):
        insert_row(
            section=section_name,
            rule_id=section_data.get("rule_id"),
            effective_from=section_data.get("effective_from"),
            source_gsr=section_data.get("inserted_by") or section_data.get("amended_by"),
            content=section_data
        )

def run_seeder():
    # Walk every top-level section of your JSON and seed it
    for section_name, section_data in data.items():
        walk_section(section_name, section_data)
        print(f"Seeded section: {section_name}")

    print("All sections imported into Supabase successfully.")

if __name__ == "__main__":
    run_seeder()

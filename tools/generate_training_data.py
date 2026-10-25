import os
import sys
import json
import random
import glob
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.schemas import BoundingBox
from backend.ocr_service.extractor import OCRExtractor
from backend.ml_layer.field_classifier import FieldClassifier
from backend.ml_layer.confidence_scorer import ComplianceConfidenceScorer

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = os.path.join("data", "synthetic_dataset")
REAL_DIR = os.path.join("data", "real_labeled")
MODELS_DIR = "models"
VERSION_TAG = "v1.0.0-hybrid"

# Template pools
BRANDS = ["FreshFarm", "PureHerb", "BioVeda", "NutriPack", "UrbanKnit", "ApexGoods", "DailyBite", "ZenithCraft"]
PRODUCTS = ["Green Tea 100g", "Almond Oil 200ml", "Cotton Crewneck T-Shirt", "Basmati Rice 5kg", "Organic Honey 500g", "Herbal Shampoo 250ml"]
MANUFACTURERS = [
    ("Apex Agro Foods Pvt. Ltd.", "Plot 42, Okhla Industrial Area, Phase-III, New Delhi", "110020"),
    ("Himalayan Botanicals Ltd.", "Sector 5, Industrial Estate, Dehradun, Uttarakhand", "248001"),
    ("Zenith Garments & Textiles", "14/B Tirupur Knitwear Park, Tamil Nadu", "641602"),
    ("Daily Needs Consumer Care", "GIDC Estate, Phase 2, Vatva, Ahmedabad, Gujarat", "382445"),
]
COUNTRIES = ["India", "Made in India", "Country of Origin: India"]
TAX_PHRASES = ["(Incl. of all taxes)", "(inclusive of all taxes)", "INCL. OF ALL TAXES", "(incl of all taxes)"]

def create_sample_label_spec(sample_id: int) -> Dict[str, Any]:
    """
    Creates a label specification with randomized compliant or defective fields.
    """
    category = random.choice(["food_beverage", "general_retail", "apparel_textile"])
    brand = random.choice(BRANDS)
    mfg_name, mfg_addr, pin = random.choice(MANUFACTURERS)
    country = random.choice(COUNTRIES)

    is_compliant = (sample_id % 3 != 0)  # ~67% compliant, ~33% defective
    defect_type = None

    # Base valid attributes
    mrp_val = round(random.uniform(49.0, 1499.0), 2)
    mrp_text = f"MRP: Rs. {mrp_val:.2f} {random.choice(TAX_PHRASES)}"
    qty_val = random.choice([50, 100, 200, 250, 500, 750, 1000, 5])
    unit = random.choice(["ml", "g", "kg", "l", "number"])
    qty_text = f"Net Qty: {qty_val} {unit}"

    mfg_m = random.randint(1, 12)
    mfg_y = random.choice([2024, 2025, 2026])
    date_text = f"Mfg Date: {mfg_m:02d}/{mfg_y}"
    mfg_text = f"Mfg by: {mfg_name}, {mfg_addr} - {pin}"
    cc_text = f"Consumer Care: care@{brand.lower()}.com, Tel: 1800-425-{random.randint(1000, 9999)}"
    fssai_no = f"1{random.randint(1000000000000, 2999999999999)}"
    fssai_text = f"FSSAI Lic. No. {fssai_no}"

    # Inject specific defects
    ground_truth = {
        "mrp": 1,
        "net_quantity": 1,
        "mfg_date": 1,
        "country": 1,
        "manufacturer": 1,
        "consumer_care": 1,
        "fssai": 1 if category == "food_beverage" else 0,
    }

    if not is_compliant:
        defect_choice = random.choice([
            "missing_mrp_tax",
            "wrong_qty_unit",
            "invalid_date_format",
            "missing_country",
            "missing_mfg_address",
        ])
        defect_type = defect_choice

        if defect_choice == "missing_mrp_tax":
            mrp_text = f"MRP: {mrp_val:.2f}"  # Missing currency and tax phrase
            ground_truth["mrp"] = 0
        elif defect_choice == "wrong_qty_unit":
            qty_text = f"Net Qty: {qty_val} spoons"  # Non-standard unit
            ground_truth["net_quantity"] = 0
        elif defect_choice == "invalid_date_format":
            date_text = "Mfg Date: 2024-99-99"  # Invalid format
            ground_truth["mfg_date"] = 0
        elif defect_choice == "missing_country":
            country = ""  # Missing country
            ground_truth["country"] = 0
        elif defect_choice == "missing_mfg_address":
            mfg_text = f"Mfg by: {mfg_name}"  # Missing address & pin
            ground_truth["manufacturer"] = 0

    lines = [
        {"field": "mrp", "text": mrp_text},
        {"field": "net_quantity", "text": qty_text},
        {"field": "mfg_date", "text": date_text},
        {"field": "country", "text": country},
        {"field": "manufacturer", "text": mfg_text},
        {"field": "consumer_care", "text": cc_text},
    ]
    if category == "food_beverage":
        lines.append({"field": "fssai", "text": fssai_text})

    # Filter out blank lines
    lines = [l for l in lines if l["text"].strip()]
    random.shuffle(lines)

    # Optional noise corruptions to simulate OCR noise
    if random.random() < 0.35:
        # e.g. '2gg.00' or '200 m1' or 'l0/2024'
        for l in lines:
            if l["field"] == "mrp" and random.random() < 0.5:
                l["text"] = l["text"].replace("99", "gg").replace("Rs.", "R$")
            elif l["field"] == "net_quantity" and random.random() < 0.5:
                l["text"] = l["text"].replace("ml", "m1")
            elif l["field"] == "mfg_date" and random.random() < 0.5:
                l["text"] = l["text"].replace("10/", "l0/").replace("12/", "l2/")

    return {
        "sample_id": f"syn_{sample_id:04d}",
        "category": category,
        "is_compliant": is_compliant,
        "defect_type": defect_type,
        "ground_truth": ground_truth,
        "lines": lines,
    }

def render_label_image(spec: Dict[str, Any], output_path: str):
    """
    Renders the lines onto a simulated product package image with noise & layout variations.
    """
    width = random.choice([700, 800, 900])
    line_count = len(spec["lines"])
    height = max(400, line_count * 65 + 100)

    # Random background: white, cream, light grey
    bg_color = random.choice([(255, 255, 255), (250, 248, 240), (245, 245, 245), (252, 252, 250)])
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Use default font or truetype if available
    try:
        font = ImageFont.truetype("arial.ttf", size=random.choice([20, 22, 24]))
    except Exception:
        font = ImageFont.load_default()

    curr_y = random.randint(30, 60)
    for l in spec["lines"]:
        curr_x = random.randint(40, 80)
        text = l["text"]
        draw.text((curr_x, curr_y), text, fill=(20, 20, 20), font=font)
        curr_y += random.randint(45, 60)

    # Add slight blur/noise occasionally
    if random.random() < 0.20:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    img.save(output_path, quality=92)

def load_real_labeled_data() -> List[Dict[str, Any]]:
    """
    Loads any user-provided real labeled samples in data/real_labeled/*.json.
    """
    real_samples = []
    json_files = glob.glob(os.path.join(REAL_DIR, "*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                real_samples.append(data)
        except Exception as e:
            print(f"Notice: Could not load real sample {jf}: {e}")
    if real_samples:
        print(f"Loaded {len(real_samples)} real labeled samples from {REAL_DIR}")
    return real_samples

def generate_and_train(num_samples: int = 150):
    """
    Generates synthetic dataset, runs OCR, splits into train/val/test,
    and trains FieldClassifier and ComplianceConfidenceScorer models.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Generating {num_samples} synthetic label specifications...")
    specs = [create_sample_label_spec(i) for i in range(num_samples)]

    # Add real samples if available
    real_samples = load_real_labeled_data()
    all_specs = specs + real_samples
    random.shuffle(all_specs)

    ocr_extractor = OCRExtractor()
    dataset = []

    print(f"Rendering images and running OCR ({ocr_extractor.active_backend or 'fallback'})...")
    for idx, spec in enumerate(all_specs):
        img_name = f"{spec['sample_id']}.jpg"
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            render_label_image(spec, img_path)

        spec["image_path"] = img_path

        # Run OCR on the rendered image to capture real OCR noise & tokens
        ocr_res = ocr_extractor.extract_text(img_path)
        ocr_lines = []
        for r in ocr_res:
            for tok in r.tokens:
                # Find matching ground truth field label via word overlap
                matched_label = "other"
                tok_text = tok.text.lower()
                for line_spec in spec["lines"]:
                    f_name = line_spec["field"]
                    f_text = line_spec["text"].lower()
                    if any(w in f_text for w in tok_text.split() if len(w) > 3):
                        matched_label = f_name
                        break

                ocr_lines.append({
                    "text": tok.text,
                    "confidence": tok.confidence,
                    "bbox": tok.bbox.model_dump(),
                    "field_label": matched_label,
                })

        spec["ocr_lines"] = ocr_lines
        dataset.append(spec)

        if (idx + 1) % 30 == 0 or idx == len(all_specs) - 1:
            print(f"  Processed {idx + 1}/{len(all_specs)} samples")

    # Train / Val / Test Split: 70% / 15% / 15%
    n_total = len(dataset)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train + n_val]
    test_data = dataset[n_train + n_val:]

    print(f"\nDataset splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    with open(os.path.join(OUTPUT_DIR, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    # ─────────────────────────────────────────────────────────
    # Model Training: 1. FieldClassifier
    # ─────────────────────────────────────────────────────────
    print("\nTraining FieldClassifier...")
    train_texts = []
    train_labels = []
    train_bboxes = []

    for sample in train_data:
        # Use line specs and OCR lines for rich training
        for line in sample["lines"]:
            train_texts.append(line["text"])
            train_labels.append(line["field"])
            train_bboxes.append({"x_min": 50, "y_min": 100, "x_max": 500, "y_max": 150})
        for ocr_l in sample["ocr_lines"]:
            if ocr_l["text"].strip():
                train_texts.append(ocr_l["text"])
                train_labels.append(ocr_l["field_label"])
                train_bboxes.append(ocr_l["bbox"])

    field_clf = FieldClassifier()
    field_clf.fit(train_texts, train_labels, train_bboxes)
    clf_model_path = os.path.join(MODELS_DIR, "field_classifier.joblib")
    field_clf.save(clf_model_path)
    print(f"  FieldClassifier trained and saved to {clf_model_path}")

    # ─────────────────────────────────────────────────────────
    # Model Training: 2. ComplianceConfidenceScorer
    # ─────────────────────────────────────────────────────────
    print("\nTraining ComplianceConfidenceScorer...")
    X_scorer = []
    y_scorer = []

    checks_to_model = ["mrp", "net_quantity", "mfg_date", "country", "manufacturer"]
    for sample in train_data:
        gt = sample.get("ground_truth", {})
        for chk in checks_to_model:
            is_compliant = gt.get(chk, 1)
            # Find associated line text
            assoc_line = next((l["text"] for l in sample["lines"] if l["field"] == chk), "")
            check_feature_dict = {
                "l1_status": "PASS" if is_compliant else "FAIL",
                "evidence_match_score": 0.95 if is_compliant else 0.15,
                "ocr_confidence": 0.90 if is_compliant else 0.70,
                "evidence_text": assoc_line if is_compliant else "",
                "field_found": bool(assoc_line and is_compliant),
                "has_statutory_phrase": bool(is_compliant and any(w in assoc_line.lower() for w in ["tax", "incl", "ml", "g", "kg", "mfg", "india"])),
            }
            feat = ComplianceConfidenceScorer.extract_features(check_feature_dict)
            X_scorer.append(feat)
            y_scorer.append(1 if is_compliant else 0)

    X_scorer = np.array(X_scorer, dtype=np.float32)
    y_scorer = np.array(y_scorer, dtype=np.int32)

    scorer = ComplianceConfidenceScorer()
    scorer.fit(X_scorer, y_scorer)
    scorer_model_path = os.path.join(MODELS_DIR, "confidence_scorer.joblib")
    scorer.save(scorer_model_path)
    print(f"  ComplianceConfidenceScorer trained and saved to {scorer_model_path}")

    # Save metadata
    metadata = {
        "version": VERSION_TAG,
        "seed": RANDOM_SEED,
        "dataset_size": len(all_specs),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "models": {
            "field_classifier": clf_model_path,
            "confidence_scorer": scorer_model_path,
        },
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata written to {meta_path}")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    generate_and_train(count)

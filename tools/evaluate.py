import os
import sys
import json
from typing import Dict, List, Any, Tuple
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.schemas import (
    NormalizedScan,
    ProductCategory,
    CheckStatus,
    CheckResult,
    Severity,
)
from backend.rule_engine.validators.common import (
    validate_mrp,
    validate_net_quantity,
    validate_date_of_mfg,
    validate_country_of_origin,
    validate_manufacturer,
)
from backend.parser_service.normalizers import (
    normalize_mrp,
    normalize_net_quantity,
    normalize_dates,
)
from backend.ml_layer.field_classifier import FieldClassifier
from backend.ml_layer.fuzzy_normalizer import FuzzyNormalizer
from backend.ml_layer.confidence_scorer import ComplianceConfidenceScorer
from backend.verification.reconciler import VerificationReconciler

TEST_DATA_PATH = os.path.join("data", "synthetic_dataset", "test.json")
MODELS_DIR = "models"

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Computes Accuracy, Precision, Recall, F1, and False-Pass Rate (FPR).
    y_true: 1 = compliant (PASS), 0 = non-compliant (FAIL/REVIEW)
    y_pred: 1 = predicted PASS, 0 = predicted non-PASS
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = max(1, len(y_true))
    acc = (tp + tn) / total
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-6, (prec + rec))
    # False-Pass Rate: fraction of non-compliant labels that were erroneously passed
    fpr = fp / max(1, (fp + tn))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_pass_rate": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

def run_evaluation():
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Held-out test dataset not found at {TEST_DATA_PATH}")
        sys.exit(1)

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    print(f"Loaded {len(test_samples)} held-out test samples from {TEST_DATA_PATH}")

    # Load trained models
    field_clf_path = os.path.join(MODELS_DIR, "field_classifier.joblib")
    scorer_path = os.path.join(MODELS_DIR, "confidence_scorer.joblib")

    field_clf = FieldClassifier(field_clf_path if os.path.exists(field_clf_path) else None)
    fuzzy_norm = FuzzyNormalizer()
    scorer = ComplianceConfidenceScorer(scorer_path if os.path.exists(scorer_path) else None)
    reconciler = VerificationReconciler(fuzzy_norm, scorer)

    CHECKS = ["mrp", "net_quantity", "mfg_date", "country", "manufacturer"]

    # Collectors for (true, pred)
    y_true_all = []
    y_l1_all = []
    y_l2_all = []
    y_hyb_all = []

    per_check_metrics = {chk: {"true": [], "l1": [], "l2": [], "hyb": []} for chk in CHECKS}

    for sample in test_samples:
        gt = sample.get("ground_truth", {})
        category_str = sample.get("category", "general_retail")
        category = ProductCategory(category_str) if category_str in [c.value for c in ProductCategory] else ProductCategory.GENERAL_RETAIL

        # Build full text from sample lines
        full_text = " ".join(l["text"] for l in sample.get("lines", []))

        # ── 1. Layer 1 Processing ──
        # Extract fields via regex normalizers
        mrp_text = next((l["text"] for l in sample["lines"] if l["field"] == "mrp"), "")
        qty_text = next((l["text"] for l in sample["lines"] if l["field"] == "net_quantity"), "")
        date_text = next((l["text"] for l in sample["lines"] if l["field"] == "mfg_date"), "")
        coo_text = next((l["text"] for l in sample["lines"] if l["field"] == "country"), "")
        mfg_text = next((l["text"] for l in sample["lines"] if l["field"] == "manufacturer"), "")

        scan = NormalizedScan(
            scan_id=sample["sample_id"],
            category=category,
            manufacturer_name=mfg_text[:60] if mfg_text else None,
            manufacturer_address=mfg_text[60:] if len(mfg_text) > 60 else None,
            country_of_origin="India" if "india" in coo_text.lower() else (coo_text or None),
            mrp=normalize_mrp(mrp_text) if mrp_text else None,
            net_quantity=normalize_net_quantity(qty_text) if qty_text else None,
        )
        dates = normalize_dates(date_text)
        if dates:
            scan.dates.append(dates)

        conf_map = {"__ocr_available__": 1.0}

        l1_checks = {
            "mrp": validate_mrp(scan, 0.90, "01", "Rule_6_1_e", conf_map),
            "net_quantity": validate_net_quantity(scan, 0.90, "02", "Rule_6_1_c", conf_map),
            "mfg_date": validate_date_of_mfg(scan, 0.90, "03", "Rule_6_1_d", conf_map),
            "country": validate_country_of_origin(scan, 0.90, "04", "Rule_6_1_aa", conf_map),
            "manufacturer": validate_manufacturer(scan, 0.90, "05", "Rule_6_1_a", conf_map),
        }

        # ── 2. Layer 2 Processing & 3. Hybrid Reconciliation ──
        for chk in CHECKS:
            yt = gt.get(chk, 1)
            l1_res = l1_checks[chk]
            l1_pred = 1 if l1_res.status == CheckStatus.PASS else 0

            # L2 model prediction
            check_feature_dict = {
                "l1_status": l1_res.status.value,
                "evidence_match_score": 0.90 if l1_res.evidence_text else 0.10,
                "ocr_confidence": l1_res.confidence or 0.85,
                "evidence_text": l1_res.evidence_text or "",
                "field_found": bool(l1_res.evidence_text),
                "has_statutory_phrase": bool(
                    l1_res.evidence_text and any(w in l1_res.evidence_text.lower() for w in ["tax", "incl", "ml", "g", "kg", "mfg", "india"])
                ),
            }
            l2_score = scorer.predict_compliance_score(check_feature_dict)
            l2_pred = 1 if l2_score >= 0.50 else 0

            # Reconciled prediction
            reconciled_res = reconciler.reconcile_check(l1_res, raw_text=full_text, all_tokens_text=full_text)
            hyb_pred = 1 if reconciled_res.status == CheckStatus.PASS else 0

            # Record
            y_true_all.append(yt)
            y_l1_all.append(l1_pred)
            y_l2_all.append(l2_pred)
            y_hyb_all.append(hyb_pred)

            per_check_metrics[chk]["true"].append(yt)
            per_check_metrics[chk]["l1"].append(l1_pred)
            per_check_metrics[chk]["l2"].append(l2_pred)
            per_check_metrics[chk]["hyb"].append(hyb_pred)

    # ── Overall Summary ──
    m_l1 = compute_metrics(y_true_all, y_l1_all)
    m_l2 = compute_metrics(y_true_all, y_l2_all)
    m_hyb = compute_metrics(y_true_all, y_hyb_all)

    print("\n" + "=" * 80)
    print(" LABELGUARD HYBRID TWO-STEP VERIFICATION ARCHITECTURE — EVALUATION REPORT")
    print("=" * 80)

    print("\n### OVERALL COMPARISON (Held-out Test Split: N = %d evaluations)\n" % len(y_true_all))
    print("| Architecture Layer       | Accuracy | Precision | Recall | F1 Score | False-Pass Rate (FPR) |")
    print("|--------------------------|----------|-----------|--------|----------|-----------------------|")
    print("| Layer 1 (Rules only)     | %7.2f%% | %8.2f%% | %5.2f%% | %7.2f%% | %20.2f%% |" % (
        m_l1["accuracy"] * 100, m_l1["precision"] * 100, m_l1["recall"] * 100, m_l1["f1"] * 100, m_l1["false_pass_rate"] * 100
    ))
    print("| Layer 2 (ML Model only)  | %7.2f%% | %8.2f%% | %5.2f%% | %7.2f%% | %20.2f%% |" % (
        m_l2["accuracy"] * 100, m_l2["precision"] * 100, m_l2["recall"] * 100, m_l2["f1"] * 100, m_l2["false_pass_rate"] * 100
    ))
    print("| Hybrid (L1 + L2 + Recon) | %7.2f%% | %8.2f%% | %5.2f%% | %7.2f%% | %20.2f%% |" % (
        m_hyb["accuracy"] * 100, m_hyb["precision"] * 100, m_hyb["recall"] * 100, m_hyb["f1"] * 100, m_hyb["false_pass_rate"] * 100
    ))

    print("\n*Note: False-Pass Rate (FPR) is the most critical metric: a lower FPR ensures non-compliant labels are NEVER erroneously passed.*")

    print("\n### PER-CHECK COMPARISON\n")
    print("| Check Name          | L1 F1   | L2 F1   | Hybrid F1 | Hybrid FPR | Status / Improvement |")
    print("|---------------------|---------|---------|-----------|------------|----------------------|")
    for chk in CHECKS:
        d = per_check_metrics[chk]
        c_l1 = compute_metrics(d["true"], d["l1"])
        c_l2 = compute_metrics(d["true"], d["l2"])
        c_hyb = compute_metrics(d["true"], d["hyb"])
        note = "Optimal (Zero FPR)" if c_hyb["false_pass_rate"] == 0 else "Controlled"
        print("| %-19s | %6.2f%% | %6.2f%% | %8.2f%% | %9.2f%% | %-20s |" % (
            chk, c_l1["f1"] * 100, c_l2["f1"] * 100, c_hyb["f1"] * 100, c_hyb["false_pass_rate"] * 100, note
        ))

    print("\n" + "=" * 80)
    return m_l1, m_l2, m_hyb

if __name__ == "__main__":
    run_evaluation()

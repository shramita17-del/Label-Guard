import os
import numpy as np
import joblib
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

class ComplianceConfidenceScorer:
    """
    Component 3 of Layer 2 ML Model:
    Calibrated classifier (Platt scaling / sigmoid) predicting the probability
    that a given Legal Metrology compliance check is truly compliant.
    """

    FEATURE_NAMES = [
        "l1_status_val",         # 1.0 = PASS, 0.5 = UNCERTAIN, 0.0 = FAIL
        "evidence_match_score", # 0.0 to 1.0
        "ocr_confidence",       # 0.0 to 1.0
        "field_found",          # 1.0 or 0.0
        "has_statutory_phrase", # 1.0 or 0.0
        "text_length_norm",     # min(1.0, len / 50.0)
        "digit_count_norm",     # min(1.0, digits / 10.0)
    ]

    def __init__(self, model_path: Optional[str] = None):
        base_clf = LogisticRegression(C=1.0, random_state=42)
        self.calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
        self.is_trained = False
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    @classmethod
    def extract_features(cls, check_dict: Dict[str, Any]) -> np.ndarray:
        """
        Extracts 7 normalized numeric features from a check dictionary.
        """
        status = str(check_dict.get("l1_status", check_dict.get("status", "FAIL"))).upper()
        if "PASS" in status:
            l1_val = 1.0
        elif "UNCERTAIN" in status or "REVIEW" in status:
            l1_val = 0.5
        else:
            l1_val = 0.0

        ev_score = float(check_dict.get("evidence_match_score", 0.0))
        ocr_conf = float(check_dict.get("ocr_confidence", check_dict.get("confidence", 0.0)))
        field_found = 1.0 if check_dict.get("field_found", bool(check_dict.get("evidence_text"))) else 0.0
        statutory = 1.0 if check_dict.get("has_statutory_phrase", False) else 0.0

        ev_text = str(check_dict.get("evidence_text", ""))
        text_len = min(1.0, len(ev_text) / 50.0)
        digit_count = min(1.0, sum(c.isdigit() for c in ev_text) / 10.0)

        return np.array([
            l1_val,
            ev_score,
            ocr_conf,
            field_found,
            statutory,
            text_len,
            digit_count,
        ], dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains calibrated classifier on feature matrix X (N x 7) and binary labels y (1=compliant, 0=non-compliant).
        """
        self.calibrated_clf.fit(X, y)
        self.is_trained = True

    def predict_compliance_score(self, check_dict: Dict[str, Any]) -> float:
        """
        Returns calibrated compliance probability (0.0 to 1.0).
        """
        feats = self.extract_features(check_dict).reshape(1, -1)
        if self.is_trained:
            probs = self.calibrated_clf.predict_proba(feats)[0]
            # Probability of class 1 (compliant)
            return round(float(probs[1]), 3)

        # Fallback calibrated formula before training
        # Weighted combination of L1 status, OCR confidence, and evidence presence
        l1_val = feats[0, 0]
        ev_score = feats[0, 1]
        ocr_conf = feats[0, 2]
        field_found = feats[0, 3]

        score = (0.50 * l1_val) + (0.25 * ocr_conf) + (0.15 * ev_score) + (0.10 * field_found)
        return round(float(np.clip(score, 0.02, 0.98)), 3)

    def save(self, filepath: str):
        joblib.dump({
            "calibrated_clf": self.calibrated_clf,
            "feature_names": self.FEATURE_NAMES,
        }, filepath)
        self.model_path = filepath

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.calibrated_clf = data["calibrated_clf"]
        self.is_trained = True
        self.model_path = filepath

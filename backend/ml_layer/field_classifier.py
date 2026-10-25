import os
import re
import numpy as np
import joblib
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix

class FieldClassifier:
    """
    Component 1 of Layer 2 ML Model:
    Classifies an OCR text line (along with position, size, and layout features)
    into a target Legal Metrology field:
      - mrp
      - net_quantity
      - mfg_date
      - country
      - manufacturer
      - consumer_care
      - fssai
      - other
    Runs in < 2ms on CPU.
    """

    LABELS = [
        "mrp",
        "net_quantity",
        "mfg_date",
        "country",
        "manufacturer",
        "consumer_care",
        "fssai",
        "other",
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=1500,
            sublinear_tf=True
        )
        self.classifier = LogisticRegression(
            C=2.0,
            max_iter=500,
            class_weight="balanced",
            random_state=42
        )
        self.is_trained = False
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    # ─────────────────────────────────────────────────────────
    # Feature Extraction
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def extract_geometry_features(
        bbox: Optional[Dict[str, int]],
        img_width: int = 1000,
        img_height: int = 1000,
        line_idx: int = 0,
        total_lines: int = 1
    ) -> np.ndarray:
        """
        Computes 9 normalized spatial and layout features.
        """
        if not bbox:
            return np.zeros(9, dtype=np.float32)

        x_min = bbox.get("x_min", 0)
        y_min = bbox.get("y_min", 0)
        x_max = bbox.get("x_max", 0)
        y_max = bbox.get("y_max", 0)

        w = max(1, x_max - x_min)
        h = max(1, y_max - y_min)

        norm_x = min(1.0, max(0.0, x_min / max(1, img_width)))
        norm_y = min(1.0, max(0.0, y_min / max(1, img_height)))
        norm_w = min(1.0, max(0.0, w / max(1, img_width)))
        norm_h = min(1.0, max(0.0, h / max(1, img_height)))
        aspect = min(10.0, w / max(1.0, float(h)))

        rel_pos = line_idx / max(1, total_lines)
        is_top = 1.0 if norm_y < 0.33 else 0.0
        is_bottom = 1.0 if norm_y > 0.66 else 0.0
        is_left = 1.0 if norm_x < 0.4 else 0.0

        return np.array([norm_x, norm_y, norm_w, norm_h, aspect, rel_pos, is_top, is_bottom, is_left], dtype=np.float32)

    @staticmethod
    def extract_text_heuristics(text: str) -> np.ndarray:
        """
        Computes 8 lexical indicator features.
        """
        t = text.lower()
        num_chars = len(t)
        num_digits = sum(c.isdigit() for c in t)
        digit_ratio = num_digits / max(1, num_chars)

        has_currency = 1.0 if re.search(r'(?:rs|inr|₹|mrp)', t) else 0.0
        has_units = 1.0 if re.search(r'\b(?:\d+)\s*(?:g|kg|ml|l|pcs|number|units)\b', t) else 0.0
        has_date_delim = 1.0 if re.search(r'\b\d{1,2}[\/\-\.]\d{2,4}\b', t) else 0.0
        has_coo_word = 1.0 if re.search(r'(?:origin|india|made in)', t) else 0.0
        has_mfg_word = 1.0 if re.search(r'(?:mfg|mfd|manufactured|packed|marketed)', t) else 0.0
        has_phone_or_mail = 1.0 if re.search(r'(?:1800|care@|email|tel|phone)', t) else 0.0

        return np.array([
            min(1.0, num_chars / 100.0),
            min(1.0, digit_ratio),
            has_currency,
            has_units,
            has_date_delim,
            has_coo_word,
            has_mfg_word,
            has_phone_or_mail,
        ], dtype=np.float32)

    def featurize(self, texts: List[str], bboxes: Optional[List[Dict[str, int]]] = None) -> csr_matrix:
        """
        Builds combined feature matrix: TF-IDF char n-grams + geometry + heuristics.
        """
        tfidf_feats = self.vectorizer.transform(texts)

        dense_feats = []
        n = len(texts)
        for i, text in enumerate(texts):
            box = bboxes[i] if bboxes and i < len(bboxes) else None
            geom = self.extract_geometry_features(box, line_idx=i, total_lines=n)
            heur = self.extract_text_heuristics(text)
            dense_feats.append(np.concatenate([geom, heur]))

        dense_matrix = csr_matrix(np.array(dense_feats, dtype=np.float32))
        return hstack([tfidf_feats, dense_matrix], format="csr")

    # ─────────────────────────────────────────────────────────
    # Training & Persistence
    # ─────────────────────────────────────────────────────────

    def fit(self, texts: List[str], labels: List[str], bboxes: Optional[List[Dict[str, int]]] = None):
        """
        Fits TF-IDF vectorizer and classifier.
        """
        self.vectorizer.fit(texts)
        X = self.featurize(texts, bboxes)
        self.classifier.fit(X, labels)
        self.is_trained = True

    def save(self, filepath: str):
        joblib.dump({
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "labels": self.classifier.classes_ if hasattr(self.classifier, "classes_") else self.LABELS,
        }, filepath)
        self.model_path = filepath

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.vectorizer = data["vectorizer"]
        self.classifier = data["classifier"]
        self.is_trained = True
        self.model_path = filepath

    # ─────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────

    def predict_line(
        self,
        text: str,
        bbox: Optional[Dict[str, int]] = None,
        line_idx: int = 0,
        total_lines: int = 1
    ) -> Tuple[str, float]:
        """
        Predicts field category and confidence score for a single line.
        """
        if not self.is_trained:
            # Fallback heuristic if not trained
            heur_label = self._heuristic_fallback(text)
            return heur_label, 0.75

        geom = self.extract_geometry_features(bbox, line_idx=line_idx, total_lines=total_lines)
        heur = self.extract_text_heuristics(text)
        dense = csr_matrix(np.concatenate([geom, heur]).reshape(1, -1))
        tfidf = self.vectorizer.transform([text])
        X = hstack([tfidf, dense], format="csr")

        probs = self.classifier.predict_proba(X)[0]
        max_idx = int(np.argmax(probs))
        label = self.classifier.classes_[max_idx]
        conf = float(probs[max_idx])
        return label, round(conf, 3)

    def _heuristic_fallback(self, text: str) -> str:
        t = text.lower()
        if re.search(r'(?:mrp|rs|inr|₹|incl)', t):
            return "mrp"
        if re.search(r'(?:net\s*qty|quantity|\b\d+\s*(?:g|kg|ml|l|pcs)\b)', t):
            return "net_quantity"
        if re.search(r'(?:mfg|mfd|date|pkd|use by|exp)', t):
            return "mfg_date"
        if re.search(r'(?:country of origin|made in|india)', t):
            return "country"
        if re.search(r'(?:manufactured|packed|marketed|address|pincode)', t):
            return "manufacturer"
        if re.search(r'(?:care|customer|email|1800|consumer)', t):
            return "consumer_care"
        if re.search(r'\b[12]\d{13}\b', t):
            return "fssai"
        return "other"

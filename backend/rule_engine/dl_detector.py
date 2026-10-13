import os
import logging
from typing import Optional, Tuple
from PIL import Image

logger = logging.getLogger("labelguard.dl_detector")

# Default HuggingFace Vision Transformer fine-tuned on AI vs Real images
DEFAULT_MODEL_NAME = os.getenv("LABELGUARD_AI_DETECTOR_MODEL", "umm-maybe/AI-image-detector")

class DeepLearningAIDetector:
    """
    Deep Learning Vision Classifier for AI-Generated & Synthetic Media Detection.
    Uses a pretrained Vision Transformer (ViT) / CNN trained on millions of real
    and synthetic images to analyze latent pixel-space relationships and artifacts.
    """
    _instance: Optional["DeepLearningAIDetector"] = None

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = None
        self._is_loaded = False
        self._load_failed = False

    @classmethod
    def get_instance(cls) -> "DeepLearningAIDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self) -> bool:
        """Lazily loads the PyTorch / HuggingFace model into memory."""
        if self._is_loaded:
            return True
        if self._load_failed:
            return False

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading AI detection model '{self.model_name}' on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info(f"AI detection model '{self.model_name}' successfully loaded.")
            return True
        except Exception as err:
            logger.warning(f"Failed to load Deep Learning model '{self.model_name}': {err}")
            self._load_failed = True
            return False

    def predict(self, image: Image.Image) -> Tuple[Optional[float], Optional[str]]:
        """
        Runs neural vision inference on a PIL image.
        Returns:
            fake_score (float or None): Probability from 0.0 to 1.0 that the image is AI-generated.
            model_info (str or None): Name / details of the deep learning model used.
        """
        if not self._is_loaded:
            loaded = self.load_model()
            if not loaded or self.model is None or self.processor is None:
                return None, None

        try:
            import torch

            rgb_image = image.convert("RGB")
            inputs = self.processor(images=rgb_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]

            # In umm-maybe/AI-image-detector:
            # id2label is {0: 'artificial', 1: 'human'}
            id2label = getattr(self.model.config, "id2label", {0: "artificial", 1: "human"})
            fake_idx = 0
            for idx, label_name in id2label.items():
                if any(syn in label_name.lower() for syn in ["artificial", "fake", "ai", "synthetic"]):
                    fake_idx = int(idx)
                    break

            fake_prob = float(probs[fake_idx].item())
            fake_prob = max(0.0, min(1.0, fake_prob))
            return fake_prob, f"ViT ({self.model_name})"
        except Exception as err:
            logger.error(f"Inference error in DL detector: {err}")
            return None, None

def get_deep_learning_fake_score(image: Image.Image) -> Tuple[Optional[float], Optional[str]]:
    """Helper function to run DL inference on a PIL image."""
    detector = DeepLearningAIDetector.get_instance()
    return detector.predict(image)

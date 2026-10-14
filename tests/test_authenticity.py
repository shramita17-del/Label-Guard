import os
import sys
import unittest
from unittest.mock import MagicMock

if 'numpy' in sys.modules and isinstance(sys.modules['numpy'], MagicMock):
    del sys.modules['numpy']
import numpy as np
from PIL import Image, PngImagePlugin
from backend.rule_engine.authenticity import (
    evaluate_image_authenticity,
    compute_spectral_features,
    inspect_metadata,
    _check_string_for_ai_signatures,
    scan_binary_for_ai_manifests,
    scan_text_for_mockup_disclaimers,
    validate_ean13_checksum,
    check_barcode_integrity
)
from backend.schemas import CheckStatus, Severity

class TestAuthenticityModule(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.getcwd(), "scratch", "test_auth")
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_ai_signature_detection(self):
        prompts = [
            "masterpiece, packaging box for milk, 8k, detailed, stable diffusion xl",
            "hyperrealistic food label, midjourney v6 --ar 16:9",
            "DALL-E 3 generated packaged snack label",
            "ComfyUI workflow: seed 12345, cfg 7.5, sampler euler_a"
        ]
        for p in prompts:
            sigs = _check_string_for_ai_signatures(p)
            self.assertGreater(len(sigs), 0, f"Failed to detect AI signature in: {p}")

    def test_clean_camera_text_no_false_positive(self):
        clean_text = "Standard Sony A7 IV photo captured at ISO 100, 50mm f/1.8"
        sigs = _check_string_for_ai_signatures(clean_text)
        self.assertEqual(len(sigs), 0)

    def test_synthetic_image_with_png_metadata_fails(self):
        img_path = os.path.join(self.test_dir, "synthetic_ai_label.png")
        img = Image.new("RGB", (256, 256), color=(240, 240, 240))
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("parameters", "packaging box label, prompt: luxury organic tea, steps: 30, sampler: dpmpp_2m")
        img.save(img_path, pnginfo=png_info)

        check_res, details = evaluate_image_authenticity(img_path)

        self.assertEqual(check_res.status, CheckStatus.FAIL)
        self.assertEqual(check_res.severity, Severity.BLOCKING)
        self.assertTrue(details.is_ai_generated)
        self.assertGreaterEqual(details.ai_confidence_percentage, 80)
        self.assertGreater(len(details.ai_signatures), 0)
        self.assertIn("prompt:", details.ai_signatures[0].lower())

    def test_binary_c2pa_and_dalle_markers(self):
        img_path = os.path.join(self.test_dir, "c2pa_ai_image.jpg")
        with open(img_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"some_bytes_c2pa_manifest_jumbf_dall-e_generated" + b"\xff\xd9")

        found = scan_binary_for_ai_manifests(img_path)
        self.assertGreater(len(found), 0)
        self.assertTrue(any("c2pa" in m.lower() for m in found))

    def test_synthetic_flat_noise_flagged_as_ai(self):
        # A perfectly smooth computer-generated image without camera grain
        img_path = os.path.join(self.test_dir, "clean_render.png")
        img = Image.new("RGB", (400, 400), color=(255, 255, 255))
        # Draw clean geometric text-like lines
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 350, 350], outline=(0, 0, 0), width=4)
        draw.text((60, 60), "100% Synthetic Graphic Without Camera Noise", fill=(0, 0, 0))
        img.save(img_path)

        check_res, details = evaluate_image_authenticity(img_path)
        # Without camera hardware and with 0 sensor grain, it must be flagged as AI / synthetic media
        self.assertTrue(details.is_ai_generated)
        self.assertEqual(check_res.status, CheckStatus.FAIL)
        self.assertEqual(check_res.severity, Severity.BLOCKING)

    def test_spectral_features_execution_speed(self):
        synthetic_pattern = np.full((256, 256), 128, dtype=np.uint8)
        synthetic_pattern[::2, ::2] = 255
        synthetic_pattern[1::2, 1::2] = 0
        hf_ratio, asymmetry, lap_var = compute_spectral_features(synthetic_pattern)

        self.assertIsInstance(hf_ratio, float)
        self.assertIsInstance(asymmetry, float)
        self.assertIsInstance(lap_var, float)
        self.assertGreaterEqual(hf_ratio, 0.0)

    def test_deep_learning_arbiter_override_at_high_threshold(self):
        # Create an image that passes noise/FFT checks
        img_path = os.path.join(self.test_dir, "clean_looking_ai.png")
        img = Image.new("RGB", (256, 256), color=(140, 150, 160))
        img.save(img_path)

        # DL Model returns fake score 0.92 (> 0.85 threshold)
        check_res, details = evaluate_image_authenticity(img_path, dl_score_override=0.92)
        self.assertTrue(details.is_ai_generated)
        self.assertEqual(check_res.status, CheckStatus.FAIL)
        self.assertEqual(check_res.severity, Severity.BLOCKING)
        self.assertIn("DEEP LEARNING", details.classification_label)
        self.assertGreaterEqual(details.ai_confidence_percentage, 85)
        self.assertEqual(details.deep_learning_score, 0.92)

    def test_deep_learning_arbiter_passes_authentic_media(self):
        # Create an image that passes noise/FFT checks
        img_path = os.path.join(self.test_dir, "real_looking.png")
        img = Image.new("RGB", (256, 256), color=(140, 150, 160))
        img.save(img_path)

        # DL Model returns fake score 0.12 (<= 0.85 threshold)
        check_res, details = evaluate_image_authenticity(img_path, dl_score_override=0.12)
        self.assertFalse(details.is_ai_generated)
        self.assertEqual(check_res.status, CheckStatus.PASS)
        self.assertEqual(details.deep_learning_score, 0.12)
        self.assertIn("AUTHENTIC", details.classification_label)

    def test_c2pa_manifest_instant_short_circuit_100_percent(self):
        img_path = os.path.join(self.test_dir, "c2pa_manifest_test.jpg")
        with open(img_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"some_bytes_c2pa_actions_trainedalgorithmicmedia" + b"\xff\xd9")

        check_res, details = evaluate_image_authenticity(img_path)
        self.assertTrue(details.is_ai_generated)
        self.assertEqual(details.ai_confidence_percentage, 100)
        self.assertEqual(check_res.status, CheckStatus.FAIL)
        self.assertEqual(check_res.severity, Severity.BLOCKING)
        self.assertEqual(details.authenticity_score, 0.0)

    def test_fallback_on_corrupt_file(self):
        corrupt_path = os.path.join(self.test_dir, "corrupt.jpg")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_DATA")

        check_res, details = evaluate_image_authenticity(corrupt_path)
        self.assertIsNotNone(check_res)
        self.assertEqual(check_res.check_id, "AUTH_01")
        self.assertIsNotNone(details)

    def test_fix1_calibrated_dl_threshold_unverified_image(self):
        # Image has no camera hardware. Under old 0.85 threshold, score 0.72 was missed.
        # Under calibrated 0.65 threshold, it must trigger FAIL and is_ai=True.
        img_path = os.path.join(self.test_dir, "unverified_ai_packaging.png")
        img = Image.new("RGB", (256, 256), color=(140, 150, 160))
        img.save(img_path)

        check_res, details = evaluate_image_authenticity(img_path, dl_score_override=0.72)
        self.assertTrue(details.is_ai_generated)
        self.assertEqual(check_res.status, CheckStatus.FAIL)
        self.assertEqual(check_res.severity, Severity.BLOCKING)
        self.assertEqual(details.deep_learning_score, 0.72)
        self.assertIn("DEEP LEARNING OVERRIDE", details.classification_label)
        self.assertEqual(details.ai_confidence_percentage, 72)

    def test_fix2_semantic_mockup_disclaimer_scanner(self):
        # Tests detection of packaging mockup disclaimers
        text_sample = "SAMPLE PACKAGING MOCKUP FOR DESIGN & DEMO PURPOSES ONLY. NOT A REAL PRODUCT."
        disclaimers = scan_text_for_mockup_disclaimers(text_sample)
        self.assertGreater(len(disclaimers), 0)
        self.assertTrue(any("sample" in d and "mockup" in d for d in disclaimers))
        self.assertTrue(any("not a real product" in d for d in disclaimers))

        # Test short-circuit in evaluate_image_authenticity
        img_path = os.path.join(self.test_dir, "mockup_product.png")
        img = Image.new("RGB", (256, 256), color=(140, 150, 160))
        img.save(img_path)

        check_res, details = evaluate_image_authenticity(
            img_path,
            extracted_text="SAMPLE PACKAGING MOCKUP FOR DESIGN & DEMO PURPOSES ONLY. NOT A REAL PRODUCT."
        )
        self.assertTrue(details.is_ai_generated)
        self.assertEqual(details.ai_confidence_percentage, 100)
        self.assertEqual(check_res.status, CheckStatus.FAIL)
        self.assertEqual(check_res.severity, Severity.BLOCKING)
        self.assertIn("SYNTHETIC MOCKUP DISCLAIMER", details.classification_label)
        self.assertGreater(len(details.mockup_disclaimers), 0)

    def test_fix3_barcode_ean13_checksum_validation(self):
        # Valid EAN-13
        is_val, exp, act = validate_ean13_checksum("4006381333931")
        self.assertTrue(is_val)
        self.assertEqual(exp, 1)
        self.assertEqual(act, 1)

        # Invalid EAN-13 from brown AI packaging (printed check digit is 9, expected is 1)
        is_val, exp, act = validate_ean13_checksum("8906123456789")
        self.assertFalse(is_val)
        self.assertEqual(exp, 1)
        self.assertEqual(act, 9)

    def test_fix3_barcode_hallucination_flagged_in_image(self):
        img_path = os.path.join(self.test_dir, "packaging_with_barcode.png")
        img = Image.new("RGB", (256, 256), color=(140, 150, 160))
        img.save(img_path)

        # Spaced out sequence as seen on AI generated barcode: "8 906123 456789"
        check_res, details = evaluate_image_authenticity(
            img_path,
            extracted_text="Coco Crisp Barcode 8 906123 456789 Net Wt 100g"
        )
        self.assertTrue(details.is_ai_generated)
        self.assertEqual(check_res.status, CheckStatus.FAIL)
        self.assertEqual(check_res.severity, Severity.BLOCKING)
        self.assertIn("HALLUCINATED BARCODE", details.classification_label)
        self.assertIsNotNone(details.barcode_status)
        self.assertEqual(details.barcode_value, "8906123456789")

    def test_maybe_ai_generated_40_to_60_percent_confidence(self):
        # 40% to 60% confidence rating must be marked as Yellow Maybe AI Generated (CheckStatus.REVIEW)
        img_path = os.path.join(self.test_dir, "borderline_ai.png")
        img = Image.new("RGB", (256, 256), color=(140, 150, 160))
        img.save(img_path)

        for score in [0.42, 0.50, 0.58]:
            check_res, details = evaluate_image_authenticity(img_path, dl_score_override=score)
            self.assertFalse(details.is_ai_generated)
            self.assertEqual(check_res.status, CheckStatus.REVIEW, f"Score {score} was not marked as REVIEW")
            self.assertEqual(details.status, CheckStatus.REVIEW)
            self.assertEqual(check_res.severity, Severity.MAJOR)
            self.assertIn("MAYBE AI GENERATED", details.classification_label)
            self.assertEqual(details.ai_confidence_percentage, int(round(score * 100)))

if __name__ == "__main__":
    unittest.main()

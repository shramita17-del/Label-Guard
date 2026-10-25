import os
import unittest
from backend.ocr_service.extractor import OCRExtractor
from backend.parser_service.layout_extractor import (
    extract_mrp, extract_net_quantity, extract_fssai, extract_date, extract_manufacturer
)


class TestITCOCRRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_path = os.path.join("data", "regression", "itc_biscuit_back.jpg")
        if not os.path.exists(cls.image_path):
            raise unittest.SkipTest(f"ITC image not found at {cls.image_path}")

        cls.extractor = OCRExtractor()
        cls.results = cls.extractor.extract_text(cls.image_path)
        cls.all_tokens = []
        for r in cls.results:
            cls.all_tokens.extend(r.tokens)

    def test_expected_ocr_tokens_found(self):
        """Fails if fewer than 6 of the 9 expected strings are found in OCR tokens."""
        expected_strings = [
            "NET WEIGHT",
            "69 g",
            "PKD",
            "15/06/26",
            "Use By",
            "11/03/27",
            "MRP",
            "40.00",
            "Lic. No."
        ]

        found_strings = []
        missed_strings = []

        all_text = " ".join([t.text for t in self.all_tokens]).lower()

        for exp in expected_strings:
            if exp.lower() in all_text:
                found_strings.append(exp)
            else:
                missed_strings.append(exp)

        print(f"\n[ITC OCR Test] Found {len(found_strings)}/{len(expected_strings)}: {found_strings}")
        if missed_strings:
            print(f"[ITC OCR Test] Missed: {missed_strings}")

        self.assertGreaterEqual(
            len(found_strings), 6,
            f"Failed: Only {len(found_strings)} of {len(expected_strings)} found. Missed: {missed_strings}"
        )

    def test_layout_extractor_anchors_and_values(self):
        """Verifies layout extractor successfully parses key mandatory fields."""
        mrp = extract_mrp(self.all_tokens)
        net_qty = extract_net_quantity(self.all_tokens)
        fssai = extract_fssai(self.all_tokens)
        date = extract_date(self.all_tokens)
        mfg = extract_manufacturer(self.all_tokens)

        print(f"\n[ITC Layout Extraction]")
        print(f"  MRP: {mrp}")
        print(f"  Net Qty: {net_qty}")
        print(f"  FSSAI: {fssai}")
        print(f"  Date: {date}")
        print(f"  Manufacturer: {mfg}")

        self.assertIsNotNone(mrp)
        self.assertIn("40", mrp)
        self.assertIsNotNone(net_qty)
        self.assertIn("69", net_qty)
        self.assertEqual(fssai, "10012031000312")
        self.assertIsNotNone(date)
        self.assertIn("15/06/26", date)
        self.assertIsNotNone(mfg)
        self.assertIn("ITC", mfg.upper())


if __name__ == "__main__":
    unittest.main()

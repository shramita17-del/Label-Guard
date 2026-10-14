import unittest
import os
from PIL import Image
from backend.schemas import (
    ProductCategory, NormalizedScan, CheckStatus, Severity,
    MRPDeclaration, NetQuantity, DateDeclaration, FiberComposition, VegNonVegMark
)
from backend.rule_engine.engine import evaluate_packaging_compliance, build_compliance_report
from backend.rule_engine.authenticity import evaluate_image_authenticity

class TestComplianceCategoriesAndVerdicts(unittest.TestCase):
    def setUp(self):
        self.scratch_dir = os.path.join(os.getcwd(), "scratch", "test_scratch")
        os.makedirs(self.scratch_dir, exist_ok=True)

    def test_food_beverage_compliance_pass(self):
        scan = NormalizedScan(
            scan_id="FB_PASS_01",
            category=ProductCategory.FOOD_BEVERAGE,
            manufacturer_name="Tata Consumer Products Ltd",
            manufacturer_address="1, Bishop Lefroy Road, Kolkata",
            pincode="700020",
            country_of_origin="India",
            mrp=MRPDeclaration(amount=250.0, currency_symbol_found=True, tax_inclusive_phrase_found=True, raw_text="₹250.00 incl. of all taxes"),
            net_quantity=NetQuantity(value=500.0, unit="g", raw_text="Net Wt: 500g"),
            dates=[DateDeclaration(date_type="mfg", month=10, year=2026, raw_text="10/2026", is_valid_format=True)],
            veg_nonveg_mark=VegNonVegMark(shape_detected="circle", color_detected="green"),
            fssai_license_number="10012011000168"
        )
        conf_map = {
            "mrp_region": 0.95,
            "net_quantity_region": 0.95,
            "date_region": 0.95,
            "coo_region": 0.95,
            "manufacturer_region": 0.95,
            "veg_mark": 0.95,
            "fssai_region": 0.95
        }
        report = build_compliance_report(scan, conf_map)
        self.assertEqual(report.overall_verdict, CheckStatus.PASS)

        # Check all specific checks are executed
        check_ids = [c.check_id for c in report.check_results]
        self.assertIn("01", check_ids)  # MRP
        self.assertIn("02", check_ids)  # Net Quantity
        self.assertIn("03", check_ids)  # Date of Mfg
        self.assertIn("04", check_ids)  # Country of Origin
        self.assertIn("05", check_ids)  # Manufacturer
        self.assertIn("F12", check_ids) # Veg Mark
        self.assertIn("F14", check_ids) # FSSAI

    def test_food_beverage_fssai_uncached_triggers_review(self):
        scan = NormalizedScan(
            scan_id="FB_REV_01",
            category=ProductCategory.FOOD_BEVERAGE,
            manufacturer_name="Tata Consumer Products Ltd",
            country_of_origin="India",
            mrp=MRPDeclaration(amount=250.0, currency_symbol_found=True, tax_inclusive_phrase_found=True, raw_text="₹250.00 incl taxes"),
            net_quantity=NetQuantity(value=500.0, unit="g", raw_text="500g"),
            dates=[DateDeclaration(date_type="mfg", month=10, year=2026, raw_text="10/2026", is_valid_format=True)],
            veg_nonveg_mark=VegNonVegMark(shape_detected="circle", color_detected="green"),
            fssai_license_number="10014011001895"  # Uncached valid syntax
        )
        report = build_compliance_report(scan)
        fssai_res = next(c for c in report.check_results if c.check_id == "F14")
        self.assertEqual(fssai_res.status, CheckStatus.REVIEW)
        self.assertEqual(report.overall_verdict, CheckStatus.REVIEW)

    def test_food_beverage_fssai_inactive_triggers_blocking_fail(self):
        scan = NormalizedScan(
            scan_id="FB_FAIL_01",
            category=ProductCategory.FOOD_BEVERAGE,
            manufacturer_name="Old Foods Ltd",
            country_of_origin="India",
            mrp=MRPDeclaration(amount=100.0, currency_symbol_found=True, tax_inclusive_phrase_found=True, raw_text="₹100"),
            net_quantity=NetQuantity(value=100.0, unit="g", raw_text="100g"),
            dates=[DateDeclaration(date_type="mfg", month=1, year=2020, raw_text="01/2020", is_valid_format=True)],
            veg_nonveg_mark=VegNonVegMark(shape_detected="circle", color_detected="green"),
            fssai_license_number="11219331000123"  # Inactive in cache
        )
        report = build_compliance_report(scan)
        fssai_res = next(c for c in report.check_results if c.check_id == "F14")
        self.assertEqual(fssai_res.status, CheckStatus.FAIL)
        self.assertEqual(fssai_res.severity, Severity.BLOCKING)
        self.assertEqual(report.overall_verdict, CheckStatus.FAIL)

    def test_apparel_textile_compliance_pass(self):
        scan = NormalizedScan(
            scan_id="AT_PASS_01",
            category=ProductCategory.APPAREL_TEXTILE,
            manufacturer_name="Raymond Limited",
            country_of_origin="India",
            mrp=MRPDeclaration(amount=1299.0, currency_symbol_found=True, tax_inclusive_phrase_found=True, raw_text="MRP Rs. 1299.00 (inclusive of all taxes)"),
            net_quantity=NetQuantity(value=1.0, unit="number", raw_text="1 N"),
            dates=[DateDeclaration(date_type="mfg", month=9, year=2026, raw_text="09/2026", is_valid_format=True)],
            fiber_composition=FiberComposition(components={"cotton": 80.0, "linen": 20.0}, sums_to_100=True)
        )
        report = build_compliance_report(scan)
        self.assertEqual(report.overall_verdict, CheckStatus.PASS)
        fiber_res = next(c for c in report.check_results if c.check_id == "A02")
        self.assertEqual(fiber_res.status, CheckStatus.PASS)

    def test_apparel_textile_math_violation_triggers_fail(self):
        scan = NormalizedScan(
            scan_id="AT_FAIL_01",
            category=ProductCategory.APPAREL_TEXTILE,
            manufacturer_name="Fashion Mills",
            country_of_origin="India",
            mrp=MRPDeclaration(amount=799.0, currency_symbol_found=True, tax_inclusive_phrase_found=True, raw_text="₹799 incl taxes"),
            net_quantity=NetQuantity(value=1.0, unit="number", raw_text="1 N"),
            dates=[DateDeclaration(date_type="mfg", month=8, year=2026, raw_text="08/2026", is_valid_format=True)],
            fiber_composition=FiberComposition(components={"cotton": 70.0, "polyester": 20.0}, sums_to_100=False)
        )
        report = build_compliance_report(scan)
        fiber_res = next(c for c in report.check_results if c.check_id == "A02")
        self.assertEqual(fiber_res.status, CheckStatus.FAIL)
        self.assertIn("math violation", fiber_res.reason)

    def test_general_retail_routing_and_common_checks(self):
        scan = NormalizedScan(
            scan_id="GR_PASS_01",
            category=ProductCategory.GENERAL_RETAIL,
            manufacturer_name="Havells India Ltd",
            country_of_origin="India",
            mrp=MRPDeclaration(amount=499.0, currency_symbol_found=True, tax_inclusive_phrase_found=True, raw_text="Rs. 499 (incl. of all taxes)"),
            net_quantity=NetQuantity(value=1.0, unit="number", raw_text="1 Unit"),
            dates=[DateDeclaration(date_type="mfg", month=6, year=2026, raw_text="06/2026", is_valid_format=True)]
        )
        results = evaluate_packaging_compliance(scan)
        check_names = [c.check_name for c in results]
        self.assertIn("MRP", check_names)
        self.assertIn("Net Quantity", check_names)
        self.assertIn("Date of Mfg", check_names)
        self.assertIn("Country of Origin", check_names)
        self.assertIn("Manufacturer/Packer", check_names)
        # Food and Textile checks must not be present
        self.assertNotIn("Veg/Non-Veg Mark", check_names)
        self.assertNotIn("Fiber Composition", check_names)

    def test_manufacturer_low_confidence_triggers_review(self):
        scan = NormalizedScan(
            scan_id="MFR_REV_01",
            category=ProductCategory.GENERAL_RETAIL,
            manufacturer_name=None
        )
        # Low confidence on manufacturer region should trigger REVIEW, not immediate FAIL
        conf_map = {"manufacturer_region": 0.65}
        results = evaluate_packaging_compliance(scan, conf_map)
        mfr_res = next(c for c in results if c.check_id == "05")
        self.assertEqual(mfr_res.status, CheckStatus.REVIEW)
        self.assertIn("confidence below 85%", mfr_res.reason)

    def test_ai_generated_image_forces_verdict_fail(self):
        # Create dummy image containing AI generator signature
        img_path = os.path.join(self.scratch_dir, "dalle_sample.jpg")
        with open(img_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"some_dall-e_metadata" + b"\xff\xd9")

        auth_check, auth_details = evaluate_image_authenticity(img_path)
        self.assertTrue(auth_details.is_ai_generated)
        self.assertEqual(auth_check.status, CheckStatus.FAIL)
        self.assertEqual(auth_check.severity, Severity.BLOCKING)

        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == '__main__':
    unittest.main()

import unittest
from backend.schemas import NormalizedScan, ProductCategory, MRPDeclaration, DateDeclaration, FiberComposition, VegNonVegMark, CheckStatus
from backend.rule_engine.validators.common import validate_mrp, validate_date_of_mfg
from backend.rule_engine.validators.apparel_textile import validate_fiber_composition
from backend.rule_engine.validators.food_beverage import validate_veg_nonveg_mark

class TestRuleEngineValidators(unittest.TestCase):
    def setUp(self):
        self.mock_scan = NormalizedScan(
            scan_id="test1",
            category=ProductCategory.GENERAL_RETAIL,
            manufacturer_name=None,
            manufacturer_address=None,
            pincode=None,
            country_of_origin=None,
            generic_name=None,
            net_quantity=None,
            consumer_care=None,
            fssai_license_number=None,
            pdp_area_cm2=None,
            mrp=MRPDeclaration(amount=199.0, currency_symbol_found=True, tax_inclusive_phrase_found=True, raw_text="Rs.199 incl tax"),
            dates=[DateDeclaration(date_type="mfg", month=10, year=2024, raw_text="10/2024", is_valid_format=True)],
            fiber_composition=FiberComposition(components={"cotton": 60.0, "polyester": 40.0}, sums_to_100=True),
            veg_nonveg_mark=VegNonVegMark(
                shape_detected="circle", 
                color_detected="green", 
                measured_mm=5.0, 
                required_mm=None, 
                pdp_area_cm2=None
            )
        )

    def test_mrp_validator_high_confidence(self):
        res = validate_mrp(self.mock_scan, field_confidence=0.95, check_id="01", rule_ref="Rule_6")
        self.assertEqual(res.status, CheckStatus.PASS)

    def test_mrp_validator_low_confidence(self):
        # Should return REVIEW as per Rule 4
        res = validate_mrp(self.mock_scan, field_confidence=0.75, check_id="01", rule_ref="Rule_6")
        self.assertEqual(res.status, CheckStatus.REVIEW)
        self.assertIn("confidence below", res.reason)

    def test_fiber_composition_pass(self):
        res = validate_fiber_composition(self.mock_scan, field_confidence=0.99, check_id="A02", rule_ref="Textile")
        self.assertEqual(res.status, CheckStatus.PASS)

    def test_veg_mark_fail(self):
        # Modify scan to have invalid shape
        self.mock_scan.veg_nonveg_mark.shape_detected = "triangle"
        res = validate_veg_nonveg_mark(self.mock_scan, field_confidence=0.90, check_id="F12", rule_ref="Rule_7")
        self.assertEqual(res.status, CheckStatus.FAIL)

if __name__ == '__main__':
    unittest.main()

import unittest
from backend.schemas import NormalizedScan, DigitalListingJSON, ProductCategory, MRPDeclaration
from backend.rule_engine.engine import compare_scans

class TestMismatchFlow(unittest.TestCase):
    def test_mrp_mismatch_detection(self):
        # Physical box says Rs. 200, Acme Corp.
        phys = NormalizedScan(
            scan_id="test2",
            category=ProductCategory.GENERAL_RETAIL,
            mrp=MRPDeclaration(amount=200.0, currency_symbol_found=True, tax_inclusive_phrase_found=True, raw_text="200"),
            manufacturer_name="Acme Corp."
        )
        
        # Amazon listing says Rs. 250 (Overcharge Fraud), Acme Corp
        dig = DigitalListingJSON(
            source_url="http://amazon.in/test",
            mrp=MRPDeclaration(amount=250.0, currency_symbol_found=True, tax_inclusive_phrase_found=True, raw_text="250"),
            manufacturer_name="Acme Corp"
        )
        
        report = compare_scans(phys, dig)
        
        # Fraud should be detected
        self.assertTrue(report.overall_mismatch_detected)
        
        # Verify MRP flagged as mismatch
        mrp_field = next(f for f in report.fields_compared if f.field_name == "mrp")
        self.assertFalse(mrp_field.is_match)
        
        # Verify Manufacturer name passed fuzzy matching
        mfg_field = next(f for f in report.fields_compared if f.field_name == "manufacturer_name")
        self.assertTrue(mfg_field.is_match)

if __name__ == '__main__':
    unittest.main()

import unittest
from backend.schemas import NormalizedScan, ProductCategory, CheckStatus
from backend.rule_engine.fssai_verifier import verify_fssai_license

class TestFSSAIVerifier(unittest.TestCase):
    def setUp(self):
        self.scan_base = NormalizedScan(
            scan_id="test1",
            category=ProductCategory.FOOD_BEVERAGE,
            fssai_license_number="10012011000168" # Valid and in cache
        )

    def test_tier1_syntax_fail_length(self):
        self.scan_base.fssai_license_number = "100"
        res = verify_fssai_license(self.scan_base)
        self.assertEqual(res.status, CheckStatus.FAIL)
        self.assertIn("Tier 1 (Syntax)", res.reason)

    def test_tier1_syntax_fail_start_digit(self):
        self.scan_base.fssai_license_number = "30012011000168" # starts with 3 (invalid)
        res = verify_fssai_license(self.scan_base)
        self.assertEqual(res.status, CheckStatus.FAIL)
        self.assertIn("Tier 1 (Syntax)", res.reason)

    def test_tier3_cache_hit_active(self):
        self.scan_base.fssai_license_number = "10012011000168"
        res = verify_fssai_license(self.scan_base)
        self.assertEqual(res.status, CheckStatus.PASS)
        self.assertIn("Tier 3 (Cache)", res.reason)

    def test_tier3_cache_hit_inactive(self):
        self.scan_base.fssai_license_number = "11219331000123"
        res = verify_fssai_license(self.scan_base)
        self.assertEqual(res.status, CheckStatus.FAIL)
        self.assertIn("Tier 3 (Cache)", res.reason)

    def test_tier3_cache_miss_review(self):
        self.scan_base.fssai_license_number = "10000000000000" # valid syntax, not in cache
        res = verify_fssai_license(self.scan_base)
        # Should gracefully return REVIEW due to network timeout + cache miss
        self.assertEqual(res.status, CheckStatus.REVIEW)
        self.assertIn("Manual review required", res.reason)

if __name__ == '__main__':
    unittest.main()

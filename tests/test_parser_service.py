import unittest
from backend.parser_service.normalizers import (
    normalize_mrp,
    normalize_net_quantity,
    normalize_dates,
    normalize_fiber_composition
)

class TestParserService(unittest.TestCase):
    def test_mrp_normalizer(self):
        res = normalize_mrp("MRP Rs. 1499.00 (incl. of all taxes)")
        self.assertIsNotNone(res)
        self.assertEqual(res.amount, 1499.0)
        self.assertTrue(res.currency_symbol_found)
        self.assertTrue(res.tax_inclusive_phrase_found)

    def test_net_quantity_normalizer(self):
        res1 = normalize_mrp("100 pieces inside")
        # Net quantity should fail mrp
        self.assertIsNone(res1)
        
        res2 = normalize_net_quantity("Net Vol: 1.5 Litres")
        self.assertIsNotNone(res2)
        self.assertEqual(res2.value, 1.5)
        self.assertEqual(res2.unit, "l")
        
        res3 = normalize_net_quantity("500 g")
        self.assertEqual(res3.unit, "g")

    def test_date_normalizer(self):
        res1 = normalize_dates("MFD 08/2024")
        self.assertIsNotNone(res1)
        self.assertEqual(res1.month, 8)
        self.assertEqual(res1.year, 2024)
        self.assertTrue(res1.is_valid_format)

    def test_fiber_composition_normalizer(self):
        # Valid sums to 100
        res1 = normalize_fiber_composition("Material: 60% Cotton, 40% Polyester")
        self.assertIsNotNone(res1)
        self.assertTrue(res1.sums_to_100)
        self.assertEqual(res1.components['cotton'], 60.0)

        # Invalid sums
        res2 = normalize_fiber_composition("Material: 90% Cotton, 5% Spandex")
        self.assertIsNotNone(res2)
        self.assertFalse(res2.sums_to_100)

if __name__ == '__main__':
    unittest.main()

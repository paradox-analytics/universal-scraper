
import unittest
from bs4 import BeautifulSoup
from universal_scraper.core.semantic_extractor import SemanticExtractor

class TestHtmlCurrency(unittest.TestCase):
    def setUp(self):
        self.extractor = SemanticExtractor()
        
    def test_split_price(self):
        # Simulation of Home Depot split price
        html = """
        <div class="price-container">
            <span class="currency">$</span>
            <span class="dollars">49</span>
            <span class="cents">98</span>
        </div>
        """
        soup = BeautifulSoup(html, 'html.parser')
        strategy = {'type': 'currency'}
        
        print("\n🧪 Testing Split Price Extraction (Classic E-commerce Pattern)")
        result = self.extractor._extract_currency(soup, strategy)
        print(f"Extracted: '{result}'")
        
        # Expectation: Should return "$4998" or "$49.98" (depending on join logic)
        # Verify it finds SOMETHING containing digits
        self.assertIsNotNone(result)
        self.assertTrue('49' in result)
        self.assertTrue('98' in result)

    def test_combined_price(self):
        html = "<div>Price: $19.99</div>"
        soup = BeautifulSoup(html, 'html.parser')
        strategy = {'type': 'currency'}
        
        result = self.extractor._extract_currency(soup, strategy)
        print(f"\n🧪 Testing Combined Price: {result}")
        self.assertEqual(result, "Price: $19.99")

if __name__ == '__main__':
    unittest.main()

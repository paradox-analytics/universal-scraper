
import logging
import sys
import unittest
from unittest.mock import MagicMock, patch
from universal_scraper.core.data_validator import LLMDataValidator

# Mock context
class MockContext:
    def __init__(self, goal, fields):
        self.goal = goal
        self.fields = fields
        self.data_type = "product"
    
    def to_llm_prompt_section(self):
        return f"Goal: {self.goal}"

# Configure logs
logging.basicConfig(level=logging.INFO)

class TestItemFiltering(unittest.TestCase):
    def setUp(self):
        self.validator = LLMDataValidator(api_key="dummy_key", enable_cache=False)
        
        # Test Data: 1 Main Product, 3 Related Products
        self.items = [
            # Item 0: Main Product (Husky Jack)
            {
                "name": "Husky 2 Ton Hydraulic Trolley Car Jack",
                "price": "$149.00",
                "id": "311259745",
                "description": "The main product on the page."
            },
            # Item 1: Related Product (Stands)
            {
                "name": "Husky 3 Ton Jack Stands (Pair)",
                "price": "$49.00",
                "id": "rel_1",
                "type": "related_product"
            },
            # Item 2: Related Product (Creeper)
            {
                "name": "Husky Mechanics Creeper",
                "price": "$39.00",
                "id": "rel_2",
                "type": "related_product"
            }
        ]
    
    @patch('universal_scraper.core.data_validator.litellm.completion')
    def test_filter_main_product(self, mock_completion):
        print("\n🧪 Testing Semantic Item Filtering")
        
        # Mock LLM response to select ONLY index 0 (Main Product)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"matching_indices": [0], "reasoning": "Item 0 is the main product matching the description."}'
        mock_completion.return_value = mock_response
        
        target = "Product on this page, but no header data, only the core product"
        fields = ["name", "price"]
        
        print(f"Goal: {target}")
        
        filtered = self.validator.filter_items_by_target(self.items, target, fields)
        
        print(f"Items before: {len(self.items)}")
        print(f"Items after:  {len(filtered)}")
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['id'], "311259745")
        print("✅ Correctly filtered to main product.")

    @patch('universal_scraper.core.data_validator.litellm.completion')
    def test_filter_loose_match(self, mock_completion):
        print("\n🧪 Testing Loose Filtering (All items valid)")
        
        # Mock LLM response to select ALL items
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"matching_indices": [0, 1, 2], "reasoning": "All items are valid tools."}'
        mock_completion.return_value = mock_response
        
        target = "All tools listed on the page"
        fields = ["name", "price"]
        
        filtered = self.validator.filter_items_by_target(self.items, target, fields)
        
        self.assertEqual(len(filtered), 3)
        print("✅ Correctly kept all items for broad target.")

if __name__ == '__main__':
    unittest.main()

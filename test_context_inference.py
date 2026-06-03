
import unittest
from unittest.mock import MagicMock, patch
from universal_scraper.core.context_manager import ContextManager

class TestContextInference(unittest.TestCase):
    def setUp(self):
        self.cm = ContextManager(api_key="dummy")
        
    @patch('universal_scraper.core.context_manager.litellm.completion')
    def test_infer_products(self, mock_completion):
        # Mock LLM returning just type, no fields (simulating conservative LLM)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"data_type": "products", "fields": [], "confidence": 0.8}'
        mock_completion.return_value = mock_response
        
        print("\n🧪 Testing Product Inference (Fallback Logic)")
        start_prompt = "Extract products from this page"
        
        context = self.cm.parse_context(start_prompt)
        
        print(f"Goal: {start_prompt}")
        print(f"Inferred Type: {context.data_type}")
        print(f"Inferred Fields: {context.fields}")
        
        # Expectation: Fallback logic injected default fields
        self.assertEqual(context.data_type, "products")
        self.assertTrue(len(context.fields) > 0)
        self.assertIn("price", context.fields)
        self.assertIn("name", context.fields)
        print("✅ Correctly injected default fields for 'products'.")

    @patch('universal_scraper.core.context_manager.litellm.completion')
    def test_infer_general(self, mock_completion):
        # Mock LLM returning general type
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"data_type": "general_data", "fields": [], "confidence": 0.9}'
        mock_completion.return_value = mock_response
        
        print("\n🧪 Testing General Inference (No Fallback)")
        start_prompt = "Get all data"
        
        context = self.cm.parse_context(start_prompt)
        
        print(f"Goal: {start_prompt}")
        print(f"Inferred Type: {context.data_type}")
        print(f"Inferred Fields: {context.fields}")
        
        # Expectation: No fields (Auto-extraction)
        self.assertEqual(context.data_type, "general_data")
        self.assertEqual(len(context.fields), 0)
        print("✅ Correctly kept empty fields for 'general_data'.")

if __name__ == '__main__':
    unittest.main()

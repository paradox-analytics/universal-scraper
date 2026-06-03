
import asyncio
import unittest
from unittest.mock import MagicMock, patch
from universal_scraper.core.scraper import UniversalScraper

class TestHomeDepotFix(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_driven_extraction_no_crash(self):
        # 1. Setup Scraper with API key (to enable context manager)
        scraper = UniversalScraper(
            api_key="sk-dummy-key",
            enable_context_validation=True
        )
        
        # 2. Mock network and LLM responses
        mock_html = """
        <html>
            <body>
                <script id="__NEXT_DATA__" type="application/json">
                {
                    "props": {
                        "pageProps": {
                            "productData": {
                                "item": {
                                    "internet_number": "311259745",
                                    "model_number": "HPL4136-VT",
                                    "product_title": "2-Ton Hydraulic Trolley Car Jack",
                                    "offers": {
                                        "price": { "currency": "$", "value": 49.98 }
                                    }
                                }
                            }
                        }
                    }
                }
                </script>
            </body>
        </html>
        """
        
        # Mock ContextManager LLM response
        mock_context_json = {
            "data_type": "products",
            "fields": ["title", "url", "price", "model"],
            "confidence": 0.9,
            "reasoning": "User wants core product details"
        }
        
        # Mock litellm.completion globally
        with patch('litellm.completion') as mock_completion:
            
            def side_effect(*args, **kwargs):
                mock_resp = MagicMock()
                # If ContextManager is calling it (based on prompt or system message)
                if "Analyze this request and infer" in str(kwargs.get('messages', '')):
                    mock_resp.choices[0].message.content = str(mock_context_json).replace("'", '"')
                    return mock_resp
                
                # If AI Generator is calling it
                if "Code Generation" in str(kwargs.get('messages', '')):
                    mock_resp.choices[0].message.content = "def extract(html):\n  return []"
                    return mock_resp
                
                # Default
                mock_resp.choices[0].message.content = "{}"
                return mock_resp

            mock_completion.side_effect = side_effect
            
            with patch.object(scraper.html_fetcher, 'fetch', return_value={'html': mock_html, 'status': 200, 'url': 'https://www.homedepot.com/p/Husky-2-Ton-Hydraulic-Trolley-Car-Jack-HPL4136-VT/311259745' }):
                
                print("\n🧪 Verifying Home Depot Fix with actual scraper logic...")
                
                # Use the exact prompt reported by the user
                prompt = "Product \n Extract title, url, price and model from the base product"
                
                # Note: target=prompt is the correct argument for UniversalScraper.scrape
                try:
                    result = await scraper.scrape(
                        url="https://www.homedepot.com/p/Husky-2-Ton-Hydraulic-Trolley-Car-Jack-HPL4136-VT/311259745",
                        fields=None, # Trigger inference
                        target=prompt
                    )
                    
                    print(f"✅ Scraping finished. Success: {result.get('success', False)}")
                    data = result.get('data', [])
                    print(f"✅ Extracted items: {len(data)}")
                    
                    if data:
                        item = data[0]
                        print(f"✅ Extracted fields: {list(item.keys())}")
                        self.assertIn('title', item)
                        self.assertIn('price', item)
                        self.assertEqual(item['price'], 49.98)
                
                except Exception as e:
                    self.fail(f"Scraper crashed with error: {e}")

if __name__ == "__main__":
    unittest.main()

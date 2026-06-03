
import asyncio
import logging
from unittest.mock import MagicMock, patch
from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.context_manager import ExtractionContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reproduce_error():
    print("🚀 reproducing Home Depot Error with specific prompt...")
    
    # 1. Setup Scraper with mocks
    scraper = UniversalScraper(
        web_unblocker_api_key="dummy",
        headless=True
    )
    
    # Mock the fetcher to return generic Home Depot HTML with JSON
    # We want to test the PIPELINE, not the fetch
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
            <div class="price-container">
                <span class="currency">$</span>
                <span class="dollars">49</span>
                <span class="cents">98</span>
            </div>
        </body>
    </html>
    """
    
    # Mock context manager to return what we think the LLM returns for this prompt
    # Prompt: "Product \n Extract title, url, price and model from the base product"
    # Expected Inference: product type, fields=[title, url, price, model]
    mock_context = ExtractionContext(
        goal="Extract title, url, price and model from the base product",
        data_type="products",
        fields=["title", "url", "price", "model"],
        raw_prompt="Product \n Extract title, url, price and model from the base product"
    )
    
    with patch.object(scraper.context_manager, 'parse_context', return_value=mock_context):
        with patch.object(scraper.html_fetcher, 'fetch', return_value={'html': mock_html, 'status': 200}):
            
            # 2. Run Scraping
            try:
                result = await scraper.scrape(
                    url="https://www.homedepot.com/p/Husky-2-Ton-Hydraulic-Trolley-Car-Jack-HPL4136-VT/311259745",
                    fields=None, # Inferred from context
                    context="Product \n Extract title, url, price and model from the base product"
                )
                
                print("\n✅ Scraping Completed Successfully (No Crash)")
                print(f"Result count: {len(result)}")
                if result:
                    print(f"First item: {result[0]}")
                    
            except Exception as e:
                print(f"\n🚨 CRASH REPRODUCED: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(reproduce_error())

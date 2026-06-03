import asyncio
import logging
from universal_scraper.core.field_discovery import FieldDiscovery
from universal_scraper.core.semantic_extractor import SemanticExtractor
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOCK_HTML_MICRODATA = """
<html>
<body>
    <div itemscope itemtype="http://schema.org/Product">
        <h1 itemprop="name">Microdata Product</h1>
        <p itemprop="description">A product described using Microdata.</p>
        <div itemprop="offers" itemscope itemtype="http://schema.org/Offer">
            <span itemprop="price">$99.99</span>
            <meta itemprop="priceCurrency" content="USD" />
        </div>
        <div itemprop="brand" itemscope itemtype="http://schema.org/Brand">
            <span itemprop="name">MicroBrand</span>
        </div>
    </div>
</body>
</html>
"""

MOCK_HTML_RDFA = """
<html>
<body vocab="http://schema.org/">
    <div typeof="Product">
        <h1 property="name">RDFa Product</h1>
        <p property="description">A product described using RDFa.</p>
        <div property="offers" typeof="Offer">
            <span property="price">$49.99</span>
            <meta property="priceCurrency" content="USD" />
        </div>
        <div property="brand" typeof="Brand">
            <span property="name">RDFaBrand</span>
        </div>
    </div>
</body>
</html>
"""

async def test_microdata():
    logger.info("\n--- Testing Microdata Discovery ---")
    discovery = FieldDiscovery()
    results = await discovery.discover_fields(MOCK_HTML_MICRODATA, "http://example.com")
    logger.info(f"Discovered fields: {results['fields']}")
    logger.info(f"Source: {results['source']}")
    
    assert 'title' in results['fields']
    assert 'price' in results['fields']
    assert 'description' in results['fields']
    assert results['source'] == 'html_microdata'

    logger.info("\n--- Testing Microdata Extraction ---")
    extractor = SemanticExtractor()
    pattern = {
        "title": {"primary": {"type": "microdata", "field": "name"}},
        "price": {"primary": {"type": "microdata", "field": "price"}},
        "description": {"primary": {"type": "microdata", "field": "description"}}
    }
    
    # In a real scenario, we'd have containers. For this test, we'll use the whole doc.
    extracted = extractor.extract(MOCK_HTML_MICRODATA, pattern)
    logger.info(f"Extracted data: {extracted}")
    
    assert extracted[0]['title'] == 'Microdata Product'
    assert extracted[0]['price'] == '$99.99'
    assert extracted[0]['description'] == 'A product described using Microdata.'

async def test_rdfa():
    logger.info("\n--- Testing RDFa Discovery ---")
    discovery = FieldDiscovery()
    results = await discovery.discover_fields(MOCK_HTML_RDFA, "http://example.com")
    logger.info(f"Discovered fields: {results['fields']}")
    logger.info(f"Source: {results['source']}")
    
    assert 'title' in results['fields']
    assert 'price' in results['fields']
    assert 'description' in results['fields']
    assert results['source'] == 'html_rdfa'

    logger.info("\n--- Testing RDFa Extraction ---")
    extractor = SemanticExtractor()
    pattern = {
        "title": {"primary": {"type": "rdfa", "field": "name"}},
        "price": {"primary": {"type": "rdfa", "field": "price"}},
        "description": {"primary": {"type": "rdfa", "field": "description"}}
    }
    
    extracted = extractor.extract(MOCK_HTML_RDFA, pattern)
    logger.info(f"Extracted data: {extracted}")
    
    assert extracted[0]['title'] == 'RDFa Product'
    assert extracted[0]['price'] == '$49.99'
    assert extracted[0]['description'] == 'A product described using RDFa.'

if __name__ == "__main__":
    asyncio.run(test_microdata())
    asyncio.run(test_rdfa())
    logger.info("\n✅ All HTML Schema tests passed!")

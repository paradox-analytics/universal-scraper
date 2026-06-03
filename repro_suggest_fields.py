import asyncio
import logging
from universal_scraper.core.field_discovery import FieldDiscovery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reproduce():
    try:
        with open("debug_homedepot.html", "r") as f:
            html = f.read()
        
        url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-Steel-GNE27JYMFS/320244018"
        
        discovery = FieldDiscovery()
        logger.info("Starting field discovery...")
        result = await discovery.discover_fields(html, url, use_llm=False, target="products")
        
        logger.info(f"Discovery result: {result}")
    except Exception as e:
        logger.error(f"Discovery failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(reproduce())

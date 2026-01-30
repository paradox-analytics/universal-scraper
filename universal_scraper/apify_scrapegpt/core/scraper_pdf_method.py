"""
PDF scraping method for UniversalScraper
This is extracted to keep scraper.py manageable
"""

import time
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


async def scrape_pdf(self, url: str, fields: List[str], start_time: float) -> Dict[str, Any]:
    """
    Scrape data from PDF document

    Args:
        url: PDF URL
        fields: Fields to extract
        start_time: Start time for performance tracking

    Returns:
        Dict with 'data', 'metadata', 'source' keys
    """
    if not self.pdf_extractor:
        return {
            'success': False,
            'error': 'PDF extraction is not available. Install pymupdf: pip install pymupdf',
            'data': [],
            'metadata': {
                'execution_time': time.time() - start_time
            }
        }

    try:
        # Extract data from PDF using LLM
        items = await self.pdf_extractor.extract(
            pdf_url=url,
            fields=fields,
            context=self.extraction_context.goal if self.extraction_context else None
        )

        execution_time = time.time() - start_time

        logger.info(f" PDF extraction complete: {len(items)} items in {execution_time:.2f}s")

        return {
            'success': True,
            'data': items,
            'source': 'pdf_llm',
            'fetch_method': 'pdf_download',
            'metadata': {
                'url': url,
                'item_count': len(items),
                'fields_requested': fields,
                'execution_time': execution_time,
                'content_type': 'application/pdf'
            }
        }

    except Exception as e:
        logger.error(f" PDF extraction failed: {e}")
        import traceback
        traceback.print_exc()

        return {
            'success': False,
            'error': str(e),
            'data': [],
            'metadata': {
                'url': url,
                'execution_time': time.time() - start_time
            }
        }



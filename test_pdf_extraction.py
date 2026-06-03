"""
Test PDF extraction functionality
"""

import asyncio
import json
import os
from universal_scraper.core.scraper import UniversalScraper


async def test_pdf_simple():
    """Test with a simple PDF"""
    print("\n" + "="*80)
    print("TEST 1: Simple PDF (W3C Dummy PDF)")
    print("="*80)
    
    scraper = UniversalScraper(
        api_key=os.environ.get('OPENAI_API_KEY'),
        use_camoufox=False  # Not needed for PDFs
    )
    
    result = await scraper.scrape(
        url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        fields=["title", "content", "author"]
    )
    
    print(f"\nSuccess: {result.get('success')}")
    print(f"Source: {result.get('source')}")
    print(f"Items extracted: {len(result.get('data', []))}")
    print(f"Execution time: {result.get('metadata', {}).get('execution_time', 0):.2f}s")
    
    if result.get('data'):
        print(f"\nSample item:")
        print(json.dumps(result['data'][0], indent=2))
    
    await scraper.close()


async def test_pdf_financial():
    """Test with a financial report (if available)"""
    print("\n" + "="*80)
    print("TEST 2: Financial Report PDF")
    print("="*80)
    
    # This is a placeholder - replace with actual PDF URL
    pdf_url = "https://example.com/financial-report.pdf"
    
    print(f"⚠️  Skipping - replace with actual PDF URL: {pdf_url}")
    print("   Example: SEC 10-K report, annual report, invoice, etc.")


async def test_html_vs_pdf():
    """Test that HTML pages still work (not treated as PDFs)"""
    print("\n" + "="*80)
    print("TEST 3: HTML Page (verify HTML still works)")
    print("="*80)
    
    scraper = UniversalScraper(
        api_key=os.environ.get('OPENAI_API_KEY'),
        use_camoufox=False,
        use_direct_llm=True
    )
    
    result = await scraper.scrape(
        url="https://news.ycombinator.com/",
        fields=["title", "url", "points"]
    )
    
    print(f"\nSuccess: {result.get('success')}")
    print(f"Source: {result.get('source')}")
    print(f"Items extracted: {len(result.get('data', []))}")
    print(f"Content type: {result.get('metadata', {}).get('content_type', 'unknown')}")
    
    await scraper.close()


async def main():
    """Run all tests"""
    print("\n🧪 Testing PDF Extraction")
    print("="*80)
    
    # Check for API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        return
    
    # Run tests
    await test_pdf_simple()
    await test_pdf_financial()
    await test_html_vs_pdf()
    
    print("\n" + "="*80)
    print("✅ All tests complete")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())








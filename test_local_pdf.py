"""
Test PDF extraction with local document
"""

import asyncio
import json
import os
from universal_scraper.core.scraper import UniversalScraper


async def test_local_pdf():
    """Test with user's local PDF"""
    pdf_path = "/Users/jevon_williams/Dropbox/Mac/Downloads/Results1765046751854.pdf"
    
    print("\n" + "="*80)
    print("Testing PDF Extraction on Local Document")
    print("="*80)
    print(f"\nPDF: {pdf_path}")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"\n❌ File not found: {pdf_path}")
        return
    
    print(f"✅ File exists ({os.path.getsize(pdf_path) / 1024:.1f} KB)")
    
    # Initialize scraper
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("\n❌ OPENAI_API_KEY not set")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        return
    
    scraper = UniversalScraper(
        api_key=api_key,
        use_camoufox=False  # Not needed for PDFs
    )
    
    # First, let's do a general extraction to see what's in the PDF
    print("\n📄 Extracting data from PDF...")
    print("   Using auto-detection to find all data")
    
    # Convert file path to file:// URL
    pdf_url = f"file://{pdf_path}"
    
    # Try extracting with common fields first
    result = await scraper.scrape(
        url=pdf_url,
        fields=[
            "title", "name", "date", "amount", "total", 
            "description", "item", "product", "service",
            "company", "customer", "client"
        ]
    )
    
    print(f"\n✅ Extraction complete!")
    print(f"   Success: {result.get('success')}")
    print(f"   Source: {result.get('source')}")
    print(f"   Items extracted: {len(result.get('data', []))}")
    print(f"   Execution time: {result.get('metadata', {}).get('execution_time', 0):.2f}s")
    
    # Show results
    if result.get('data'):
        print(f"\n📊 Extracted Data:")
        print("="*80)
        
        for idx, item in enumerate(result['data'][:5], 1):  # Show first 5 items
            print(f"\nItem {idx}:")
            for key, value in item.items():
                if value:  # Only show non-empty values
                    print(f"  {key}: {value}")
        
        if len(result['data']) > 5:
            print(f"\n... and {len(result['data']) - 5} more items")
        
        # Save full results
        output_file = "pdf_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Full results saved to: {output_file}")
    
    elif result.get('error'):
        print(f"\n❌ Error: {result.get('error')}")
    
    await scraper.close()


async def main():
    await test_local_pdf()


if __name__ == "__main__":
    asyncio.run(main())








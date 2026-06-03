import asyncio
import os
import sys
import json
from bs4 import BeautifulSoup

# Add the project root to sys.path
sys.path.append(os.getcwd())

from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.hybrid_extractor import HybridMarkdownExtractor

async def reproduce():
    url = "https://www.producthunt.com/categories/vibe-coding"
    
    # Use HybridFetcher to get the HTML
    # Format: host:port:username:password
    brd_key = "brd.superproxy.io:22225:brd-customer-hl_803e8195-zone-web_unlocker1:t8mhp1qev1i1"
    fetcher = HybridFetcher(
        force_mode="browser",
        web_unblocker_api_key=brd_key
    )
    
    print(f"Fetching {url}...")
    result = await fetcher.fetch(url)
    html = result.get("html", "")
    
    if not html:
        print("Failed to fetch HTML")
        return

    print(f"Fetched {len(html)} bytes of HTML")
    
    # Run HybridMarkdownExtractor
    extractor = HybridMarkdownExtractor()
    extracted = extractor.extract(html, url=url)
    
    print("\n--- Markdown Preview (first 500 chars) ---")
    print(extracted.markdown[:500])
    
    print("\n--- Metadata Summary Preview ---")
    print(extracted.get_metadata_summary()[:500])
    
    # Check for non-printable characters
    def has_non_printable(s):
        return any(ord(c) < 32 and c not in '\n\r\t' for c in s)

    if has_non_printable(extracted.markdown):
        print("\nWARNING: Markdown contains non-printable characters!")
        # Find some examples
        for i, c in enumerate(extracted.markdown):
            if ord(c) < 32 and c not in '\n\r\t':
                print(f"Found non-printable at index {i}: {repr(c)}")
                if i > 10:
                    print(f"Context: {repr(extracted.markdown[i-10:i+10])}")
                break
    else:
        print("\nMarkdown seems clean of non-printable characters.")

    # Save to file for manual inspection
    with open("repro_output.md", "w") as f:
        f.write(extracted.markdown)
    
    with open("repro_metadata.txt", "w") as f:
        f.write(extracted.get_metadata_summary())

if __name__ == "__main__":
    asyncio.run(reproduce())

"""Debug HTML mismatch - what HTML does the scraper actually use?"""

import asyncio
import os
from universal_scraper import UniversalScraper
from bs4 import BeautifulSoup

async def debug():
    print("\n🔍 Debugging HTML mismatch\n")
    
    # Monkey-patch the AI generator to capture the HTML it receives
    captured_html = []
    
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        use_camoufox=False,
        headless=True,
        enable_auto_pagination=False
    )
    
    # Capture the generate_extraction_code method
    original_generate = scraper.ai_generator.generate_extraction_code
    
    def capture_and_generate(*args, **kwargs):
        if 'cleaned_html' in kwargs:
            captured_html.append(kwargs['cleaned_html'])
        return original_generate(*args, **kwargs)
    
    scraper.ai_generator.generate_extraction_code = capture_and_generate
    
    try:
        result = await scraper.scrape(
            url='https://stackoverflow.com/questions?tab=newest',
            fields=['title', 'votes']
        )
        
        items = result.get('data', [])
        print(f"📊 Scraper Results: {len(items)} items, votes={sum(1 for i in items if i.get('votes'))}/{len(items)}\n")
        
        # Now test the captured HTML directly
        if captured_html:
            html = captured_html[0]
            soup = BeautifulSoup(html, 'html.parser')
            
            print(f"🔍 Captured HTML Analysis:")
            print(f"   HTML length: {len(html):,} bytes")
            
            summaries = soup.select('div.s-post-summary')
            print(f"   Post summaries: {len(summaries)}")
            
            votes = soup.select('span[itemprop="upvoteCount"]')
            print(f"   Vote elements: {len(votes)}")
            
            if votes:
                print(f"   First vote value: '{votes[0].text.strip()}'")
            
            # Test the generated code against this HTML
            print(f"\n🧪 Testing generated code against captured HTML:")
            
            containers = soup.find_all('div', class_='s-post-summary js-post-summary')
            print(f"   Containers found: {len(containers)}")
            
            if containers:
                first = containers[0]
                title = first.select_one('h3 a')
                votes_elem = first.select_one('span.s-post-summary--stats-item-number[itemprop="upvoteCount"]')
                
                print(f"   First container:")
                print(f"      Title found? {title is not None}")
                print(f"      Votes elem found? {votes_elem is not None}")
                if votes_elem:
                    print(f"      Votes value: '{votes_elem.text.strip()}'")
                else:
                    print(f"      Available spans: {[s.get('itemprop') for s in first.select('span[itemprop]')]}")
        
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(debug())






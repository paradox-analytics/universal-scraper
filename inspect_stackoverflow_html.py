"""Inspect Stack Overflow HTML structure to see sibling layout"""

from universal_scraper.core.html_fetcher import HTMLFetcher
from bs4 import BeautifulSoup

def inspect():
    print("\n🔍 Inspecting Stack Overflow HTML Structure\n")
    
    fetcher = HTMLFetcher()
    
    result = fetcher.fetch('https://stackoverflow.com/questions?tab=newest')
    html = result['html']
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find first few question containers
    questions = soup.select('div.s-post-summary')[:3]
    
    print(f"Found {len(questions)} question containers\n")
    
    for i, q in enumerate(questions, 1):
        print(f"{'='*80}")
        print(f"QUESTION {i}")
        print(f"{'='*80}\n")
        
        # Show the parent structure
        parent = q.parent
        print(f"📦 Parent: {parent.name}.{'.'.join(parent.get('class', [])[:3])}")
        
        # Show main container
        print(f"\n📦 Main container: div.s-post-summary")
        title = q.select_one('h3.s-post-summary--content-title')
        if title:
            print(f"   ✅ Title: {title.get_text(strip=True)[:50]}...")
        
        # Look for votes in main container
        votes_in_container = q.select_one('span.s-post-summary--stats-item-number, div.s-post-summary--stats-item-number')
        print(f"   Votes in container? {votes_in_container is not None}")
        if votes_in_container:
            print(f"      Value: {votes_in_container.get_text(strip=True)}")
        
        # Check siblings
        print(f"\n📦 Siblings:")
        siblings = []
        next_sib = q.find_next_sibling()
        count = 0
        while next_sib and count < 3:
            if hasattr(next_sib, 'name'):
                sig = f"{next_sib.name}.{'.'.join(next_sib.get('class', [])[:3])}"
                siblings.append(sig)
                print(f"   → {sig}")
                count += 1
            next_sib = next_sib.find_next_sibling()
        
        if not siblings:
            print(f"   (no siblings)")
        
        # Show full HTML structure for first question
        if i == 1:
            print(f"\n📄 Full HTML (first 1500 chars):")
            print(f"{str(q)[:1500]}")
            print(f"...")
        
        print()

if __name__ == "__main__":
    inspect()


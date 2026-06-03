"""Test the generated code directly against Stack Overflow HTML"""

from universal_scraper.core.html_fetcher import HTMLFetcher
from bs4 import BeautifulSoup

def extract_data(soup):
    items = []
    
    # Find all post summary containers
    containers = soup.find_all('div', class_='s-post-summary js-post-summary')
    
    print(f"Found {len(containers)} containers\n")
    
    for i, elem in enumerate(containers[:5], 1):
        item = {}
        
        # Extract title
        title_elem = elem.select_one('h3 a')
        item['title'] = title_elem.text.strip() if title_elem else None
        
        # Extract votes
        votes_elem = elem.select_one('span.s-post-summary--stats-item-number[itemprop="upvoteCount"]')
        item['votes'] = votes_elem.text.strip() if votes_elem else None
        
        # Debug
        print(f"Item {i}:")
        print(f"  Title: {item['title'][:50] if item['title'] else None}...")
        print(f"  Votes: {item['votes']}")
        print(f"  votes_elem found? {votes_elem is not None}")
        
        if i == 1 and not votes_elem:
            # Show what's in the stats section
            stats = elem.select('span.s-post-summary--stats-item-number')
            print(f"\n  Found {len(stats)} stats elements:")
            for s in stats:
                print(f"    - {s.get('class')}, itemprop={s.get('itemprop')}: {s.text.strip()}")
        
        print()
        
        items.append(item)
    
    return items

# Fetch HTML
fetcher = HTMLFetcher()
result = fetcher.fetch('https://stackoverflow.com/questions?tab=newest')
soup = BeautifulSoup(result['html'], 'html.parser')

print("🔍 Testing Generated Code\n")
data = extract_data(soup)

print(f"\n📊 Final Results: {len(data)} items")
print(f"Votes extracted: {sum(1 for item in data if item.get('votes'))}/{len(data)}")






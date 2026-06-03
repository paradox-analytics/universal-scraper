"""Check if HTML cleaner is removing votes from Stack Overflow"""

from universal_scraper.core.html_fetcher import HTMLFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from bs4 import BeautifulSoup

# Fetch HTML
fetcher = HTMLFetcher()
result = fetcher.fetch('https://stackoverflow.com/questions?tab=newest')
raw_html = result['html']

print("🔍 Checking if HTML cleaner removes votes\n")

# Check raw HTML
soup_raw = BeautifulSoup(raw_html, 'html.parser')
votes_raw = soup_raw.select('span.s-post-summary--stats-item-number[itemprop="upvoteCount"]')
print(f"RAW HTML: Found {len(votes_raw)} vote elements")
if votes_raw:
    print(f"  Example: {votes_raw[0].text.strip()}")

# Clean HTML
cleaner = SmartHTMLCleaner()
cleaned_result = cleaner.clean(raw_html)
cleaned_html = cleaned_result.get('cleaned_html', '') if isinstance(cleaned_result, dict) else cleaned_result

# Check cleaned HTML
soup_cleaned = BeautifulSoup(cleaned_html, 'html.parser')
votes_cleaned = soup_cleaned.select('span.s-post-summary--stats-item-number[itemprop="upvoteCount"]')
print(f"\nCLEANED HTML: Found {len(votes_cleaned)} vote elements")
if votes_cleaned:
    print(f"  Example: {votes_cleaned[0].text.strip()}")

print(f"\n{'✅' if len(votes_raw) == len(votes_cleaned) else '❌'} Cleaner preserved votes: {len(votes_cleaned)}/{len(votes_raw)}")


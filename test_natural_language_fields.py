"""
Test Natural Language Field Generation

Demonstrates the universal field generation feature inspired by Oxylabs AI Scraper.
Users can describe what they want in plain English.
"""

import asyncio
import os
from universal_scraper import UniversalScraper

async def test_field_generation():
    print("\n" + "="*80)
    print("🤖 NATURAL LANGUAGE FIELD GENERATION TEST")
    print("="*80 + "\n")
    
    api_key = os.environ['OPENAI_API_KEY']
    
    # Test 1: E-commerce (product data)
    print("Test 1: E-commerce Products")
    print("-" * 80)
    prompt1 = "I want product names, prices in USD, star ratings, and customer review counts"
    print(f"Prompt: \"{prompt1}\"")
    
    fields1 = await UniversalScraper.generate_fields_from_prompt(
        prompt=prompt1,
        url="https://example.com/products",
        api_key=api_key
    )
    print(f"Generated Fields: {fields1}\n")
    
    # Test 2: Job board
    print("Test 2: Job Listings")
    print("-" * 80)
    prompt2 = "Get job titles, company names, locations, salaries, and posted dates"
    print(f"Prompt: \"{prompt2}\"")
    
    fields2 = await UniversalScraper.generate_fields_from_prompt(
        prompt=prompt2,
        url="https://indeed.com/jobs",
        api_key=api_key
    )
    print(f"Generated Fields: {fields2}\n")
    
    # Test 3: News articles
    print("Test 3: News Articles")
    print("-" * 80)
    prompt3 = "I need article headlines, authors, publication times, and article summaries"
    print(f"Prompt: \"{prompt3}\"")
    
    fields3 = await UniversalScraper.generate_fields_from_prompt(
        prompt=prompt3,
        url="https://techcrunch.com",
        api_key=api_key
    )
    print(f"Generated Fields: {fields3}\n")
    
    # Test 4: With descriptions
    print("Test 4: With Descriptions (Gaming)")
    print("-" * 80)
    prompt4 = "Extract game titles, developers, platforms, prices, and genres as arrays"
    print(f"Prompt: \"{prompt4}\"")
    
    fields4 = await UniversalScraper.generate_fields_from_prompt(
        prompt=prompt4,
        url="https://example.com/games",
        api_key=api_key,
        return_descriptions=True  # Get descriptions too
    )
    print("Generated Fields with Descriptions:")
    for field, desc in fields4.items():
        print(f"  • {field}: {desc}")
    print()
    
    print("="*80)
    print("✅ All tests complete!")
    print("="*80)

async def test_scrape_from_prompt():
    """Test the convenience method: scrape directly from prompt"""
    print("\n" + "="*80)
    print("🚀 SCRAPE FROM PROMPT TEST (Full Demo)")
    print("="*80 + "\n")
    
    api_key = os.environ['OPENAI_API_KEY']
    
    print("Using natural language to scrape Hacker News:")
    print('Prompt: "I want post titles, points, and comment counts"\n')
    
    result = await UniversalScraper.scrape_from_prompt(
        url="https://news.ycombinator.com/",
        prompt="I want post titles, points, and comment counts",
        api_key=api_key,
        use_camoufox=False  # Fast test
    )
    
    items = result.get('data', [])
    quality = result.get('quality', 0)
    
    print(f"📊 Results:")
    print(f"   Items: {len(items)}")
    print(f"   Quality: {quality:.0f}%")
    print(f"\n   Sample Items:")
    for i, item in enumerate(items[:3], 1):
        print(f"   {i}. {item}")
    
    print("\n" + "="*80)
    print("✅ Universal scraping from natural language works!")
    print("="*80)

if __name__ == "__main__":
    print("\n🎯 Testing Universal Features from Oxylabs AI Scraper\n")
    
    # Test 1: Field generation only
    asyncio.run(test_field_generation())
    
    # Test 2: Full scraping from prompt
    asyncio.run(test_scrape_from_prompt())






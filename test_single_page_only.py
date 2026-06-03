"""
True single-page test - NO pagination detection at all
Tests Phase 1 (HTML cleaning) + Phase 2 (code generation prompts)
"""
import asyncio
import time
import os
from universal_scraper.core.scraper import UniversalScraper

async def test_site(name, url, context):
    """Test a single page with no pagination"""
    print("\n" + "="*80)
    print(f"🧪 TESTING: {name}")
    print("="*80)
    print(f"URL: {url}")
    print(f"Context: {context}")
    print(f"Mode: SINGLE PAGE ONLY (no pagination)\n")
    
    start = time.time()
    
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found - context validation will be disabled")
    
    scraper = UniversalScraper(
        api_key=api_key,  # Explicitly pass API key
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context=context,
        enable_context_validation=True,
        log_level=20  # INFO level
    )
    
    # Monkey-patch the scraper to skip pagination detection
    original_scrape = scraper.scrape
    
    async def scrape_without_pagination(url, fields=None, **kwargs):
        """Wrapper that prevents pagination from running"""
        # Call original scrape but intercept pagination
        result = await original_scrape(url, fields=fields, **kwargs)
        return result
    
    # Actually, let me just call the core extraction without pagination
    # by directly calling the internal methods
    
    print("⏱️  Fetching HTML...")
    fetch_result = await scraper.html_fetcher.fetch(url)
    html = fetch_result['html']
    captured_json = fetch_result.get('captured_json', [])
    
    print(f"   ✅ Fetched {len(html):,} bytes")
    print(f"   ✅ Captured {len(captured_json)} JSON blob(s)")
    
    # Try JSON extraction first
    print("\n🔍 Detecting JSON sources...")
    json_results = scraper.json_detector.detect_and_extract(
        html, url, captured_json=captured_json
    )
    
    sources_found = len(json_results.get('sources', []))
    print(f"   ✅ Found {sources_found} JSON source(s)")
    
    # If we have context validation, use it
    extracted_data = []
    source_used = 'none'
    
    if hasattr(scraper, 'json_analyzer') and scraper.json_analyzer and sources_found > 0:
        print("\n🎯 Using context-driven JSON analysis...")
        
        # Prepare sources for ranking
        sources_list = json_results.get('sources', [])
        data_list = json_results.get('data', [])
        
        if sources_list and data_list:
            json_sources_dict = dict(zip(sources_list, data_list))
            
            # Rank sources
            try:
                rankings = scraper.json_analyzer.rank_sources(
                    json_sources=json_sources_dict,
                    url=url,
                    context=scraper.context_manager.context
                )
                
                if rankings:
                    print(f"   ✅ Ranked {len(rankings)} source(s)")
                    
                    # Try top source
                    top_source = rankings[0]
                    source_name = top_source['source']
                    confidence = top_source['confidence']
                    
                    print(f"\n🔍 Trying top source: {source_name} (confidence: {confidence:.2f})")
                    
                    # Extract from this source
                    source_data = json_sources_dict.get(source_name, [])
                    if isinstance(source_data, list):
                        extracted_data = source_data
                    elif isinstance(source_data, dict):
                        # Try to find arrays in the dict
                        for value in source_data.values():
                            if isinstance(value, list) and len(value) > 0:
                                extracted_data = value
                                break
                    
                    if extracted_data:
                        source_used = 'json'
                        print(f"   ✅ Extracted {len(extracted_data)} items from JSON")
            except Exception as e:
                print(f"   ⚠️  JSON analysis failed: {e}")
    
    # Fallback: Use simple JSON extraction if context analysis not available
    if not extracted_data and json_results.get('data'):
        print("\n🔍 Using simple JSON extraction (no context validation)...")
        extracted_data = json_results.get('data', [])
        if extracted_data:
            source_used = 'json'
            print(f"   ✅ Extracted {len(extracted_data)} items from JSON")
    
    # If no JSON data, try HTML code generation
    if not extracted_data:
        print("\n🧹 Cleaning HTML for code generation...")
        cleaned = scraper.html_cleaner.clean(html)
        cleaned_html = cleaned['html']
        
        print(f"   ✅ Cleaned: {cleaned['original_size']:,} → {cleaned['cleaned_size']:,} bytes ({cleaned['reduction_percent']:.1f}% reduction)")
        
        # Infer fields from context if available
        fields = []
        if hasattr(scraper, 'context_manager') and scraper.context_manager.context:
            fields = scraper.context_manager.context.fields or []
        
        print(f"\n🤖 Generating extraction code for {len(fields)} fields...")
        
        # Generate code
        context_str = None
        if hasattr(scraper, 'context_manager') and scraper.context_manager.context:
            context_str = scraper.context_manager.context.goal
        
        gen_result = scraper.ai_generator.generate_extraction_code(
            cleaned_html,
            fields,
            url,
            extraction_context=context_str
        )
        
        code = gen_result['code']
        print(f"   ✅ Generated {len(code)} chars of code")
        
        # Execute code
        print("\n⚡ Executing extraction code...")
        extracted_data = scraper._execute_extraction_code(code, html)
        
        if extracted_data:
            source_used = 'html'
            print(f"   ✅ Extracted {len(extracted_data)} items from HTML")
    
    elapsed = time.time() - start
    
    # Summary
    print("\n" + "="*80)
    print(f"⏱️  Completed in {elapsed:.1f} seconds")
    print("="*80)
    
    print(f"\n📊 RESULTS:")
    print(f"   Items extracted: {len(extracted_data)}")
    print(f"   Data source: {source_used}")
    print(f"   Success: {'✅ YES' if len(extracted_data) > 0 else '❌ NO'}")
    
    if len(extracted_data) > 0:
        print(f"\n📝 Sample (first 2 items):")
        for i, item in enumerate(extracted_data[:2], 1):
            print(f"\n   Item {i}:")
            for key, value in list(item.items())[:4]:
                value_str = str(value)[:60]
                print(f"      {key}: {value_str}")
    
    print("\n" + "="*80 + "\n")
    
    return {
        'name': name,
        'items': len(extracted_data),
        'source': source_used,
        'time': elapsed,
        'success': len(extracted_data) > 0
    }


async def main():
    print("\n" + "="*80)
    print("🔬 PHASE 1 + 2 VALIDATION - TRUE SINGLE PAGE TEST")
    print("="*80)
    print("Phase 1: HTML Cleaner (minify, don't remove)")
    print("Phase 2: Code Generation Prompts (few-shot + context)")
    print("="*80 + "\n")
    
    tests = [
        {
            'name': 'Reddit r/webscraping',
            'url': 'https://www.reddit.com/r/webscraping/',
            'context': 'Extract Reddit posts with title, author, upvotes, comments count'
        },
        {
            'name': 'Apify Homepage',
            'url': 'https://apify.com/',
            'context': 'Extract featured Actors/scrapers with their name, description, author'
        }
    ]
    
    results = []
    
    for test in tests:
        result = await test_site(test['name'], test['url'], test['context'])
        results.append(result)
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80 + "\n")
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"{status} {r['name']}: {r['items']} items from {r['source']} in {r['time']:.1f}s")
    
    success_count = sum(1 for r in results if r['success'])
    print(f"\n✅ Success rate: {success_count}/{len(results)}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())


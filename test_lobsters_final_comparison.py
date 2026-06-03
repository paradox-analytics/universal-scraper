#!/usr/bin/env python3
"""
Final test: Use langchain's Html2TextTransformer in our extraction
"""
import asyncio
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
import html2text


async def test_with_langchain_transformer():
    """Test extraction using langchain's transformer"""
    
    print("\n" + "="*80)
    print("🧪 TESTING WITH LANGCHAIN'S HTML2TEXTTRANSFORMER")
    print("="*80)
    print()
    
    url = "https://lobste.rs/"
    fields = ['title', 'points', 'comments']
    
    # Fetch
    fetcher = HybridFetcher(proxy_config=None, enable_cache=False, headless=True, use_camoufox=False)
    fetch_result = await fetcher.fetch(url)
    
    # Clean
    cleaner = SmartHTMLCleaner()
    clean_result = cleaner.clean(fetch_result['html'])
    cleaned_html = clean_result['html']
    
    print(f"✅ Fetched & cleaned: {len(cleaned_html):,} bytes\n")
    
    # Test 1: Our html2text (current - with ignore_links=False)
    print("="*80)
    print("Test 1: OUR html2text (ignore_links=False)")
    print("="*80)
    
    extractor_ours = DirectLLMExtractor(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o-mini",
        max_tokens_per_chunk=4000,
        quality_mode="balanced",
        use_html2text=True
    )
    
    items_ours = await extractor_ours.extract(
        cleaned_html,
        fields,
        context="Extract all story listings"
    )
    
    points_ours = sum(1 for item in items_ours if item.get('points') not in [None, '', 'N/A'])
    
    print(f"✅ Extracted: {len(items_ours)} items")
    print(f"   Points: {points_ours}/{len(items_ours)} ({points_ours/len(items_ours)*100:.0f}%)")
    print()
    
    # Test 2: Convert using langchain, then extract
    print("="*80)
    print("Test 2: LANGCHAIN Html2TextTransformer + Our Extractor")
    print("="*80)
    
    # Convert HTML using langchain's transformer
    transformer = Html2TextTransformer()
    doc = Document(page_content=cleaned_html)
    transformed_docs = transformer.transform_documents([doc])
    langchain_text = transformed_docs[0].page_content if transformed_docs else cleaned_html
    
    print(f"Converted to text: {len(langchain_text):,} chars")
    
    # Now extract from this text
    # Note: We need to disable html2text since we already converted
    extractor_langchain = DirectLLMExtractor(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o-mini",
        max_tokens_per_chunk=4000,
        quality_mode="balanced",
        use_html2text=False  # Already converted
    )
    
    items_langchain = await extractor_langchain.extract(
        langchain_text,
        fields,
        context="Extract all story listings"
    )
    
    points_langchain = sum(1 for item in items_langchain if item.get('points') not in [None, '', 'N/A'])
    
    print(f"✅ Extracted: {len(items_langchain)} items")
    print(f"   Points: {points_langchain}/{len(items_langchain)} ({points_langchain/len(items_langchain)*100:.0f}%)")
    print()
    
    # Test 3: No conversion (raw HTML)
    print("="*80)
    print("Test 3: RAW HTML (no conversion)")
    print("="*80)
    
    extractor_raw = DirectLLMExtractor(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o-mini",
        max_tokens_per_chunk=4000,
        quality_mode="balanced",
        use_html2text=False
    )
    
    items_raw = await extractor_raw.extract(
        cleaned_html,
        fields,
        context="Extract all story listings"
    )
    
    points_raw = sum(1 for item in items_raw if item.get('points') not in [None, '', 'N/A'])
    
    print(f"✅ Extracted: {len(items_raw)} items")
    print(f"   Points: {points_raw}/{len(items_raw)} ({points_raw/len(items_raw)*100:.0f}%)")
    print()
    
    # Comparison
    print("="*80)
    print("📊 FINAL COMPARISON")
    print("="*80)
    print()
    
    print(f"{'Method':<40} {'Items':<10} {'Points Coverage':<20}")
    print("-" * 80)
    print(f"{'Our html2text (ignore_links=False)':<40} {len(items_ours):<10} {points_ours}/{len(items_ours)} ({points_ours/len(items_ours)*100:.0f}%)")
    print(f"{'Langchain Html2TextTransformer':<40} {len(items_langchain):<10} {points_langchain}/{len(items_langchain)} ({points_langchain/len(items_langchain)*100:.0f}%)")
    print(f"{'Raw HTML (no conversion)':<40} {len(items_raw):<10} {points_raw}/{len(items_raw)} ({points_raw/len(items_raw)*100:.0f}%)")
    print()
    
    # Show samples
    print("Samples from each method:")
    print("-" * 80)
    
    print("\n1. Our html2text:")
    for i, item in enumerate(items_ours[:2], 1):
        print(f"   {i}. {item.get('title', '')[:50]} | points={item.get('points')} | comments={item.get('comments')}")
    
    print("\n2. Langchain:")
    for i, item in enumerate(items_langchain[:2], 1):
        print(f"   {i}. {item.get('title', '')[:50]} | points={item.get('points')} | comments={item.get('comments')}")
    
    print("\n3. Raw HTML:")
    for i, item in enumerate(items_raw[:2], 1):
        print(f"   {i}. {item.get('title', '')[:50]} | points={item.get('points')} | comments={item.get('comments')}")
    
    print()
    
    # Verdict
    print("="*80)
    print("🎯 CONCLUSION")
    print("="*80)
    print()
    
    if points_langchain > points_ours + 5:
        print("✅ LANGCHAIN's Html2TextTransformer SOLVES THE PROBLEM!")
        print(f"   • Langchain: {points_langchain} items with points")
        print(f"   • Our html2text: {points_ours} items with points")
        print(f"   • Improvement: +{points_langchain - points_ours} items")
        print()
        print("💡 SOLUTION: Replace our html2text with langchain's Html2TextTransformer")
    elif points_raw > points_ours + 5 and points_raw > points_langchain + 5:
        print("✅ RAW HTML WORKS BEST!")
        print(f"   • Raw HTML: {points_raw} items with points")
        print(f"   • Our html2text: {points_ours} items with points")
        print(f"   • Langchain: {points_langchain} items with points")
        print()
        print("💡 SOLUTION: Disable html2text for Lobsters (and similar sites)")
    elif all(abs(points_ours - p) < 5 for p in [points_langchain, points_raw]):
        print("⚪ ALL METHODS PERFORM SIMILARLY")
        print(f"   • Our html2text: {points_ours}/{len(items_ours)}")
        print(f"   • Langchain: {points_langchain}/{len(items_langchain)}")
        print(f"   • Raw HTML: {points_raw}/{len(items_raw)}")
        print()
        print("💡 The problem is likely chunking, quality filtering, or prompt")
    else:
        print(f"Results: Our={points_ours}, Langchain={points_langchain}, Raw={points_raw}")
        print("Need more investigation...")
    
    print()


if __name__ == "__main__":
    asyncio.run(test_with_langchain_transformer())




#!/usr/bin/env python3
"""
Compare langchain's Html2TextTransformer vs our html2text
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
import html2text
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document


async def compare_html2text_methods():
    """Compare different html2text approaches"""
    
    print("\n" + "="*80)
    print("🔬 COMPARING HTML-TO-TEXT METHODS")
    print("="*80)
    print()
    
    url = "https://lobste.rs/"
    
    # Fetch
    fetcher = HybridFetcher(proxy_config=None, enable_cache=False, headless=True, use_camoufox=False)
    fetch_result = await fetcher.fetch(url)
    
    # Clean
    cleaner = SmartHTMLCleaner()
    clean_result = cleaner.clean(fetch_result['html'])
    cleaned_html = clean_result['html']
    
    print(f"✅ Fetched & cleaned: {len(cleaned_html):,} bytes\n")
    
    # Extract just the first story for detailed comparison
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(cleaned_html, 'html.parser')
    first_story = soup.find('li', class_='story')
    
    if first_story:
        story_html = str(first_story)
        
        print("="*80)
        print("📄 ORIGINAL HTML (first story)")
        print("="*80)
        print(story_html[:500])
        print("...")
        print()
        
        # Method 1: Our html2text
        print("="*80)
        print("Method 1: OUR html2text library")
        print("="*80)
        
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = False
        h.body_width = 0
        h.single_line_break = True
        
        our_text = h.handle(story_html)
        print(our_text[:500])
        print("...")
        print()
        
        # Check if score is present
        if '76' in our_text or '61' in our_text:
            print("✅ Score (76 or 61) found in text")
        else:
            print("❌ Score NOT found in text")
        print()
        
        # Method 2: Langchain's Html2TextTransformer
        print("="*80)
        print("Method 2: LANGCHAIN's Html2TextTransformer")
        print("="*80)
        
        transformer = Html2TextTransformer()
        doc = Document(page_content=story_html)
        transformed = transformer.transform_documents([doc])
        langchain_text = transformed[0].page_content if transformed else ""
        
        print(langchain_text[:500])
        print("...")
        print()
        
        # Check if score is present
        if '76' in langchain_text or '61' in langchain_text:
            print("✅ Score (76 or 61) found in text")
        else:
            print("❌ Score NOT found in text")
        print()
        
        # Method 3: Raw HTML (no conversion)
        print("="*80)
        print("Method 3: RAW HTML (no conversion)")
        print("="*80)
        print(story_html[:500])
        print("...")
        print()
        
        if '76' in story_html or '61' in story_html:
            print("✅ Score found in HTML")
        else:
            print("❌ Score NOT found in HTML")
        print()
        
        # Analysis
        print("="*80)
        print("📊 DETAILED COMPARISON")
        print("="*80)
        print()
        
        print(f"{'Method':<30} {'Length':<10} {'Has Score':<15}")
        print("-" * 80)
        print(f"{'Original HTML':<30} {len(story_html):<10} {'Yes' if '76' in story_html or '61' in story_html else 'No':<15}")
        print(f"{'Our html2text':<30} {len(our_text):<10} {'Yes' if '76' in our_text or '61' in our_text else 'No':<15}")
        print(f"{'Langchain Html2Text':<30} {len(langchain_text):<10} {'Yes' if '76' in langchain_text or '61' in langchain_text else 'No':<15}")
        print()
        
        # Verdict
        print("="*80)
        print("🎯 VERDICT")
        print("="*80)
        print()
        
        our_has_score = '76' in our_text or '61' in our_text
        langchain_has_score = '76' in langchain_text or '61' in langchain_text
        
        if langchain_has_score and not our_has_score:
            print("🔵 Langchain's Html2TextTransformer PRESERVES the score!")
            print("   This is why ScrapeGraphAI works on Lobsters")
            print()
            print("💡 SOLUTION: Switch to langchain's Html2TextTransformer")
            print("   OR configure our html2text differently")
        elif our_has_score and not langchain_has_score:
            print("🟢 Our html2text PRESERVES the score better!")
            print("   But we still have issues - must be elsewhere")
        elif not our_has_score and not langchain_has_score:
            print("⚠️  BOTH lose the score during conversion!")
            print("   The problem must be elsewhere")
        else:
            print("🟢 BOTH preserve the score!")
            print("   The problem must be in extraction, not conversion")
        
        # Show exactly where the score is
        print()
        print("="*80)
        print("🔍 WHERE IS THE SCORE IN EACH VERSION?")
        print("="*80)
        print()
        
        # Find score in HTML
        if '76' in story_html:
            idx = story_html.find('76')
            print(f"In HTML around position {idx}:")
            print(f"  ...{story_html[max(0,idx-50):idx+50]}...")
            print()
        
        # Find score in our text
        if '76' in our_text:
            idx = our_text.find('76')
            print(f"In our text around position {idx}:")
            print(f"  ...{our_text[max(0,idx-50):idx+50]}...")
            print()
        else:
            print("❌ NOT in our text")
            print()
        
        # Find score in langchain text
        if '76' in langchain_text:
            idx = langchain_text.find('76')
            print(f"In langchain text around position {idx}:")
            print(f"  ...{langchain_text[max(0,idx-50):idx+50]}...")
            print()
        else:
            print("❌ NOT in langchain text")
            print()


if __name__ == "__main__":
    asyncio.run(compare_html2text_methods())




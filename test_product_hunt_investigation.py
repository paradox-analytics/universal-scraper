#!/usr/bin/env python3
"""
Investigate Product Hunt CSS selector issue
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
from universal_scraper.core.dom_pattern_detector import DOMPatternDetector
from bs4 import BeautifulSoup

async def main():
    print("\n" + "="*100)
    print("🔍 INVESTIGATING: Product Hunt CSS Selector Issue")
    print("="*100)
    
    fetcher = HybridFetcher(proxy_config=None, headless=True, use_camoufox=True, enable_cache=False)
    html_cleaner = SmartHTMLCleaner()
    dom_detector = DOMPatternDetector()
    
    url = "https://www.producthunt.com/"
    
    # Fetch
    print(f"📥 Fetching {url}...")
    result = await fetcher.fetch(url)
    html = result['html']
    print(f"✅ Fetched {len(html):,} bytes")
    print()
    
    # Clean
    print("🧹 Cleaning HTML...")
    cleaned_result = html_cleaner.clean(html)
    cleaned_html = cleaned_result['html']
    print(f"✅ Cleaned: {len(html):,} → {len(cleaned_html):,} bytes")
    print()
    
    # Detect patterns
    print("🔍 Detecting DOM patterns...")
    dom_patterns = dom_detector.detect_patterns(cleaned_html)
    best_pattern = dom_patterns.get('best_pattern')
    
    if best_pattern:
        print(f"✅ Best pattern found:")
        print(f"   Type: {best_pattern.get('type')}")
        print(f"   Confidence: {best_pattern.get('confidence', 0):.0%}")
        print(f"   Count: {best_pattern.get('count')} items")
        print()
        
        selector = best_pattern.get('selector', '')
        print(f"🎯 CSS Selector:")
        print(f"   {selector}")
        print()
        
        # Analyze selector
        print("🔬 Selector Analysis:")
        print(f"   Length: {len(selector)} characters")
        
        # Check for problematic patterns
        issues = []
        
        if 'has-[[' in selector:
            issues.append("❌ Contains 'has-[[' (double brackets - invalid CSS)")
        
        if selector.count('[') != selector.count(']'):
            issues.append("❌ Unbalanced brackets")
        
        if '\\:' in selector:
            issues.append("⚠️  Contains escaped colons (Tailwind CSS)")
        
        if len(selector) > 200:
            issues.append("⚠️  Very long selector (>200 chars)")
        
        if issues:
            print("   Issues found:")
            for issue in issues:
                print(f"      {issue}")
        else:
            print("   ✅ No obvious issues")
        
        print()
        
        # Try to use the selector
        print("🧪 Testing selector with BeautifulSoup...")
        try:
            soup = BeautifulSoup(cleaned_html, 'html.parser')
            containers = soup.select(selector)
            print(f"   ✅ Selector works! Found {len(containers)} containers")
        except Exception as e:
            print(f"   ❌ Selector failed: {e}")
            print()
            
            # Try to fix the selector
            print("🔧 Attempting to fix selector...")
            
            # Fix 1: Remove has-[[ patterns
            fixed_selector = selector.replace('has-[[', 'has-[')
            fixed_selector = fixed_selector.replace(']]\\:', ']\\:')
            
            print(f"   Fixed selector: {fixed_selector[:100]}...")
            
            try:
                containers = soup.select(fixed_selector)
                print(f"   ✅ Fixed selector works! Found {len(containers)} containers")
            except Exception as e2:
                print(f"   ❌ Still fails: {e2}")
                
                # Try simpler approach - find by tag and classes
                print()
                print("🔧 Trying simplified selector...")
                
                # Extract just the tag and first few classes
                parts = selector.split('.')
                if parts:
                    tag = parts[0] if parts[0] else 'div'
                    # Take first 3 classes
                    classes = [p.split('[')[0].split(':')[0] for p in parts[1:4] if p]
                    simple_selector = f"{tag}.{'.'.join(classes)}"
                    
                    print(f"   Simplified: {simple_selector}")
                    
                    try:
                        containers = soup.select(simple_selector)
                        print(f"   ✅ Simplified selector works! Found {len(containers)} containers")
                        
                        # Show sample
                        if containers:
                            print()
                            print("📋 Sample container:")
                            sample = containers[0]
                            print(f"   Tag: {sample.name}")
                            print(f"   Classes: {sample.get('class', [])[:5]}")
                            print(f"   Text preview: {sample.get_text()[:100]}")
                    except Exception as e3:
                        print(f"   ❌ Also fails: {e3}")
    else:
        print("❌ No pattern detected")
    
    print()
    print("="*100)

if __name__ == "__main__":
    asyncio.run(main())





#!/usr/bin/env python3
"""
Detailed comparison: Our extraction vs ScrapeGraphAI
Show exactly which items we extract and which we miss
"""
import asyncio
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from universal_scraper.core.scraper import UniversalScraper


# ScrapeGraphAI's extracted titles (from our earlier test)
SCRAPEGRAPHAI_TITLES = [
    "The Death of Arduino?",
    "Loose wire leads to blackout, contact with Francis Scott Key bridge",
    "Building more with GPT-5.1-Codex-Max",
    "Europe is scaling back GDPR and relaxing AI laws",
    "Researchers discover security vulnerability in WhatsApp",
    "Meta Segment Anything Model 3",
    "Static Web Hosting on the Intel N150: FreeBSD, SmartOS, NetBSD, OpenBSD and Linu",
    "It's your fault my laptop knows where I am",
    "Cognitive and mental health correlates of short-form video use",
    "Pozsar's Bretton Woods III: Sometimes Money Can't Solve the Problem",
    "How to identify a prime number without a computer",
    "Launch HN: Mosaic (YC W25) – Agentic Video Editing",
    "Screw it, I'm installing Linux",
    "Thunderbird adds native Microsoft Exchange email support",
    "Show HN: DNS Benchmark Tool – Compare and monitor resolvers",
    "Larry Summers resigns from OpenAI board",
    "A $1k AWS mistake",
    "Control LLM Spend and Access with any-LLM-gateway",
    "Exploring the limits of large language models as quant traders",
    "What Killed Perl?",
    "The Future of Programming (2013) [video]",
    "Comparing Integers and Doubles",
    "Reproducible C++ builds by logging Git hashes",
    "Racing karts on a Rust GPU kernel driver",
    "Multimodal Diffusion Language Models for Thinking-Aware Editing and Generation",
    "Netherlands returns control of Nexperia to Chinese owner",
    "The peaceful transfer of power in open source projects",
    "To launch something new, you need \"social dandelions\"",
    "Sam 3D: Powerful 3D Reconstruction for Physical World Images",
    "The Subversive Hyperlink"
]


async def test_comparison():
    """Compare our extraction with ScrapeGraphAI"""
    print("\n" + "="*100)
    print("🔍 DETAILED EXTRACTION COMPARISON")
    print("="*100)
    print()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)
    
    url = "https://news.ycombinator.com/"
    fields = ["title", "points", "comments"]
    
    # Test with balanced mode
    scraper = UniversalScraper(
        api_key=api_key,
        use_direct_llm=True,
        quality_mode="balanced",
        fetch_mode="hybrid",
        use_camoufox=False,
        enable_cache=False,
        enable_llm_pagination=False,
        enable_auto_pagination=False
    )
    
    print("📥 Fetching and extracting...")
    result = await scraper.scrape(
        url=url,
        fields=fields,
        force_html=True,
        scroll_to_bottom=False,
        click_load_more=None,
        wait_for_selector=None
    )
    
    await scraper.close()
    
    items = result.get('data', [])
    our_titles = [item.get('title', '') for item in items if item.get('title')]
    
    print(f"\n✅ Extracted {len(items)} items")
    print()
    
    # Show what we extracted
    print("="*100)
    print("📋 OUR EXTRACTION (Balanced Mode)")
    print("="*100)
    for i, item in enumerate(items, 1):
        title = item.get('title', 'N/A')
        points = item.get('points') if item.get('points') is not None else 'N/A'
        comments = item.get('comments') if item.get('comments') is not None else 'N/A'
        title_str = str(title)[:70] if title else 'N/A'
        print(f"{i:2}. {title_str:<70} | pts:{str(points):>4} | cmt:{str(comments):>4}")
    
    print()
    print("="*100)
    print("🆚 COMPARISON WITH SCRAPEGRAPHAI")
    print("="*100)
    print()
    
    # Find matches
    matched_titles = []
    for our_title in our_titles:
        # Fuzzy match (handle slight differences)
        for sg_title in SCRAPEGRAPHAI_TITLES:
            if our_title.lower().strip() in sg_title.lower().strip() or sg_title.lower().strip() in our_title.lower().strip():
                matched_titles.append(sg_title)
                break
    
    print(f"✅ Matched: {len(matched_titles)}/{len(SCRAPEGRAPHAI_TITLES)}")
    print(f"❌ Missed: {len(SCRAPEGRAPHAI_TITLES) - len(matched_titles)}")
    print()
    
    # Show missed items
    if len(matched_titles) < len(SCRAPEGRAPHAI_TITLES):
        print("="*100)
        print("❌ ITEMS WE MISSED (that ScrapeGraphAI found)")
        print("="*100)
        missed_items = []
        for i, sg_title in enumerate(SCRAPEGRAPHAI_TITLES, 1):
            if sg_title not in matched_titles:
                missed_items.append((i, sg_title))
                print(f"{i:2}. {sg_title}")
        
        print()
        print(f"Total missed: {len(missed_items)}")
        print()
        
        # Analyze why we might have missed them
        print("="*100)
        print("🔍 ANALYSIS: Why did we miss these items?")
        print("="*100)
        print()
        
        if len(missed_items) <= 7:
            print("Hypothesis: These might be:")
            print("  • Items at the bottom of the page (below fold)")
            print("  • Items added after ScrapeGraphAI's test")
            print("  • Items our quality filter rejected")
            print("  • Items with very low engagement (points/comments)")
            print()
            
            # Check if there's a pattern in missed items
            low_engagement_pattern = any(i <= 5 for i, _ in missed_items if i > 25)
            if low_engagement_pattern:
                print("📊 Pattern detected: Missed items appear to be at the bottom (items #25-30)")
                print("   Likely cause: Page content or our extraction is focusing on top stories")
        else:
            print("⚠️  Significant difference detected. Possible causes:")
            print("  • Page content changed between tests")
            print("  • Quality filtering too aggressive")
            print("  • LLM prompt needs adjustment")
    
    # Show extra items we got (that they didn't)
    our_extra = []
    for our_title in our_titles:
        matched = False
        for sg_title in SCRAPEGRAPHAI_TITLES:
            if our_title.lower().strip() in sg_title.lower().strip() or sg_title.lower().strip() in our_title.lower().strip():
                matched = True
                break
        if not matched:
            our_extra.append(our_title)
    
    if our_extra:
        print()
        print("="*100)
        print("➕ EXTRA ITEMS WE FOUND (that ScrapeGraphAI missed)")
        print("="*100)
        for i, title in enumerate(our_extra, 1):
            print(f"{i}. {title}")
    
    print()
    print("="*100)
    print("📊 QUALITY ASSESSMENT")
    print("="*100)
    print()
    
    # Check our data quality
    total_fields = len(items) * len(fields)
    filled_fields = sum(
        1 for item in items
        for field in fields
        if item.get(field) and str(item.get(field)).strip() and str(item.get(field)) != 'None'
    )
    
    completeness = (filled_fields / total_fields * 100) if total_fields > 0 else 0
    
    print(f"Our extraction:")
    print(f"  • Items: {len(items)}")
    print(f"  • Completeness: {completeness:.1f}%")
    print(f"  • Match rate: {len(matched_titles)}/{len(SCRAPEGRAPHAI_TITLES)} ({len(matched_titles)/len(SCRAPEGRAPHAI_TITLES)*100:.1f}%)")
    print()
    
    print(f"ScrapeGraphAI extraction:")
    print(f"  • Items: {len(SCRAPEGRAPHAI_TITLES)}")
    print(f"  • Completeness: 100.0% (all items had all fields)")
    print()
    
    # Verdict
    print("="*100)
    print("🎯 VERDICT")
    print("="*100)
    print()
    
    if len(matched_titles) >= 28:
        print("✅ EXCELLENT: We match 93%+ of ScrapeGraphAI's extraction")
        print("   The small difference is acceptable (page content changes, different timing)")
    elif len(matched_titles) >= 25:
        print("✅ GOOD: We match 83%+ of ScrapeGraphAI's extraction")
        print("   Minor improvement needed in prompt or quality filtering")
    elif len(matched_titles) >= 20:
        print("⚠️  FAIR: We match 67%+ of ScrapeGraphAI's extraction")
        print("   Need to investigate why we're missing 30%+ of items")
    else:
        print("❌ NEEDS IMPROVEMENT: We match <67% of ScrapeGraphAI's extraction")
        print("   Significant investigation needed")
    
    print()


if __name__ == "__main__":
    asyncio.run(test_comparison())


#!/usr/bin/env python3
"""
Test Data Quality - Deep dive into extracted data quality
Focus on accuracy, completeness, and usefulness
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict

script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner


def analyze_field_quality(items: List[Dict], field: str, field_type: str = "text") -> Dict:
    """
    Analyze quality of a single field across all items
    
    Args:
        items: List of extracted items
        field: Field name to analyze
        field_type: Expected type (text, number, url)
    
    Returns:
        Quality metrics for the field
    """
    if not items:
        return {"status": "NO_ITEMS"}
    
    values = [item.get(field) for item in items]
    non_null = [v for v in values if v is not None and str(v).strip() != '']
    
    # Basic metrics
    total = len(items)
    filled = len(non_null)
    fill_rate = (filled / total) * 100 if total > 0 else 0
    
    # Quality checks
    issues = []
    
    if field_type == "number":
        # Check if values look like numbers
        non_numeric = []
        for v in non_null:
            v_str = str(v).strip()
            # Remove common number separators
            v_clean = v_str.replace(',', '').replace('$', '').replace('%', '')
            if not any(c.isdigit() for c in v_clean):
                non_numeric.append(v_str)
        
        if non_numeric:
            issues.append(f"{len(non_numeric)} non-numeric values: {non_numeric[:3]}")
    
    elif field_type == "url":
        # Check if values look like URLs
        non_urls = []
        for v in non_null:
            v_str = str(v).strip()
            if not (v_str.startswith('http://') or v_str.startswith('https://') or v_str.startswith('/')):
                non_urls.append(v_str[:50])
        
        if non_urls:
            issues.append(f"{len(non_urls)} non-URL values: {non_urls[:3]}")
    
    elif field_type == "text":
        # Check for common garbage patterns
        garbage_patterns = ['_optimistic_', 'operationid', 'correlation', 'si=', 'byg_desktop']
        garbage_values = []
        
        for v in non_null:
            v_str = str(v).lower()
            if any(pattern in v_str for pattern in garbage_patterns):
                garbage_values.append(str(v)[:50])
        
        if garbage_values:
            issues.append(f"{len(garbage_values)} garbage values: {garbage_values[:3]}")
        
        # Check for suspiciously short/long values
        very_short = [v for v in non_null if len(str(v).strip()) < 3]
        very_long = [v for v in non_null if len(str(v).strip()) > 200]
        
        if len(very_short) > total * 0.2:
            issues.append(f"{len(very_short)} very short values (< 3 chars)")
        
        if len(very_long) > total * 0.1:
            issues.append(f"{len(very_long)} very long values (> 200 chars)")
    
    # Determine quality grade
    if fill_rate >= 90 and not issues:
        grade = "A"
        status = "EXCELLENT"
    elif fill_rate >= 70 and len(issues) <= 1:
        grade = "B"
        status = "GOOD"
    elif fill_rate >= 50:
        grade = "C"
        status = "ACCEPTABLE"
    else:
        grade = "F"
        status = "POOR"
    
    return {
        "field": field,
        "fill_rate": fill_rate,
        "filled": filled,
        "total": total,
        "issues": issues,
        "grade": grade,
        "status": status,
        "sample_values": [str(v)[:60] for v in non_null[:3]]
    }


async def test_source_quality(url: str, fields: List[str], field_types: Dict[str, str], name: str):
    """Test and analyze data quality for a single source"""
    print("\n" + "="*100)
    print(f"🔬 QUALITY TEST: {name}")
    print("="*100)
    print(f"URL: {url}")
    print(f"Fields: {fields}")
    print()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return None
    
    # Fetch and extract
    print("📥 Fetching HTML...")
    fetcher = HybridFetcher(proxy_config=None, headless=True, use_camoufox=True, enable_cache=False)
    result = await fetcher.fetch(url)
    html = result['html']
    print(f"✅ Fetched {len(html):,} bytes\n")
    
    print("🧹 Cleaning HTML...")
    cleaner = SmartHTMLCleaner()
    cleaned_result = cleaner.clean(html)
    cleaned_html = cleaned_result['html']
    print(f"✅ Cleaned: {len(cleaned_html):,} bytes\n")
    
    print("🤖 Extracting with DirectLLM...")
    extractor = DirectLLMExtractor(api_key=api_key)
    items = await extractor.extract(cleaned_html, fields)
    print(f"✅ Extracted {len(items)} items\n")
    
    if not items:
        print("❌ No items extracted!")
        return None
    
    # Analyze quality
    print("="*100)
    print("📊 DATA QUALITY ANALYSIS")
    print("="*100)
    print()
    
    overall_grades = []
    
    for field in fields:
        field_type = field_types.get(field, "text")
        quality = analyze_field_quality(items, field, field_type)
        
        grade_emoji = {
            "A": "✅",
            "B": "👍",
            "C": "⚠️",
            "F": "❌"
        }.get(quality['grade'], "❓")
        
        print(f"{grade_emoji} {field} (Grade {quality['grade']}): {quality['status']}")
        print(f"   Fill rate: {quality['fill_rate']:.1f}% ({quality['filled']}/{quality['total']})")
        
        if quality['issues']:
            for issue in quality['issues']:
                print(f"   ⚠️  {issue}")
        
        if quality['sample_values']:
            print(f"   Sample values:")
            for i, val in enumerate(quality['sample_values'], 1):
                print(f"      {i}. {val}")
        
        print()
        overall_grades.append(quality['grade'])
    
    # Overall grade
    grade_scores = {"A": 4, "B": 3, "C": 2, "F": 0}
    avg_score = sum(grade_scores.get(g, 0) for g in overall_grades) / len(overall_grades)
    
    if avg_score >= 3.5:
        overall = "A - EXCELLENT"
        emoji = "🌟"
    elif avg_score >= 2.5:
        overall = "B - GOOD"
        emoji = "👍"
    elif avg_score >= 1.5:
        overall = "C - ACCEPTABLE"
        emoji = "⚠️"
    else:
        overall = "F - POOR"
        emoji = "❌"
    
    print("="*100)
    print(f"{emoji} OVERALL QUALITY: {overall}")
    print("="*100)
    print()
    
    # Show actual extracted items
    print("📋 Sample Extracted Items:")
    print("-" * 100)
    for i, item in enumerate(items[:3], 1):
        print(f"\nItem {i}:")
        for key, value in item.items():
            value_str = str(value)[:80] if value else "(empty)"
            print(f"  • {key}: {value_str}")
    
    if len(items) > 3:
        print(f"\n... and {len(items) - 3} more items")
    
    print()
    
    return {
        "name": name,
        "items_count": len(items),
        "overall_grade": overall,
        "field_grades": {field: quality['grade'] for field, quality in zip(fields, [analyze_field_quality(items, f, field_types.get(f, "text")) for f in fields])},
        "data": items
    }


async def main():
    print("\n" + "="*100)
    print("🔬 DATA QUALITY DEEP DIVE")
    print("="*100)
    print()
    
    print("Focus: Extracting HIGH-QUALITY, ACCURATE, USEFUL data")
    print("Not just quantities, but real-world usability")
    print()
    
    # Test cases with expected field types
    test_cases = [
        {
            "url": "https://news.ycombinator.com/",
            "fields": ["article_title", "points", "author", "comments_count"],
            "field_types": {
                "article_title": "text",
                "points": "number",
                "author": "text",
                "comments_count": "number"
            },
            "name": "Hacker News"
        },
        {
            "url": "https://www.producthunt.com/",
            "fields": ["product_name", "tagline", "upvotes"],
            "field_types": {
                "product_name": "text",
                "tagline": "text",
                "upvotes": "number"
            },
            "name": "Product Hunt"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        result = await test_source_quality(
            test_case["url"],
            test_case["fields"],
            test_case["field_types"],
            test_case["name"]
        )
        
        if result:
            results.append(result)
        
        # Pause between tests
        await asyncio.sleep(2)
    
    # Final summary
    print("\n" + "="*100)
    print("📊 QUALITY SUMMARY")
    print("="*100)
    
    for result in results:
        print(f"\n{result['name']}: {result['overall_grade']}")
        print(f"   Items: {result['items_count']}")
        print(f"   Field grades:")
        for field, grade in result['field_grades'].items():
            grade_emoji = {"A": "✅", "B": "👍", "C": "⚠️", "F": "❌"}.get(grade, "❓")
            print(f"      {grade_emoji} {field}: {grade}")
    
    print()
    
    # Check if all passed
    all_excellent = all(result['overall_grade'].startswith("A") for result in results)
    
    if all_excellent:
        print("🌟 EXCELLENT! All sources produced high-quality data.")
        print("   Ready for production use.")
    else:
        print("⚠️  Some sources need attention.")
        print("   Review issues and refine extraction logic.")
    
    print()


if __name__ == "__main__":
    asyncio.run(main())





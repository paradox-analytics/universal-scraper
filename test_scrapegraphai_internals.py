#!/usr/bin/env python3
"""
Investigate ScrapeGraphAI's internal processing for Lobsters
"""
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# First, let's look at what they actually extract
from scrapegraphai.graphs import SmartScraperGraph
import json


def test_scrapegraphai_lobsters_verbose():
    """Test ScrapeGraphAI on Lobsters with detailed output"""
    
    print("\n" + "="*80)
    print("🔍 INVESTIGATING SCRAPEGRAPHAI'S LOBSTERS EXTRACTION")
    print("="*80)
    print()
    
    # Configure with verbose
    graph_config = {
        "llm": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.environ.get('OPENAI_API_KEY'),
        },
        "verbose": True,  # See what they're doing
        "headless": True,
    }
    
    url = "https://lobste.rs/"
    prompt = "Extract all story listings with title, points, and comments count"
    
    print(f"URL: {url}")
    print(f"Prompt: {prompt}")
    print()
    print("="*80)
    print("Running ScrapeGraphAI (verbose mode)...")
    print("="*80)
    print()
    
    # Run ScrapeGraphAI
    smart_scraper = SmartScraperGraph(
        prompt=prompt,
        source=url,
        config=graph_config
    )
    
    result = smart_scraper.run()
    
    # Extract items
    items = []
    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, list):
                items = value
                break
    elif isinstance(result, list):
        items = result
    
    print()
    print("="*80)
    print("📊 SCRAPEGRAPHAI RESULTS")
    print("="*80)
    print()
    
    print(f"Total items: {len(items)}")
    print()
    
    if items:
        # Analyze what fields they extracted
        print("Field names in results:")
        if items:
            all_keys = set()
            for item in items:
                all_keys.update(item.keys())
            print(f"  {', '.join(sorted(all_keys))}")
        print()
        
        # Check completeness
        points_fields = ['points', 'score', 'votes', 'upvotes', 'Points', 'Score']
        comments_fields = ['comments', 'comment_count', 'comments_count', 'Comments']
        
        points_count = 0
        comments_count = 0
        points_field_used = None
        comments_field_used = None
        
        for item in items:
            # Check which field they use for points
            for field in points_fields:
                if field in item and item[field] not in [None, '', 'N/A']:
                    points_count += 1
                    points_field_used = field
                    break
            
            # Check which field they use for comments
            for field in comments_fields:
                if field in item and item[field] not in [None, '', 'N/A']:
                    comments_count += 1
                    comments_field_used = field
                    break
        
        print(f"Data completeness:")
        print(f"  • Title: {len(items)}/{len(items)} (100%)")
        print(f"  • Points/Score: {points_count}/{len(items)} ({points_count/len(items)*100:.0f}%)")
        if points_field_used:
            print(f"    → Field name used: '{points_field_used}'")
        print(f"  • Comments: {comments_count}/{len(items)} ({comments_count/len(items)*100:.0f}%)")
        if comments_field_used:
            print(f"    → Field name used: '{comments_field_used}'")
        print()
        
        # Show detailed samples
        print("Detailed samples (first 5 items):")
        print("-" * 80)
        for i, item in enumerate(items[:5], 1):
            print(f"\n{i}. Full item data:")
            print(json.dumps(item, indent=2))
        print()
    
    print()
    print("="*80)
    print("🔍 KEY OBSERVATIONS")
    print("="*80)
    print()
    
    if points_count > 20:  # If they got most points
        print(f"✅ They successfully extracted {points_count}/{len(items)} points")
        if points_field_used:
            print(f"   → Using field name: '{points_field_used}'")
        print()
        print("💡 How they might be doing it:")
        print("   1. They might NOT use html2text (keeping raw HTML)")
        print("   2. They might have better HTML preprocessing")
        print("   3. Their Parse Node might preserve link text")
        print("   4. Their prompt might be more effective")
    else:
        print(f"⚠️  They also struggle with points: {points_count}/{len(items)}")
        print("   This suggests it's a challenging extraction regardless of approach")
    
    return items


def inspect_scrapegraphai_parse_node():
    """Read and analyze their Parse Node implementation"""
    
    print("\n" + "="*80)
    print("📖 SCRAPEGRAPHAI PARSE NODE ANALYSIS")
    print("="*80)
    print()
    
    parse_node_path = "/Users/jevon_williams/Library/Python/3.9/lib/python/site-packages/scrapegraphai/nodes/parse_node.py"
    
    try:
        with open(parse_node_path, 'r') as f:
            content = f.read()
        
        print("Key findings from parse_node.py:")
        print("-" * 80)
        
        # Check if they use html2text
        if 'Html2TextTransformer' in content:
            print("✅ They DO use Html2TextTransformer")
            
            # Find the configuration
            if 'parse_html' in content:
                print("   • Has 'parse_html' config option")
            
            if 'transform_documents' in content:
                print("   • Converts HTML to text using Html2TextTransformer")
            
            print()
            print("Relevant code snippet:")
            print("-" * 80)
            
            # Extract the relevant section
            lines = content.split('\n')
            in_execute = False
            for i, line in enumerate(lines):
                if 'def execute' in line:
                    in_execute = True
                    start = max(0, i - 2)
                
                if in_execute and i >= start and i < start + 35:
                    print(line)
                
                if in_execute and i >= start + 35:
                    break
            
            print("-" * 80)
        else:
            print("❌ They DON'T use Html2TextTransformer")
        
        print()
        
        # Check their html2text configuration
        print("Html2TextTransformer configuration:")
        print("-" * 80)
        
        # Look for Html2TextTransformer imports and usage
        if 'Html2TextTransformer()' in content:
            print("✅ Uses default Html2TextTransformer() with no custom config")
            print()
            print("💡 This means they use langchain's default html2text settings")
            print("   Let's check what those defaults are...")
            print()
            
            # Try to import and inspect
            try:
                from langchain_community.document_transformers import Html2TextTransformer
                transformer = Html2TextTransformer()
                
                print("Html2TextTransformer attributes:")
                for attr in dir(transformer):
                    if not attr.startswith('_'):
                        try:
                            val = getattr(transformer, attr)
                            if not callable(val):
                                print(f"  • {attr}: {val}")
                        except:
                            pass
            except Exception as e:
                print(f"   Couldn't inspect: {e}")
        
        print()
        
    except FileNotFoundError:
        print("❌ Couldn't find parse_node.py")
        print(f"   Looked at: {parse_node_path}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def main():
    """Run investigation"""
    
    # Part 1: See what they actually extract
    items = test_scrapegraphai_lobsters_verbose()
    
    # Part 2: Inspect their code
    inspect_scrapegraphai_parse_node()
    
    # Part 3: Summary
    print("\n" + "="*80)
    print("🎯 FINAL ANALYSIS")
    print("="*80)
    print()
    
    if items:
        points_count = 0
        for item in items:
            for field in ['points', 'score', 'votes', 'upvotes', 'Points', 'Score']:
                if field in item and item[field] not in [None, '', 'N/A']:
                    points_count += 1
                    break
        
        if points_count > 20:
            print(f"✅ ScrapeGraphAI successfully extracts {points_count}/{len(items)} points from Lobsters")
            print()
            print("Possible reasons:")
            print("  1. Langchain's Html2TextTransformer might preserve link text better")
            print("  2. Their prompt engineering might be more effective")
            print("  3. Their chunking strategy might work better for this site")
            print("  4. They might have site-specific handling")
            print()
            print("💡 To match them, we should:")
            print("  • Test langchain's Html2TextTransformer vs our html2text")
            print("  • OR disable html2text for sites with data in links")
            print("  • OR improve our html2text configuration")
        else:
            print(f"⚠️  ScrapeGraphAI also struggles: {points_count}/{len(items)} points")
            print()
            print("This suggests Lobsters is challenging for both systems")
    
    print()


if __name__ == "__main__":
    main()




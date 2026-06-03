#!/usr/bin/env python3
"""
Test ScrapeGraphAI's approach on our failing sources
Compare their extraction quality with ours
"""
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set API key
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')

def test_scrapegraphai():
    """Test ScrapeGraphAI on our failing sources"""
    from scrapegraphai.graphs import SmartScraperGraph
    
    # Configuration
    graph_config = {
        "llm": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.environ['OPENAI_API_KEY'],
        },
        "verbose": True,
        "headless": True,
    }
    
    test_sources = [
        {
            "url": "https://www.amazon.com/s?k=laptop",
            "prompt": "Extract all laptop product listings with product title, price, and rating",
            "name": "Amazon",
            "our_failure": "Extracted marketing copy instead of products, 100% empty prices"
        },
        {
            "url": "https://news.ycombinator.com/",
            "prompt": "Extract all article listings with title, points, and comments count",
            "name": "Hacker News",
            "our_failure": "97% empty titles, sequential numbers (1,2,3) for points"
        },
        {
            "url": "https://old.reddit.com/r/programming/",
            "prompt": "Extract all post listings with post title, author username, and upvotes",
            "name": "Reddit",
            "our_failure": "Titles OK, but authors were timestamps/counts instead of usernames"
        },
    ]
    
    for source in test_sources:
        print("\n" + "="*100)
        print(f"🧪 TESTING: {source['name']}")
        print("="*100)
        print(f"URL: {source['url']}")
        print(f"Prompt: {source['prompt']}")
        print(f"\nOur Failure: {source['our_failure']}")
        print()
        
        try:
            # Create the SmartScraperGraph instance
            smart_scraper_graph = SmartScraperGraph(
                prompt=source['prompt'],
                source=source['url'],
                config=graph_config
            )
            
            # Run the graph
            print("⏳ Running ScrapeGraphAI extraction...")
            result = smart_scraper_graph.run()
            
            print()
            print("="*100)
            print(f"📊 SCRAPEGRAPHAI RESULTS - {source['name']}")
            print("="*100)
            
            if isinstance(result, dict):
                # Pretty print result
                import json
                print(json.dumps(result, indent=2))
                
                # Analyze quality
                if isinstance(result, list):
                    print(f"\n📈 Extracted {len(result)} items")
                    
                    if result:
                        print("\nSample item:")
                        for key, value in list(result[0].items())[:5]:
                            print(f"  • {key}: {str(value)[:80]}")
                elif isinstance(result, dict) and any(isinstance(v, list) for v in result.values()):
                    # Find the list in the result
                    for key, value in result.items():
                        if isinstance(value, list) and len(value) > 0:
                            print(f"\n📈 Extracted {len(value)} items under key '{key}'")
                            if value:
                                print("\nSample item:")
                                for k, v in list(value[0].items())[:5]:
                                    print(f"  • {k}: {str(v)[:80]}")
                            break
            else:
                print(result)
            
            print()
            
        except Exception as e:
            print(f"\n❌ ScrapeGraphAI FAILED: {e}")
            import traceback
            traceback.print_exc()
        
        print()


def analyze_scrapegraphai_code():
    """Analyze ScrapeGraphAI's source code to understand their approach"""
    print("\n" + "="*100)
    print("🔍 ANALYZING SCRAPEGRAPHAI ARCHITECTURE")
    print("="*100)
    
    try:
        import scrapegraphai
        from scrapegraphai.graphs import SmartScraperGraph
        import inspect
        
        print(f"\nScrapeGraphAI version: {scrapegraphai.__version__ if hasattr(scrapegraphai, '__version__') else 'unknown'}")
        print(f"Installation path: {scrapegraphai.__file__}")
        print()
        
        # Inspect SmartScraperGraph
        print("📋 SmartScraperGraph Methods:")
        methods = [m for m in dir(SmartScraperGraph) if not m.startswith('_')]
        for method in methods[:10]:
            print(f"  • {method}")
        
        print()
        print("🔍 SmartScraperGraph Signature:")
        sig = inspect.signature(SmartScraperGraph.__init__)
        print(f"  {sig}")
        
        print()
        print("📖 Key Findings:")
        print("  • Graph-based architecture")
        print("  • Direct LLM extraction (no pattern generation)")
        print("  • Configurable LLM backend (OpenAI, Anthropic, etc.)")
        print("  • Handles JS rendering via playwright")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*100)
    print("🔬 SCRAPEGRAPHAI DEEP DIVE - Learning from Successful Universal Scraper")
    print("="*100)
    print("\nThis test will:")
    print("  1. Run ScrapeGraphAI on our failing sources")
    print("  2. Compare their extraction quality vs ours")
    print("  3. Analyze their code to understand their approach")
    print()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Please export it first.")
        sys.exit(1)
    
    print("Starting tests...\n")
    
    # First, analyze their architecture
    analyze_scrapegraphai_code()
    
    print()
    print("Starting tests on real sources...\n")
    
    # Then test on real sources
    test_scrapegraphai()
    
    print("\n" + "="*100)
    print("🏁 DEEP DIVE COMPLETE")
    print("="*100)
    print("\nKey Learnings:")
    print("  • Observe how they handle extraction")
    print("  • Compare data quality with our results")
    print("  • Understand their architecture choices")
    print()

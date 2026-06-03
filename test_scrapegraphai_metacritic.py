"""
Test ScrapeGraphAI on Metacritic (non-async version)
"""
import os
from scrapegraphai.graphs import SmartScraperGraph

def test_scrapegraphai_metacritic():
    """Test with ScrapeGraphAI (synchronous)"""
    print("\n" + "="*80)
    print("🔍 TESTING SCRAPEGRAPHAI ON METACRITIC")
    print("="*80)
    
    url = "https://www.metacritic.com/browse/game/all/all/current-year/"
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return None
    
    graph_config = {
        "llm": {
            "api_key": api_key,
            "model": "openai/gpt-4o-mini"
        },
        "verbose": False,
        "headless": True
    }
    
    # Create the SmartScraperGraph instance
    smart_scraper_graph = SmartScraperGraph(
        prompt="Extract all video games with their name, description, and Metascore rating",
        source=url,
        config=graph_config
    )
    
    print(f"\n📥 Running ScrapeGraphAI on: {url}")
    print(f"   This may take 30-60 seconds...")
    
    result = smart_scraper_graph.run()
    
    print(f"\n📊 RESULTS:")
    
    # Parse result
    items = None
    if isinstance(result, dict):
        # Try to find the items in various possible keys
        for key in ['games', 'items', 'results', 'data', 'video_games']:
            if key in result and isinstance(result[key], list):
                items = result[key]
                break
        
        if not items:
            # Maybe result itself contains the data
            if any(isinstance(v, list) for v in result.values()):
                items = [v for v in result.values() if isinstance(v, list)][0]
            else:
                # Single item or nested structure
                print(f"   Result structure: {list(result.keys())}")
                items = [result] if result else []
    elif isinstance(result, list):
        items = result
    else:
        items = []
    
    print(f"   Items extracted: {len(items) if items else 0}")
    
    if items and len(items) > 0:
        print(f"\n📝 First 5 items:")
        for i, item in enumerate(items[:5], 1):
            if isinstance(item, dict):
                name = item.get('name') or item.get('title') or item.get('game') or 'N/A'
                score = item.get('score') or item.get('rating') or item.get('metascore') or 'N/A'
                desc = item.get('description') or 'N/A'
                if desc and len(str(desc)) > 80:
                    desc = str(desc)[:80] + "..."
                
                print(f"\n   {i}. {name}")
                print(f"      Score: {score}")
                print(f"      Description: {desc}")
            else:
                print(f"\n   {i}. {item}")
    else:
        print("   ⚠️  No items extracted!")
        print(f"   Raw result type: {type(result)}")
        print(f"   Raw result: {str(result)[:500]}...")
    
    return items


if __name__ == "__main__":
    test_scrapegraphai_metacritic()




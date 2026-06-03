#!/usr/bin/env python3
"""
Run ScrapeGraphAI with verbose mode to see their prompt
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

from scrapegraphai.graphs import SmartScraperGraph

def test_verbose():
    """Test ScrapeGraphAI with verbose to see their prompts"""
    
    # Configuration with verbose
    graph_config = {
        "llm": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.environ['OPENAI_API_KEY'],
        },
        "verbose": True,  # Enable verbose to see prompts
        "headless": True,
    }
    
    url = "https://news.ycombinator.com/"
    
    print("\n" + "="*100)
    print("🔍 SCRAPEGRAPHAI VERBOSE MODE - See Their Prompts")
    print("="*100)
    print()
    
    # Create and run
    smart_scraper = SmartScraperGraph(
        prompt="Extract all article listings with title, points, and comments count",
        source=url,
        config=graph_config
    )
    
    result = smart_scraper.run()
    
    print()
    print("="*100)
    print("📊 RESULT")
    print("="*100)
    
    # Extract items
    items = []
    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, list):
                items = value
                break
    
    print(f"Extracted {len(items)} items")
    print()


if __name__ == "__main__":
    test_verbose()




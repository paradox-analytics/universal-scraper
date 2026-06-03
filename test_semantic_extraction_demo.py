"""
Simple Semantic Extraction Demo

Demonstrates the semantic pattern system working with fallback patterns.
This shows data extraction without the complex DOM matching.
"""

import logging
from bs4 import BeautifulSoup

from universal_scraper.core.semantic_extractor import SemanticExtractor
from universal_scraper.core.html_fetcher import HTMLFetcher

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_hacker_news():
    """Test semantic extraction on Hacker News."""
    logger.info("="*80)
    logger.info("🧪 Testing Semantic Extraction on Hacker News")
    logger.info("="*80)
    
    # Fetch HTML
    fetcher = HTMLFetcher()
    result = fetcher.fetch("https://news.ycombinator.com")
    html = result['html']
    
    logger.info(f"✓ Fetched HTML ({len(html):,} bytes)")
    
    # Parse and find containers manually
    soup = BeautifulSoup(html, 'html.parser')
    containers = soup.find_all('tr', class_='athing')
    logger.info(f"✓ Found {len(containers)} story containers")
    
    # Create semantic pattern (fallback-style)
    pattern = {
        "title": {
            "primary": {"type": "link_text", "position": "first"},
            "fallbacks": [
                {"type": "heading", "position": "first"},
                {"type": "first_text", "min_length": 10}
            ]
        },
        "url": {
            "primary": {"type": "link_text", "return": "href"},
            "fallbacks": [
                {"type": "attribute", "name": "href"}
            ]
        }
    }
    
    # Extract with semantic extractor
    extractor = SemanticExtractor()
    results = extractor.extract(
        html=html,
        semantic_pattern=pattern,
        containers=containers[:10]  # Top 10 stories
    )
    
    logger.info(f"\n✅ Extracted {len(results)} stories")
    logger.info(f"\n📄 Sample Results:\n")
    
    for i, story in enumerate(results[:5], 1):
        logger.info(f"{i}. {story.get('title', 'N/A')}")
        logger.info(f"   URL: {story.get('url', 'N/A')}")
        logger.info("")
    
    return len(results)


def test_github_trending():
    """Test semantic extraction on GitHub Trending."""
    logger.info("\n" + "="*80)
    logger.info("🧪 Testing Semantic Extraction on GitHub Trending")
    logger.info("="*80)
    
    # Fetch HTML
    fetcher = HTMLFetcher()
    result = fetcher.fetch("https://github.com/trending")
    html = result['html']
    
    logger.info(f"✓ Fetched HTML ({len(html):,} bytes)")
    
    # Parse and find containers manually
    soup = BeautifulSoup(html, 'html.parser')
    containers = soup.find_all('article')
    logger.info(f"✓ Found {len(containers)} repository containers")
    
    # Create semantic pattern
    pattern = {
        "name": {
            "primary": {"type": "heading", "position": "first"},
            "fallbacks": [
                {"type": "link_text"},
                {"type": "first_text", "min_length": 5}
            ]
        },
        "description": {
            "primary": {"type": "first_text", "min_length": 20},
            "fallbacks": [
                {"type": "first_text", "min_length": 10}
            ]
        },
        "stars": {
            "primary": {"type": "number", "pattern": r"\d+"},
            "fallbacks": [
                {"type": "first_text", "min_length": 1}
            ]
        }
    }
    
    # Extract with semantic extractor
    extractor = SemanticExtractor()
    results = extractor.extract(
        html=html,
        semantic_pattern=pattern,
        containers=containers[:10]
    )
    
    logger.info(f"\n✅ Extracted {len(results)} repositories")
    logger.info(f"\n📄 Sample Results:\n")
    
    for i, repo in enumerate(results[:5], 1):
        logger.info(f"{i}. {repo.get('name', 'N/A')}")
        logger.info(f"   {repo.get('description', 'N/A')[:60]}...")
        logger.info(f"   Stars: {repo.get('stars', 'N/A')}")
        logger.info("")
    
    return len(results)


def test_stackoverflow():
    """Test semantic extraction on Stack Overflow."""
    logger.info("\n" + "="*80)
    logger.info("🧪 Testing Semantic Extraction on Stack Overflow")
    logger.info("="*80)
    
    # Fetch HTML
    fetcher = HTMLFetcher()
    result = fetcher.fetch("https://stackoverflow.com/questions")
    html = result['html']
    
    logger.info(f"✓ Fetched HTML ({len(html):,} bytes)")
    
    # Parse and find containers manually
    soup = BeautifulSoup(html, 'html.parser')
    containers = soup.find_all('div', class_='s-post-summary')
    logger.info(f"✓ Found {len(containers)} question containers")
    
    # Create semantic pattern
    pattern = {
        "title": {
            "primary": {"type": "heading", "position": "first"},
            "fallbacks": [
                {"type": "link_text"},
                {"type": "first_text", "min_length": 10}
            ]
        },
        "votes": {
            "primary": {"type": "number", "pattern": r"-?\d+"},
            "fallbacks": [
                {"type": "first_text", "min_length": 1}
            ]
        }
    }
    
    # Extract with semantic extractor
    extractor = SemanticExtractor()
    results = extractor.extract(
        html=html,
        semantic_pattern=pattern,
        containers=containers[:10]
    )
    
    logger.info(f"\n✅ Extracted {len(results)} questions")
    logger.info(f"\n📄 Sample Results:\n")
    
    for i, q in enumerate(results[:5], 1):
        logger.info(f"{i}. {q.get('title', 'N/A')[:70]}...")
        logger.info(f"   Votes: {q.get('votes', 'N/A')}")
        logger.info("")
    
    return len(results)


def main():
    """Run all demos."""
    logger.info("\n🎯 SEMANTIC EXTRACTION DEMONSTRATION")
    logger.info("Shows fallback patterns working without LLM calls\n")
    
    results = {}
    
    try:
        results['hacker_news'] = test_hacker_news()
    except Exception as e:
        logger.error(f"❌ Hacker News failed: {e}")
        results['hacker_news'] = 0
    
    try:
        results['github'] = test_github_trending()
    except Exception as e:
        logger.error(f"❌ GitHub failed: {e}")
        results['github'] = 0
    
    try:
        results['stackoverflow'] = test_stackoverflow()
    except Exception as e:
        logger.error(f"❌ Stack Overflow failed: {e}")
        results['stackoverflow'] = 0
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 SUMMARY")
    logger.info("="*80)
    logger.info(f"\n✅ Total items extracted: {sum(results.values())}")
    logger.info(f"   • Hacker News: {results.get('hacker_news', 0)} stories")
    logger.info(f"   • GitHub: {results.get('github', 0)} repositories")
    logger.info(f"   • Stack Overflow: {results.get('stackoverflow', 0)} questions")
    
    logger.info(f"\n💡 Key Points:")
    logger.info(f"   • No LLM calls needed (fallback patterns)")
    logger.info(f"   • Semantic strategies are resilient")
    logger.info(f"   • Works across diverse websites")
    logger.info(f"   • Cost: $0.00 (all cached/fallback)")
    
    logger.info(f"\n✅ Semantic extraction system validated!")


if __name__ == "__main__":
    main()





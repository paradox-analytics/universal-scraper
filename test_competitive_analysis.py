"""
Competitive Analysis: Universal Scraper vs ScrapeGraphAI
Tests across multiple use cases and site types
"""

import asyncio
import json
import time
import os
from typing import Dict, List, Any
from datetime import datetime

# Our scraper
from universal_scraper.core.scraper import UniversalScraper

# ScrapeGraphAI
try:
    from scrapegraphai.graphs import SmartScraperGraph
    SCRAPEGRAPH_AVAILABLE = True
except ImportError:
    SCRAPEGRAPH_AVAILABLE = False
    print("⚠️  ScrapeGraphAI not installed. Install with: pip install scrapegraphai")


class CompetitiveTest:
    """Compare Universal Scraper vs ScrapeGraphAI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.results = []
        
        # Initialize our scraper
        self.our_scraper = UniversalScraper(
            api_key=api_key,
            model_name="gpt-4o-mini",
            fetch_mode="hybrid",
            enable_cache=True
        )
    
    async def test_site(
        self,
        url: str,
        category: str,
        fields: List[str],
        extraction_goal: str,
        site_name: str
    ) -> Dict[str, Any]:
        """Test a single site with both scrapers"""
        
        print(f"\n{'='*80}")
        print(f"🧪 Testing: {site_name}")
        print(f"📁 Category: {category}")
        print(f"🔗 URL: {url}")
        print(f"📋 Fields: {', '.join(fields)}")
        print(f"{'='*80}\n")
        
        result = {
            'site_name': site_name,
            'category': category,
            'url': url,
            'fields': fields,
            'extraction_goal': extraction_goal,
            'timestamp': datetime.now().isoformat(),
            'our_scraper': {},
            'scrapegraph': {},
            'comparison': {}
        }
        
        # Test 1: Our Scraper
        print("🔵 Testing Universal Scraper...")
        our_result = await self.test_our_scraper(url, fields, extraction_goal)
        result['our_scraper'] = our_result
        
        # Test 2: ScrapeGraphAI
        if SCRAPEGRAPH_AVAILABLE:
            print("\n🟢 Testing ScrapeGraphAI...")
            scrapegraph_result = await self.test_scrapegraph(url, extraction_goal)
            result['scrapegraph'] = scrapegraph_result
            
            # Compare
            print("\n📊 Comparing results...")
            comparison = self.compare_results(our_result, scrapegraph_result, fields)
            result['comparison'] = comparison
        else:
            result['scrapegraph'] = {'error': 'ScrapeGraphAI not installed'}
        
        # Print summary
        self.print_result_summary(result)
        
        self.results.append(result)
        return result
    
    async def test_our_scraper(
        self,
        url: str,
        fields: List[str],
        extraction_goal: str
    ) -> Dict[str, Any]:
        """Test with our scraper"""
        
        start_time = time.time()
        
        try:
            result = await self.our_scraper.scrape(
                url=url,
                fields=fields
            )
            
            duration = time.time() - start_time
            
            return {
                'success': True,
                'items': result['data'],
                'item_count': len(result['data']),
                'duration': duration,
                'source': result.get('source', 'unknown'),
                'metadata': result.get('metadata', {}),
                'error': None
            }
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Error: {e}")
            return {
                'success': False,
                'items': [],
                'item_count': 0,
                'duration': duration,
                'source': 'error',
                'metadata': {},
                'error': str(e)
            }
    
    async def test_scrapegraph(
        self,
        url: str,
        extraction_goal: str
    ) -> Dict[str, Any]:
        """Test with ScrapeGraphAI"""
        
        if not SCRAPEGRAPH_AVAILABLE:
            return {'error': 'Not installed'}
        
        start_time = time.time()
        
        try:
            # Configure ScrapeGraphAI
            graph_config = {
                "llm": {
                    "api_key": self.api_key,
                    "model": "openai/gpt-4o-mini",
                },
                "verbose": False,
                "headless": True
            }
            
            # Create scraper
            smart_scraper = SmartScraperGraph(
                prompt=extraction_goal,
                source=url,
                config=graph_config
            )
            
            # Run scraper
            result = smart_scraper.run()
            
            duration = time.time() - start_time
            
            # Normalize result to list of items
            items = []
            if isinstance(result, dict):
                # Try to find list in result
                for key, value in result.items():
                    if isinstance(value, list):
                        items = value
                        break
                if not items:
                    items = [result]
            elif isinstance(result, list):
                items = result
            else:
                items = []
            
            return {
                'success': True,
                'items': items,
                'item_count': len(items),
                'duration': duration,
                'raw_result': result,
                'error': None
            }
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Error: {e}")
            return {
                'success': False,
                'items': [],
                'item_count': 0,
                'duration': duration,
                'raw_result': None,
                'error': str(e)
            }
    
    def compare_results(
        self,
        our_result: Dict,
        scrapegraph_result: Dict,
        expected_fields: List[str]
    ) -> Dict[str, Any]:
        """Compare two scraper results"""
        
        comparison = {
            'our_items': our_result['item_count'],
            'scrapegraph_items': scrapegraph_result['item_count'],
            'item_delta': our_result['item_count'] - scrapegraph_result['item_count'],
            'item_delta_pct': 0.0,
            'our_duration': our_result['duration'],
            'scrapegraph_duration': scrapegraph_result['duration'],
            'speed_delta': scrapegraph_result['duration'] - our_result['duration'],
            'speed_ratio': 0.0,
            'our_completeness': 0.0,
            'scrapegraph_completeness': 0.0,
            'completeness_delta': 0.0,
            'winner_items': '',
            'winner_speed': '',
            'winner_quality': ''
        }
        
        # Item count comparison
        if scrapegraph_result['item_count'] > 0:
            comparison['item_delta_pct'] = (
                (our_result['item_count'] - scrapegraph_result['item_count']) 
                / scrapegraph_result['item_count'] * 100
            )
        
        # Speed comparison
        if our_result['duration'] > 0:
            comparison['speed_ratio'] = (
                scrapegraph_result['duration'] / our_result['duration']
            )
        
        # Completeness (field coverage)
        our_completeness = self.calculate_completeness(
            our_result['items'], expected_fields
        )
        scrapegraph_completeness = self.calculate_completeness(
            scrapegraph_result['items'], expected_fields
        )
        
        comparison['our_completeness'] = our_completeness
        comparison['scrapegraph_completeness'] = scrapegraph_completeness
        comparison['completeness_delta'] = our_completeness - scrapegraph_completeness
        
        # Determine winners
        comparison['winner_items'] = (
            'ours' if our_result['item_count'] > scrapegraph_result['item_count']
            else 'scrapegraph' if scrapegraph_result['item_count'] > our_result['item_count']
            else 'tie'
        )
        
        comparison['winner_speed'] = (
            'ours' if our_result['duration'] < scrapegraph_result['duration']
            else 'scrapegraph' if scrapegraph_result['duration'] < our_result['duration']
            else 'tie'
        )
        
        comparison['winner_quality'] = (
            'ours' if our_completeness > scrapegraph_completeness
            else 'scrapegraph' if scrapegraph_completeness > our_completeness
            else 'tie'
        )
        
        return comparison
    
    def calculate_completeness(
        self,
        items: List[Dict],
        expected_fields: List[str]
    ) -> float:
        """Calculate field completeness percentage"""
        
        if not items or not expected_fields:
            return 0.0
        
        total_fields = len(items) * len(expected_fields)
        filled_fields = 0
        
        for item in items:
            for field in expected_fields:
                # Check if field exists and has non-empty value
                value = item.get(field)
                if value is not None and value != '' and value != []:
                    filled_fields += 1
        
        return (filled_fields / total_fields * 100) if total_fields > 0 else 0.0
    
    def print_result_summary(self, result: Dict):
        """Print summary for a single test"""
        
        our = result['our_scraper']
        scrapegraph = result['scrapegraph']
        comp = result.get('comparison', {})
        
        print(f"\n{'─'*80}")
        print(f"📊 Results Summary: {result['site_name']}")
        print(f"{'─'*80}")
        
        print(f"\n🔵 Universal Scraper:")
        print(f"   Items: {our['item_count']}")
        print(f"   Time: {our['duration']:.2f}s")
        print(f"   Source: {our.get('source', 'unknown')}")
        if comp:
            print(f"   Completeness: {comp['our_completeness']:.1f}%")
        
        if scrapegraph.get('success'):
            print(f"\n🟢 ScrapeGraphAI:")
            print(f"   Items: {scrapegraph['item_count']}")
            print(f"   Time: {scrapegraph['duration']:.2f}s")
            if comp:
                print(f"   Completeness: {comp['scrapegraph_completeness']:.1f}%")
        
        if comp:
            print(f"\n🏆 Comparison:")
            print(f"   Items: {comp['item_delta']:+d} ({comp['item_delta_pct']:+.1f}%) - Winner: {comp['winner_items']}")
            print(f"   Speed: {comp['speed_delta']:+.2f}s ({comp['speed_ratio']:.2f}x) - Winner: {comp['winner_speed']}")
            print(f"   Quality: {comp['completeness_delta']:+.1f}% - Winner: {comp['winner_quality']}")
        
        print(f"\n{'─'*80}\n")
    
    def print_final_summary(self):
        """Print final summary across all tests"""
        
        print(f"\n{'='*80}")
        print(f"📊 FINAL SUMMARY - Universal Scraper vs ScrapeGraphAI")
        print(f"{'='*80}\n")
        
        # Aggregate statistics
        total_tests = len(self.results)
        our_wins_items = 0
        our_wins_speed = 0
        our_wins_quality = 0
        
        total_our_items = 0
        total_scrapegraph_items = 0
        total_our_time = 0
        total_scrapegraph_time = 0
        our_completeness_avg = 0
        scrapegraph_completeness_avg = 0
        
        for result in self.results:
            comp = result.get('comparison', {})
            if not comp:
                continue
            
            if comp['winner_items'] == 'ours':
                our_wins_items += 1
            if comp['winner_speed'] == 'ours':
                our_wins_speed += 1
            if comp['winner_quality'] == 'ours':
                our_wins_quality += 1
            
            total_our_items += result['our_scraper']['item_count']
            total_scrapegraph_items += result['scrapegraph']['item_count']
            total_our_time += result['our_scraper']['duration']
            total_scrapegraph_time += result['scrapegraph']['duration']
            our_completeness_avg += comp['our_completeness']
            scrapegraph_completeness_avg += comp['scrapegraph_completeness']
        
        if total_tests > 0:
            our_completeness_avg /= total_tests
            scrapegraph_completeness_avg /= total_tests
        
        # Print by category
        print("📋 Results by Category:\n")
        
        categories = {}
        for result in self.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        for category, tests in categories.items():
            print(f"  {category}:")
            for test in tests:
                comp = test.get('comparison', {})
                if comp:
                    print(f"    • {test['site_name']}: "
                          f"Items {test['our_scraper']['item_count']} vs {test['scrapegraph']['item_count']} "
                          f"({comp['winner_items']}), "
                          f"Speed {test['our_scraper']['duration']:.1f}s vs {test['scrapegraph']['duration']:.1f}s "
                          f"({comp['winner_speed']})")
            print()
        
        # Overall statistics
        print("🏆 Overall Statistics:\n")
        print(f"  Total Tests: {total_tests}")
        print(f"  Universal Scraper Wins:")
        print(f"    • Items: {our_wins_items}/{total_tests} ({our_wins_items/total_tests*100:.0f}%)")
        print(f"    • Speed: {our_wins_speed}/{total_tests} ({our_wins_speed/total_tests*100:.0f}%)")
        print(f"    • Quality: {our_wins_quality}/{total_tests} ({our_wins_quality/total_tests*100:.0f}%)")
        print()
        print(f"  Total Items Extracted:")
        print(f"    • Universal Scraper: {total_our_items}")
        print(f"    • ScrapeGraphAI: {total_scrapegraph_items}")
        print(f"    • Delta: {total_our_items - total_scrapegraph_items:+d} ({(total_our_items/total_scrapegraph_items-1)*100:+.1f}%)")
        print()
        print(f"  Total Execution Time:")
        print(f"    • Universal Scraper: {total_our_time:.1f}s")
        print(f"    • ScrapeGraphAI: {total_scrapegraph_time:.1f}s")
        print(f"    • Delta: {total_our_time - total_scrapegraph_time:+.1f}s ({total_our_time/total_scrapegraph_time:.2f}x)")
        print()
        print(f"  Average Completeness:")
        print(f"    • Universal Scraper: {our_completeness_avg:.1f}%")
        print(f"    • ScrapeGraphAI: {scrapegraph_completeness_avg:.1f}%")
        print(f"    • Delta: {our_completeness_avg - scrapegraph_completeness_avg:+.1f}%")
        
        print(f"\n{'='*80}\n")
    
    def save_results(self, filename: str = "competitive_analysis_results.json"):
        """Save results to JSON file"""
        
        with open(filename, 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'total_tests': len(self.results),
                'results': self.results
            }, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
    
    async def close(self):
        """Clean up resources"""
        await self.our_scraper.close()


# Test suite definition
TEST_SITES = [
    # E-commerce (3 sites)
    {
        'site_name': 'Books to Scrape',
        'category': 'E-commerce',
        'url': 'https://books.toscrape.com/',
        'fields': ['title', 'price', 'rating', 'availability'],
        'extraction_goal': 'Extract all book listings with title, price, rating, and availability'
    },
    {
        'site_name': 'Quotes to Scrape',
        'category': 'E-commerce',
        'url': 'https://quotes.toscrape.com/',
        'fields': ['text', 'author', 'tags'],
        'extraction_goal': 'Extract all quotes with text, author, and tags'
    },
    {
        'site_name': 'ScrapingBee E-commerce Demo',
        'category': 'E-commerce',
        'url': 'https://www.scrapingbee.com/blog/web-scraping-javascript/',
        'fields': ['title', 'date', 'author'],
        'extraction_goal': 'Extract blog post title, date, and author'
    },
    
    # News/Content (3 sites)
    {
        'site_name': 'Hacker News',
        'category': 'News/Content',
        'url': 'https://news.ycombinator.com/',
        'fields': ['title', 'points', 'comments', 'author'],
        'extraction_goal': 'Extract all article listings with title, points, comments, and author'
    },
    {
        'site_name': 'BBC News Tech',
        'category': 'News/Content',
        'url': 'https://www.bbc.com/news/technology',
        'fields': ['headline', 'summary', 'timestamp'],
        'extraction_goal': 'Extract news articles with headline, summary, and timestamp'
    },
    {
        'site_name': 'Product Hunt',
        'category': 'News/Content',
        'url': 'https://www.producthunt.com/',
        'fields': ['name', 'tagline', 'votes', 'comments'],
        'extraction_goal': 'Extract product listings with name, tagline, votes, and comments'
    },
    
    # Directory/Listings (3 sites)
    {
        'site_name': 'GitHub Trending',
        'category': 'Directory',
        'url': 'https://github.com/trending',
        'fields': ['repository', 'description', 'stars', 'language'],
        'extraction_goal': 'Extract trending repositories with name, description, stars, and language'
    },
    {
        'site_name': 'PyPI Trending',
        'category': 'Directory',
        'url': 'https://pypi.org/search/?q=&o=-zscore',
        'fields': ['package_name', 'description', 'version'],
        'extraction_goal': 'Extract Python packages with name, description, and version'
    },
    {
        'site_name': 'Stack Overflow Questions',
        'category': 'Directory',
        'url': 'https://stackoverflow.com/questions',
        'fields': ['title', 'votes', 'answers', 'views', 'tags'],
        'extraction_goal': 'Extract questions with title, votes, answers, views, and tags'
    },
    
    # Other (3 sites)
    {
        'site_name': 'Wikipedia Python',
        'category': 'Reference',
        'url': 'https://en.wikipedia.org/wiki/Python_(programming_language)',
        'fields': ['title', 'summary', 'infobox_data'],
        'extraction_goal': 'Extract article title, summary paragraph, and infobox data'
    },
    {
        'site_name': 'IMDb Top Movies',
        'category': 'Entertainment',
        'url': 'https://www.imdb.com/chart/top/',
        'fields': ['title', 'year', 'rating', 'votes'],
        'extraction_goal': 'Extract top movies with title, year, rating, and number of votes'
    },
    {
        'site_name': 'Reddit Python',
        'category': 'Social',
        'url': 'https://www.reddit.com/r/Python/',
        'fields': ['title', 'author', 'upvotes', 'comments'],
        'extraction_goal': 'Extract posts with title, author, upvotes, and comment count'
    }
]


async def main():
    """Run competitive analysis"""
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    if not SCRAPEGRAPH_AVAILABLE:
        print("⚠️  Warning: ScrapeGraphAI not installed")
        print("   Only testing Universal Scraper")
        print("   To install: pip install scrapegraphai")
        print()
    
    tester = CompetitiveTest(api_key)
    
    print(f"\n{'='*80}")
    print(f"🚀 Competitive Analysis: Universal Scraper vs ScrapeGraphAI")
    print(f"{'='*80}")
    print(f"Testing {len(TEST_SITES)} sites across 4 categories")
    print(f"{'='*80}\n")
    
    try:
        # Run all tests
        for site in TEST_SITES:
            await tester.test_site(**site)
            # Brief pause between tests
            await asyncio.sleep(2)
        
        # Print final summary
        tester.print_final_summary()
        
        # Save results
        tester.save_results()
    
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())



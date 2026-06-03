#!/usr/bin/env python3
"""
Update Home Depot strategy in unified cache

Migrates the simple .scraping_strategies.json to the new unified format
using ScrapingStrategyDetector
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.scraping_strategy_detector import ScrapingStrategyDetector

# Initialize detector
detector = ScrapingStrategyDetector()

# Record Home Depot JSON-LD strategy
print("📊 Recording Home Depot strategy...")

detector.record_strategy(
    url="https://www.homedepot.com/p/product/123",
    extraction_method="json_ld",
    proxy_type="web_unblocker",
    browser_config={
        'humanize': True,
        'geoip': False,
        'stealth': False,
        'timeout': 120000,
        'blockImages': True,
        'generateCanvas': True,
        'rotateProfile': False,
        'proxyRotationInterval': 'per_domain'
    },
    success=True,
    html_quality="EXCELLENT",
    extraction_details={
        'script_id': 'thd-helmet__script--productStructureData',
        'script_type': 'application/ld+json',
        'fields_available': ['name', 'sku', 'brand', 'price', 'rating', 'reviews', 'availability'],
        'reliability': 'HIGH',
        'speed': 'FAST'
    },
    performance_metrics={
        'elapsed_time': 69.7,
        'html_size': 1156726,
        'retry_count': 1
    }
)

print("\n✅ Strategy recorded successfully!")

# Display summary
summary = detector.export_summary()
print(f"\n📈 Strategy Summary:")
print(f"   Total domains: {summary['total_domains']}")

for domain, info in summary['domains'].items():
    print(f"\n   {domain}:")
    print(f"      Attempts: {info['total_attempts']}")
    print(f"      Success Rate: {info['success_rate']:.1%}")
    print(f"      Method: {info['recommended_method']}")
    print(f"      Proxy: {info['recommended_proxy']}")

# Test retrieval
print(f"\n🔍 Testing strategy retrieval...")
strategy = detector.get_strategy("https://www.homedepot.com/p/test/456")

if strategy:
    print(f"   ✅ Retrieved strategy:")
    print(f"      Extraction: {strategy['extraction_method']}")
    print(f"      Proxy: {strategy['proxy_type']}")
    print(f"      Confidence: {strategy['confidence']:.2f}")
    print(f"      Success Rate: {strategy['success_rate']:.1%}")

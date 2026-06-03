#!/usr/bin/env python3
"""
Test Adaptive Anti-Blocking System

Quick test to verify core components work correctly.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.browser_config_generator import BrowserConfigGenerator, ConfigPreset
from universal_scraper.core.blocking_detector import BlockingDetector, BlockingType
from universal_scraper.core.config_learner import ConfigurationLearner
from universal_scraper.core.adaptive_antiblocking_agent import AdaptiveAntiBlockingAgent

def test_config_generator():
    """Test configuration generator"""
    print("\n" + "=" * 80)
    print("🧪 Testing BrowserConfigGenerator")
    print("=" * 80)
    
    generator = BrowserConfigGenerator()
    
    # Test each preset
    for preset in [ConfigPreset.STEALTH, ConfigPreset.BALANCED, ConfigPreset.AGGRESSIVE]:
        config = generator.generate(preset=preset)
        print(f"\n{preset.value.upper()} Preset:")
        print(f"  Block Images: {config['blockImages']}")
        print(f"  Generate Canvas: {config['generateCanvasString']}")
        print(f"  Rotate Profile: {config['rotateProfile']}")
        print(f"  Viewport: {config['dynamicBrowserWidth']}x{config['dynamicBrowserHeight']}")
    
    # Test variations
    variations = generator.generate_variations(num_variations=3)
    print(f"\n✅ Generated {len(variations)} variations")
    
    # Test Camoufox conversion
    camoufox_config = generator.to_camoufox_config(config)
    print(f"\n✅ Camoufox config: {camoufox_config.keys()}")

def test_blocking_detector():
    """Test blocking detector"""
    print("\n" + "=" * 80)
    print("🧪 Testing BlockingDetector")
    print("=" * 80)
    
    detector = BlockingDetector()
    
    # Test Cloudflare detection
    cloudflare_html = "<html><body>Checking your browser before accessing...</body></html>"
    result = detector.detect(html=cloudflare_html, status_code=503, headers={'cf-ray': '123'})
    print(f"\nCloudflare Detection:")
    print(f"  Type: {result['type_name']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Is Blocked: {result['is_blocked']}")
    
    # Test 403
    result = detector.detect(status_code=403)
    print(f"\n403 Detection:")
    print(f"  Type: {result['type_name']}")
    print(f"  Confidence: {result['confidence']}")
    
    # Test timeout
    result = detector.detect(error_message="NS_ERROR_NET_TIMEOUT")
    print(f"\nTimeout Detection:")
    print(f"  Type: {result['type_name']}")
    print(f"  Confidence: {result['confidence']}")
    
    # Test recommendations
    recommendations = detector.get_bypass_recommendations(BlockingType.CLOUDFLARE)
    print(f"\n✅ Cloudflare Recommendations: {len(recommendations)} suggestions")

def test_config_learner():
    """Test configuration learner"""
    print("\n" + "=" * 80)
    print("🧪 Testing ConfigurationLearner")
    print("=" * 80)
    
    learner = ConfigurationLearner()
    
    # Record some attempts
    test_config = {'blockImages': True, 'generateCanvasString': True}
    
    for i in range(5):
        learner.record_attempt(
            domain='example.com',
            config=test_config,
            success=i % 2 == 0,  # Alternate success/failure
            blocking_type='cloudflare' if i % 2 == 1 else 'none',
            response_time=1.5
        )
    
    # Get stats
    stats = learner.get_domain_stats('example.com')
    print(f"\nDomain Stats:")
    print(f"  Total Attempts: {stats['total_attempts']}")
    print(f"  Success Rate: {stats['success_rate']:.2%}")
    print(f"  Configs Tried: {stats['configs_tried']}")
    
    # Get best config
    best_config = learner.get_best_config('example.com')
    print(f"\n✅ Best config found: {best_config is not None}")
    
    # Get recommendations
    recommendations = learner.get_recommendations('example.com')
    print(f"✅ Recommendations: {len(recommendations)}")

async def test_adaptive_agent():
    """Test adaptive anti-blocking agent"""
    print("\n" + "=" * 80)
    print("🧪 Testing AdaptiveAntiBlockingAgent")
    print("=" * 80)
    
    agent = AdaptiveAntiBlockingAgent(max_parallel_tests=2)
    
    # Mock fetcher function
    async def mock_fetcher(url, config):
        await asyncio.sleep(0.1)  # Simulate fetch
        return {
            'html': '<html><body>Success!</body></html>',
            'status_code': 200,
            'headers': {}
        }
    
    # Test fetch with adaptation
    result = await agent.fetch_with_adaptation(
        url='https://example.com/test',
        fetcher_func=mock_fetcher,
        preset=ConfigPreset.BALANCED,
        max_attempts=3
    )
    
    print(f"\nFetch Result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Status: {result.get('status_code')}")
    print(f"  HTML Size: {len(result.get('html', ''))}")
    
    # Get domain insights
    insights = agent.get_domain_insights('example.com')
    print(f"\nDomain Insights:")
    print(f"  Total Attempts: {insights['stats']['total_attempts']}")
    print(f"  Success Rate: {insights['stats']['success_rate']:.2%}")
    print(f"  Has Learned Config: {insights['has_learned_config']}")
    
    print(f"\n✅ Adaptive agent test complete!")

async def main():
    """Run all tests"""
    print("=" * 80)
    print("🚀 ADAPTIVE ANTI-BLOCKING SYSTEM - COMPONENT TESTS")
    print("=" * 80)
    
    try:
        test_config_generator()
        test_blocking_detector()
        test_config_learner()
        await test_adaptive_agent()
        
        print("\n" + "=" * 80)
        print("✨ ALL TESTS PASSED!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

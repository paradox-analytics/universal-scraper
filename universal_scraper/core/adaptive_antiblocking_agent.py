"""
Adaptive Anti-Blocking Agent

AI-driven agent that learns and adapts browser configurations to bypass blocking.
Orchestrates configuration testing, learning, and LLM-based optimization.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

from universal_scraper.core.browser_config_generator import BrowserConfigGenerator, ConfigPreset
from universal_scraper.core.blocking_detector import BlockingDetector, BlockingType
from universal_scraper.core.config_learner import ConfigurationLearner

logger = logging.getLogger(__name__)


class AdaptiveAntiBlockingAgent:
    """
    AI-driven agent for adaptive anti-blocking.
    
    Workflow:
    1. Check if we have a learned config for the domain
    2. If yes, try it first
    3. If no or if it fails, generate and test multiple configs in parallel
    4. Learn from results and cache successful configs
    5. Use LLM to suggest optimizations (future enhancement)
    """
    
    def __init__(
        self,
        redis_cache=None,
        llm_api_key: Optional[str] = None,
        max_parallel_tests: int = 3,
        enable_llm_optimization: bool = False
    ):
        """
        Initialize adaptive anti-blocking agent.
        
        Args:
            redis_cache: Redis cache for persistent learning
            llm_api_key: API key for LLM-based optimization
            max_parallel_tests: Maximum parallel configuration tests
            enable_llm_optimization: Enable LLM-based config optimization
        """
        self.config_generator = BrowserConfigGenerator()
        self.blocking_detector = BlockingDetector()
        self.learner = ConfigurationLearner(redis_cache=redis_cache)
        
        self.llm_api_key = llm_api_key
        self.max_parallel_tests = max_parallel_tests
        self.enable_llm_optimization = enable_llm_optimization
        
        logger.info(f"🤖 Adaptive Anti-Blocking Agent initialized")
        logger.info(f"   Max parallel tests: {max_parallel_tests}")
        logger.info(f"   LLM optimization: {enable_llm_optimization}")
    
    async def fetch_with_adaptation(
        self,
        url: str,
        fetcher_func,
        preset: ConfigPreset = ConfigPreset.BALANCED,
        max_attempts: int = 5,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Fetch URL with adaptive anti-blocking.
        
        This method is FULLY AUTONOMOUS - it will:
        1. Analyze historical data for the domain
        2. Determine optimal timeout dynamically
        3. Select best strategy (preset, Web Unblocker, etc.)
        4. Test configurations intelligently
        5. Learn and adapt in real-time
        
        Args:
            url: Target URL
            fetcher_func: Async function to fetch with config
            preset: Starting configuration preset (may be overridden)
            max_attempts: Maximum configuration attempts
            progress_callback: Optional callback for progress updates
            
        Returns:
            Fetch result with HTML, status, and metadata
        """
        domain = urlparse(url).netloc
        
        logger.info(f"🎯 Adaptive fetch: {url}")
        logger.info(f"   Domain: {domain}")
        
        # STEP 1: Analyze domain and determine optimal strategy
        strategy = self.learner.get_optimal_strategy(domain)
        optimal_timeout = self.learner.get_optimal_timeout(domain)
        
        logger.info(f"🧠 Intelligent Strategy Selected:")
        logger.info(f"   Preset: {strategy['preset']}")
        logger.info(f"   Timeout: {optimal_timeout}ms")
        logger.info(f"   Web Unblocker: {strategy['use_web_unblocker']}")
        logger.info(f"   Proxy Rotation: {strategy['proxy_rotation']}")
        logger.info(f"   Reason: {strategy['reason']}")
        
        if progress_callback:
            await progress_callback({
                'stage': 'strategy_selected',
                'domain': domain,
                'strategy': strategy,
                'timeout': optimal_timeout
            })
        
        # Override preset based on learned strategy
        if strategy['preset'] == 'stealth':
            preset = ConfigPreset.STEALTH
        elif strategy['preset'] == 'aggressive':
            preset = ConfigPreset.AGGRESSIVE
        else:
            preset = ConfigPreset.BALANCED
        
        # STEP 2: Try learned configuration if available
        learned_config = self.learner.get_best_config(domain)
        
        if learned_config:
            logger.info(f"📚 Found learned config for {domain}, trying it first")
            
            if progress_callback:
                await progress_callback({
                    'stage': 'trying_learned_config',
                    'domain': domain,
                    'attempt': 1,
                    'max_attempts': max_attempts
                })
            
            result = await self._test_configuration(
                url,
                learned_config,
                fetcher_func,
                domain,
                timeout=optimal_timeout
            )
            
            if result['success']:
                logger.info(f"✅ Learned config worked!")
                result['strategy_used'] = 'learned'
                return result
            else:
                logger.info(f"⚠️  Learned config failed, trying alternatives")
        
        # STEP 3: If Web Unblocker is recommended, try it
        if strategy['use_web_unblocker']:
            logger.info(f"🛡️  Strategy recommends Web Unblocker - trying it")
            
            if progress_callback:
                await progress_callback({
                    'stage': 'trying_web_unblocker',
                    'domain': domain
                })
            
            # Generate Web Unblocker config
            unblocker_config = self.config_generator.generate(
                preset=ConfigPreset.STEALTH,
                overrides={'use_web_unblocker': True}
            )
            
            result = await self._test_configuration(
                url,
                unblocker_config,
                fetcher_func,
                domain,
                timeout=optimal_timeout
            )
            
            if result['success']:
                logger.info(f"✅ Web Unblocker worked!")
                result['strategy_used'] = 'web_unblocker'
                return result
        
        # STEP 4: Generate and test multiple configurations
        logger.info(f"🔬 Testing {self.max_parallel_tests} configurations in parallel")
        
        # Generate variations based on selected preset
        configs = self.config_generator.generate_variations(
            base_preset=preset,
            num_variations=self.max_parallel_tests
        )
        
        # Apply strategy-specific overrides
        for config in configs:
            config['proxyRotationInterval'] = strategy['proxy_rotation']
            if strategy.get('increase_delays'):
                config['humanize_delays'] = True
        
        # Test in parallel
        tasks = []
        for i, config in enumerate(configs):
            if progress_callback:
                await progress_callback({
                    'stage': 'testing_variations',
                    'domain': domain,
                    'attempt': i + 1,
                    'max_attempts': len(configs)
                })
            
            task = self._test_configuration(
                url,
                config,
                fetcher_func,
                domain,
                timeout=optimal_timeout
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Find first successful result
        for result in results:
            if isinstance(result, dict) and result.get('success'):
                logger.info(f"✅ Found working configuration!")
                result['strategy_used'] = f'{preset.value}_variation'
                return result
        
        # STEP 5: If all failed, try preset escalation
        if preset != ConfigPreset.STEALTH:
            logger.info(f"⚠️  All variations failed, escalating to STEALTH preset")
            
            if progress_callback:
                await progress_callback({
                    'stage': 'escalating_to_stealth',
                    'domain': domain
                })
            
            stealth_config = self.config_generator.generate(preset=ConfigPreset.STEALTH)
            result = await self._test_configuration(
                url,
                stealth_config,
                fetcher_func,
                domain,
                timeout=optimal_timeout
            )
            
            if result['success']:
                logger.info(f"✅ STEALTH preset worked!")
                result['strategy_used'] = 'stealth_escalation'
                return result
        
        # STEP 6: Final analysis and recommendations
        logger.warning(f"❌ All configurations failed for {domain}")
        
        # Get updated strategy based on new failures
        updated_strategy = self.learner.get_optimal_strategy(domain)
        recommendations = self.learner.get_recommendations(domain)
        
        return {
            'success': False,
            'html': '',
            'status_code': 0,
            'error': 'All configurations failed',
            'recommendations': recommendations,
            'next_strategy': updated_strategy,
            'blocking_analysis': result.get('blocking_analysis', {}),
            'timeout_used': optimal_timeout
        }
    
    async def _test_configuration(
        self,
        url: str,
        config: Dict[str, Any],
        fetcher_func,
        domain: str,
        timeout: int = 60000
    ) -> Dict[str, Any]:
        """
        Test a single configuration.
        
        Args:
            url: Target URL
            config: Browser configuration
            fetcher_func: Fetcher function
            domain: Domain name
            timeout: Timeout in milliseconds (dynamically determined)
            
        Returns:
            Test result
        """
        import time
        start_time = time.time()
        
        try:
            # Convert to Camoufox format
            camoufox_config = self.config_generator.to_camoufox_config(config)
            
            # Add timeout to config
            camoufox_config['timeout'] = timeout
            
            # Fetch with this config
            result = await fetcher_func(url, camoufox_config)
            
            response_time = time.time() - start_time
            
            # Detect blocking
            blocking_analysis = self.blocking_detector.detect(
                html=result.get('html', ''),
                status_code=result.get('status_code', 0),
                headers=result.get('headers', {}),
                error_message=result.get('error', '')
            )
            
            success = (
                result.get('status_code') == 200 and
                len(result.get('html', '')) > 1000 and
                not blocking_analysis['is_blocked']
            )
            
            # Record attempt
            self.learner.record_attempt(
                domain=domain,
                config=config,
                success=success,
                blocking_type=blocking_analysis['type_name'],
                response_time=response_time,
                details={
                    'status_code': result.get('status_code'),
                    'html_size': len(result.get('html', '')),
                    'blocking_confidence': blocking_analysis['confidence']
                }
            )
            
            # Add metadata to result
            result['success'] = success
            result['blocking_analysis'] = blocking_analysis
            result['config_used'] = config
            result['response_time'] = response_time
            
            return result
            
        except Exception as e:
            logger.error(f"Configuration test failed: {e}")
            
            # Record failure
            self.learner.record_attempt(
                domain=domain,
                config=config,
                success=False,
                blocking_type='error',
                response_time=time.time() - start_time,
                details={'error': str(e)}
            )
            
            return {
                'success': False,
                'html': '',
                'status_code': 0,
                'error': str(e),
                'config_used': config
            }
    
    def get_domain_insights(self, domain: str) -> Dict[str, Any]:
        """
        Get insights about a domain's anti-blocking behavior.
        
        Args:
            domain: Target domain
            
        Returns:
            Insights dict with stats and recommendations
        """
        stats = self.learner.get_domain_stats(domain)
        recommendations = self.learner.get_recommendations(domain)
        best_config = self.learner.get_best_config(domain)
        
        return {
            'domain': domain,
            'stats': stats,
            'recommendations': recommendations,
            'has_learned_config': best_config is not None,
            'best_config_preview': {
                'preset': 'learned',
                'success_rate': stats.get('success_rate', 0.0)
            } if best_config else None
        }
    
    async def optimize_with_llm(
        self,
        domain: str,
        failed_configs: List[Dict[str, Any]],
        blocking_type: BlockingType
    ) -> Dict[str, Any]:
        """
        Use LLM to suggest configuration optimizations.
        
        This is a future enhancement that will use an LLM to analyze
        failed configurations and suggest improvements.
        
        Args:
            domain: Target domain
            failed_configs: List of failed configurations
            blocking_type: Type of blocking detected
            
        Returns:
            Optimized configuration
        """
        if not self.enable_llm_optimization or not self.llm_api_key:
            logger.warning("LLM optimization not enabled")
            return self.config_generator.generate(preset=ConfigPreset.STEALTH)
        
        # TODO: Implement LLM-based optimization
        # This would:
        # 1. Analyze failed configs
        # 2. Identify patterns
        # 3. Use LLM to suggest tweaks
        # 4. Generate optimized config
        
        logger.info("LLM optimization not yet implemented, using STEALTH preset")
        return self.config_generator.generate(preset=ConfigPreset.STEALTH)

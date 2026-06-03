"""
Scraping Strategy Detector

Integrates browser configuration learning with extraction method detection.
Provides unified strategy caching for complete domain-specific scraping strategies.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ScrapingStrategyDetector:
    """
    Detects and caches optimal scraping strategies per domain.

    Integrates:
    - Browser configuration learning (from ConfigurationLearner)
    - Extraction method detection (JSON-LD, GraphQL, HTML)
    - Proxy configuration
    - Anti-blocking tactics
    """

    def __init__(self, cache_file: Optional[Path] = None, config_learner=None, redis_client=None):
        """
        Initialize strategy detector.

        Args:
            cache_file: Path to JSON cache file (fallback if no Redis)
            config_learner: Existing ConfigurationLearner instance
            redis_client: Redis client for distributed caching
        """
        self.cache_file = cache_file or Path('.scraping_strategies.json')
        self.config_learner = config_learner
        self.redis_client = redis_client
        self.strategies: Dict[str, Dict[str, Any]] = {}

        # Redis key prefix
        self.redis_prefix = "scraping_strategy:"

        # Load existing cache
        self._load_cache()

        logger.info("🎯 Scraping Strategy Detector initialized")
        logger.info(f"   Cache file: {self.cache_file}")
        logger.info(f"   Redis: {'✅ Enabled' if redis_client else '❌ Disabled (using file cache)'}")
        logger.info(f"   Cached domains: {len(self.strategies)}")

    def _load_cache(self):
        """Load strategies from Redis or cache file"""
        if self.redis_client:
            try:
                # Load all strategies from Redis
                keys = self.redis_client.keys(f"{self.redis_prefix}*")
                for key in keys:
                    domain = key.decode('utf-8').replace(self.redis_prefix, '')
                    strategy_json = self.redis_client.get(key)
                    if strategy_json:
                        self.strategies[domain] = json.loads(strategy_json)

                logger.info(f"✅ Loaded {len(self.strategies)} strategies from Redis")
                return
            except Exception as e:
                logger.warning(f"Failed to load from Redis: {e}, falling back to file cache")

        # Fallback to file cache
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.strategies = json.load(f)
                logger.info(f"✅ Loaded {len(self.strategies)} cached strategies from file")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.strategies = {}
        else:
            self.strategies = {}

    def _save_cache(self):
        """Save strategies to Redis and/or cache file"""
        # Save to Redis if available
        if self.redis_client:
            try:
                for domain, strategy in self.strategies.items():
                    key = f"{self.redis_prefix}{domain}"
                    self.redis_client.set(
                        key,
                        json.dumps(strategy),
                        ex=86400 * 30  # 30 day expiration
                    )
                logger.debug(f"💾 Saved {len(self.strategies)} strategies to Redis")
            except Exception as e:
                logger.error(f"Failed to save to Redis: {e}")

        # Always save to file as backup
        try:
            # Ensure parent directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.cache_file, 'w') as f:
                json.dump(self.strategies, f, indent=2)

            logger.debug(f"💾 Saved {len(self.strategies)} strategies to file")
        except Exception as e:
            logger.error(f"Failed to save cache to file: {e}")

    def record_strategy(
        self,
        url: str,
        extraction_method: str,
        proxy_type: str,
        browser_config: Dict[str, Any],
        success: bool,
        html_quality: str = "UNKNOWN",
        extraction_details: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        blocking_info: Optional[Dict[str, Any]] = None
    ):
        """
        Record a scraping strategy attempt.

        Args:
            url: Target URL
            extraction_method: Method used (json_ld, graphql, json, html)
            proxy_type: Proxy type (residential, web_unblocker, none)
            browser_config: Browser configuration used
            success: Whether the attempt succeeded
            html_quality: Quality of HTML captured
            extraction_details: Details about extraction (script IDs, endpoints, etc.)
            performance_metrics: Speed, reliability metrics
        """
        domain = urlparse(url).netloc

        # Initialize domain strategy if needed
        if domain not in self.strategies:
            self.strategies[domain] = {
                'domain': domain,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'total_attempts': 0,
                'successful_attempts': 0,
                'extraction_methods': {},
                'proxy_types': {},
                'browser_configs': [],
                'recommended_strategy': None
            }

        strategy = self.strategies[domain]

        # Update counters
        strategy['total_attempts'] += 1
        strategy['last_updated'] = datetime.now().isoformat()

        if success:
            strategy['successful_attempts'] += 1

        # Track extraction method performance
        if extraction_method not in strategy['extraction_methods']:
            strategy['extraction_methods'][extraction_method] = {
                'attempts': 0,
                'successes': 0,
                'success_rate': 0.0,
                'details': extraction_details or {}
            }

        method_stats = strategy['extraction_methods'][extraction_method]
        method_stats['attempts'] += 1
        if success:
            method_stats['successes'] += 1
        method_stats['success_rate'] = method_stats['successes'] / method_stats['attempts']

        if extraction_details:
            method_stats['details'].update(extraction_details)

        # Track proxy type performance
        if proxy_type not in strategy['proxy_types']:
            strategy['proxy_types'][proxy_type] = {
                'attempts': 0,
                'successes': 0,
                'success_rate': 0.0
            }

        proxy_stats = strategy['proxy_types'][proxy_type]
        proxy_stats['attempts'] += 1
        if success:
            proxy_stats['successes'] += 1
        proxy_stats['success_rate'] = proxy_stats['successes'] / proxy_stats['attempts']

        # Initialize request history if missing
        if 'request_history' not in strategy:
            strategy['request_history'] = {
                'total_requests': 0,
                'recent_blocks': 0,
                'block_rate': 0.0,
                'last_block_time': None,
                'consecutive_failures': 0,
                'history': []  # Last 20 results (1=success, 0=fail)
            }

        history = strategy['request_history']
        history['total_requests'] += 1

        # Track consecutive failures
        if success:
            history['consecutive_failures'] = 0
            history['history'].append(1)
        else:
            history['consecutive_failures'] += 1
            history['history'].append(0)

            # Check if it was a block
            if blocking_info and blocking_info.get('is_blocked'):
                history['recent_blocks'] += 1
                history['last_block_time'] = datetime.now().isoformat()

        # Keep history limited
        if len(history['history']) > 20:
            history['history'].pop(0)
            # Adjust recent blocks count if we removed a block event (approximate)
            # Ideally we'd store full event objects but simple list is faster

        # Calculate block rate (last 20 requests)
        failures = history['history'].count(0)
        history['block_rate'] = failures / len(history['history']) if history['history'] else 0.0

        # Store browser config if successful
        if success and browser_config:
            strategy['browser_configs'].append({
                'config': browser_config,
                'timestamp': datetime.now().isoformat(),
                'extraction_method': extraction_method,
                'proxy_type': proxy_type,
                'html_quality': html_quality,
                'performance': performance_metrics or {}
            })

            # Keep only last 10 successful configs
            strategy['browser_configs'] = strategy['browser_configs'][-10:]

        # Update recommended strategy
        self._update_recommended_strategy(domain)

        # Integrate with ConfigurationLearner if available
        if self.config_learner and browser_config:
            self.config_learner.record_attempt(
                domain=domain,
                config=browser_config,
                success=success,
                blocking_type="none" if success else "unknown",
                response_time=performance_metrics.get('elapsed_time', 0.0) if performance_metrics else 0.0
            )

        # Save to cache
        self._save_cache()

        logger.info(f"📊 Recorded strategy: {domain} - {extraction_method} via {proxy_type} - {'✅' if success else '❌'}")

    def _update_recommended_strategy(self, domain: str):
        """Update the recommended strategy for a domain"""
        strategy = self.strategies[domain]

        # Find best extraction method
        best_method = None
        best_method_score = 0.0

        for method, stats in strategy['extraction_methods'].items():
            if stats['attempts'] >= 1:  # Require at least 1 attempt
                # Score based on success rate and reliability
                reliability_scores = {
                    'json_ld': 1.0,  # Most reliable
                    'graphql': 0.9,
                    'json': 0.8,
                    'html': 0.6
                }

                score = stats['success_rate'] * reliability_scores.get(method, 0.5)

                if score > best_method_score:
                    best_method_score = score
                    best_method = method

        # Find best proxy type
        best_proxy = None
        best_proxy_score = 0.0

        for proxy, stats in strategy['proxy_types'].items():
            if stats['attempts'] >= 1:
                if stats['success_rate'] > best_proxy_score:
                    best_proxy_score = stats['success_rate']
                    best_proxy = proxy

        # Get best browser config
        best_browser_config = None
        if strategy['browser_configs']:
            # Use most recent successful config
            best_browser_config = strategy['browser_configs'][-1]['config']

        # Update recommendation
        if best_method and best_proxy:
            strategy['recommended_strategy'] = {
                'extraction_method': best_method,
                'extraction_details': strategy['extraction_methods'][best_method]['details'],
                'proxy_type': best_proxy,
                'browser_config': best_browser_config,
                'confidence': min(best_method_score, best_proxy_score),
                'success_rate': strategy['successful_attempts'] / max(strategy['total_attempts'], 1),
                'last_updated': datetime.now().isoformat()
            }

    def get_strategy(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get recommended strategy for a URL.

        Args:
            url: Target URL

        Returns:
            Strategy dict or None if no strategy cached
        """
        domain = urlparse(url).netloc

        if domain in self.strategies:
            strategy = self.strategies[domain]
            recommended = strategy.get('recommended_strategy')

            if recommended:
                logger.info(f"📚 Found cached strategy for {domain}")
                logger.info(f"   Method: {recommended['extraction_method']}")
                logger.info(f"   Proxy: {recommended['proxy_type']}")
                logger.info(f"   Confidence: {recommended['confidence']:.2f}")
                logger.info(f"   Success Rate: {recommended['success_rate']:.2%}")

                return recommended

        logger.debug(f"No cached strategy for {domain}")
        return None

    def is_strategy_degraded(self, domain: str) -> bool:
        """
        Check if cached strategy is failing at scale

        Returns True if:
        - Block rate > 30% in last 20 requests
        - 3+ consecutive failures
        - Last success > 1 hour ago (if attempts made)
        """
        if domain not in self.strategies:
            return False

        strategy = self.strategies[domain]
        if 'request_history' not in strategy:
            return False

        history = strategy['request_history']

        # Criteria 1: High failure rate
        if history['block_rate'] > 0.3:
            logger.warning(f"⚠️ Strategy degraded for {domain}: High block rate ({history['block_rate']:.1%})")
            return True

        # Criteria 2: Consecutive failures
        if history['consecutive_failures'] >= 3:
            logger.warning(f"⚠️ Strategy degraded for {domain}: {history['consecutive_failures']} consecutive failures")
            return True

        return False

    def get_escalated_strategy(
        self,
        domain: str,
        current_strategy: Optional[Dict] = None,
        blocking_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Return more aggressive config when current strategy fails

        Escalation levels:
        1. Add humanize=True
        2. Add stealth=True
        3. Switch proxy type
        4. Increase timeout

        Smart Escalation:
        - If 'challenge' detected (Cloudflare/Captcha) -> Jump to Web Unblocker
        """
        # Base config
        config = {
            'proxy_type': 'residential',
            'browser_config': {
                'humanize': True,
                'stealth': False,
                'geoip': True,  # Enable GeoIP by default for better trust score
                'timeout': 30000
            }
        }

        if current_strategy:
            # Start from current
            config['proxy_type'] = current_strategy.get('proxy_type', 'residential')
            if 'browser_config' in current_strategy:
                config['browser_config'].update(current_strategy['browser_config'])

        # SMART ESCALATION: Check blocking info
        if blocking_info and blocking_info.get('block_type') == 'challenge':
            logger.warning(f"🚨 Challenge detected for {domain} - escalating directly to Web Unblocker")
            config['proxy_type'] = 'web_unblocker'
            config['browser_config']['stealth'] = True
            config['browser_config']['geoip'] = True # Force enable for unblocker
            config['browser_config']['timeout'] = 60000
            return config

        # Standard Escalation Logic
        browser_conf = config['browser_config']

        # Level 1: Ensure humanize
        if not browser_conf.get('humanize'):
            browser_conf['humanize'] = True
            return config

        # Level 2: Enable stealth
        if not browser_conf.get('stealth'):
            browser_conf['stealth'] = True
            return config

        # Level 3: Increase timeout
        if browser_conf.get('timeout', 30000) < 60000:
            browser_conf['timeout'] = 60000
            return config

        # Level 4: Switch proxy (if residential, try web_unblocker)
        if config['proxy_type'] == 'residential':
            config['proxy_type'] = 'web_unblocker'
            # Reset browser config for unblocker (it handles most things)
            browser_conf['stealth'] = False
            browser_conf['geoip'] = True  # Force enable for unblocker
            return config

        # Level 5: Max power
        browser_conf['timeout'] = 90000
        browser_conf['stealth'] = True
        browser_conf['geoip'] = True  # Force enable for unblocker
        config['proxy_type'] = 'web_unblocker'

        return config

    def get_all_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get all cached strategies"""
        return self.strategies

    def clear_domain(self, domain: str):
        """Clear cached strategy for a domain"""
        if domain in self.strategies:
            del self.strategies[domain]
            self._save_cache()
            logger.info(f"🗑️  Cleared strategy for {domain}")

    def export_summary(self) -> Dict[str, Any]:
        """Export summary of all strategies"""
        summary = {
            'total_domains': len(self.strategies),
            'domains': {}
        }

        for domain, strategy in self.strategies.items():
            summary['domains'][domain] = {
                'total_attempts': strategy['total_attempts'],
                'success_rate': strategy['successful_attempts'] / max(strategy['total_attempts'], 1),
                'recommended_method': strategy.get('recommended_strategy', {}).get('extraction_method'),
                'recommended_proxy': strategy.get('recommended_strategy', {}).get('proxy_type'),
                'last_updated': strategy['last_updated']
            }

        return summary

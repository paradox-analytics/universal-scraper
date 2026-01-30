"""
Configuration Learner

Learns which browser configurations work best for each domain.
Tracks success rates, stores successful configs, and suggests optimizations.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ConfigurationLearner:
    """
    Learns optimal browser configurations per domain.
    
    Tracks:
    - Success/failure rates per configuration
    - Best performing configs per domain
    - Temporal patterns (time-based success rates)
    - Blocking type correlations
    """
    
    def __init__(self, redis_cache=None):
        """
        Initialize configuration learner.
        
        Args:
            redis_cache: Optional Redis cache for persistent storage
        """
        self.redis_cache = redis_cache
        
        # In-memory storage (fallback if no Redis)
        self.domain_configs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.config_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🧠 Configuration Learner initialized")
    
    def record_attempt(
        self,
        domain: str,
        config: Dict[str, Any],
        success: bool,
        blocking_type: str = "none",
        response_time: float = 0.0,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Record a configuration attempt.
        
        Args:
            domain: Target domain
            config: Browser configuration used
            success: Whether the attempt succeeded
            blocking_type: Type of blocking encountered (if any)
            response_time: Response time in seconds
            details: Additional details about the attempt
        """
        # Create config hash for tracking
        config_hash = self._hash_config(config)
        
        # Initialize stats if needed
        if config_hash not in self.config_stats:
            self.config_stats[config_hash] = {
                'config': config,
                'total_attempts': 0,
                'successes': 0,
                'failures': 0,
                'success_rate': 0.0,
                'avg_response_time': 0.0,
                'blocking_types': defaultdict(int),
                'domains': set(),
                'last_success': None,
                'last_failure': None,
            }
        
        stats = self.config_stats[config_hash]
        
        # Update stats
        stats['total_attempts'] += 1
        stats['domains'].add(domain)
        
        if success:
            stats['successes'] += 1
            stats['last_success'] = datetime.now().isoformat()
        else:
            stats['failures'] += 1
            stats['last_failure'] = datetime.now().isoformat()
            stats['blocking_types'][blocking_type] += 1
        
        # Update success rate
        stats['success_rate'] = stats['successes'] / stats['total_attempts']
        
        # Update avg response time
        if response_time > 0:
            current_avg = stats['avg_response_time']
            n = stats['total_attempts']
            stats['avg_response_time'] = ((current_avg * (n - 1)) + response_time) / n
        
        # Store domain-specific record
        self.domain_configs[domain].append({
            'config_hash': config_hash,
            'success': success,
            'blocking_type': blocking_type,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        })
        
        # Persist to Redis if available
        if self.redis_cache:
            self._persist_to_redis(domain, config_hash, stats)
        
        logger.debug(f"Recorded attempt: domain={domain}, success={success}, config_hash={config_hash[:8]}")
    
    def get_best_config(
        self,
        domain: str,
        min_attempts: int = 3,
        recency_weight: float = 0.3
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best performing configuration for a domain.
        
        Args:
            domain: Target domain
            min_attempts: Minimum attempts required to consider a config
            recency_weight: Weight for recent successes (0-1)
            
        Returns:
            Best configuration dict or None
        """
        # Get all configs tried for this domain
        domain_attempts = self.domain_configs.get(domain, [])
        
        if not domain_attempts:
            logger.debug(f"No learned configs for {domain}")
            return None
        
        # Group by config hash
        config_performance = defaultdict(lambda: {'successes': 0, 'attempts': 0, 'recent_successes': 0})
        
        now = datetime.now()
        recent_threshold = now - timedelta(hours=24)
        
        for attempt in domain_attempts:
            config_hash = attempt['config_hash']
            config_performance[config_hash]['attempts'] += 1
            
            if attempt['success']:
                config_performance[config_hash]['successes'] += 1
                
                # Check if recent
                attempt_time = datetime.fromisoformat(attempt['timestamp'])
                if attempt_time > recent_threshold:
                    config_performance[config_hash]['recent_successes'] += 1
        
        # Find best config
        best_config_hash = None
        best_score = 0.0
        
        for config_hash, perf in config_performance.items():
            if perf['attempts'] < min_attempts:
                continue
            
            # Calculate score
            success_rate = perf['successes'] / perf['attempts']
            recency_score = perf['recent_successes'] / max(perf['successes'], 1)
            
            score = (success_rate * (1 - recency_weight)) + (recency_score * recency_weight)
            
            if score > best_score:
                best_score = score
                best_config_hash = config_hash
        
        if best_config_hash:
            config = self.config_stats[best_config_hash]['config']
            logger.info(f"Best config for {domain}: score={best_score:.2f}, hash={best_config_hash[:8]}")
            return config
        
        return None
    
    def get_recommendations(
        self,
        domain: str,
        blocking_type: str = "none"
    ) -> List[str]:
        """
        Get recommendations for improving success rate.
        
        Args:
            domain: Target domain
            blocking_type: Type of blocking encountered
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check if we have data for this domain
        domain_attempts = self.domain_configs.get(domain, [])
        
        if not domain_attempts:
            recommendations.append("No historical data for this domain - trying multiple configurations")
            return recommendations
        
        # Analyze recent failures
        recent_failures = [a for a in domain_attempts[-10:] if not a['success']]
        
        if len(recent_failures) > 7:
            recommendations.append("High failure rate detected - consider using Web Unblocker")
        
        # Analyze blocking types
        blocking_counts = defaultdict(int)
        for attempt in domain_attempts[-20:]:
            if not attempt['success']:
                blocking_counts[attempt['blocking_type']] += 1
        
        if blocking_counts:
            most_common_block = max(blocking_counts.items(), key=lambda x: x[1])
            recommendations.append(f"Most common blocking: {most_common_block[0]} ({most_common_block[1]} times)")
        
        # Check success rate trend
        if len(domain_attempts) >= 10:
            recent_success_rate = sum(1 for a in domain_attempts[-10:] if a['success']) / 10
            overall_success_rate = sum(1 for a in domain_attempts if a['success']) / len(domain_attempts)
            
            if recent_success_rate < overall_success_rate * 0.5:
                recommendations.append("Success rate declining - site may have updated anti-bot protection")
        
        return recommendations
    
    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """
        Get statistics for a specific domain.
        
        Args:
            domain: Target domain
            
        Returns:
            Statistics dict
        """
        attempts = self.domain_configs.get(domain, [])
        
        if not attempts:
            return {
                'total_attempts': 0,
                'success_rate': 0.0,
                'configs_tried': 0
            }
        
        successes = sum(1 for a in attempts if a['success'])
        unique_configs = len(set(a['config_hash'] for a in attempts))
        
        # Blocking type breakdown
        blocking_breakdown = defaultdict(int)
        for attempt in attempts:
            if not attempt['success']:
                blocking_breakdown[attempt['blocking_type']] += 1
        
        return {
            'total_attempts': len(attempts),
            'successes': successes,
            'failures': len(attempts) - successes,
            'success_rate': successes / len(attempts),
            'configs_tried': unique_configs,
            'blocking_breakdown': dict(blocking_breakdown),
            'avg_response_time': sum(a.get('response_time', 0) for a in attempts) / len(attempts)
        }
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Create a hash for a configuration"""
        # Sort keys for consistent hashing
        config_str = json.dumps(config, sort_keys=True)
        import hashlib
        return hashlib.md5(config_str.encode()).hexdigest()
    
    
    def get_optimal_timeout(self, domain: str, default_timeout: int = 60000) -> int:
        """
        Calculate optimal timeout for a domain based on historical data.
        
        Args:
            domain: Target domain
            default_timeout: Default timeout in milliseconds
            
        Returns:
            Optimal timeout in milliseconds
        """
        attempts = self.domain_configs.get(domain, [])
        
        if not attempts:
            return default_timeout
        
        # Analyze response times from successful attempts
        successful_times = [
            a.get('response_time', 0) * 1000  # Convert to ms
            for a in attempts
            if a.get('success') and a.get('response_time', 0) > 0
        ]
        
        if successful_times:
            # Use 95th percentile + 50% buffer
            import statistics
            p95 = statistics.quantiles(successful_times, n=20)[18]  # 95th percentile
            optimal = int(p95 * 1.5)
            
            # Clamp between 30s and 300s
            optimal = max(30000, min(300000, optimal))
            
            logger.info(f"Optimal timeout for {domain}: {optimal}ms (based on {len(successful_times)} successful attempts)")
            return optimal
        
        # Check for timeout patterns
        timeout_count = sum(1 for a in attempts if a.get('blocking_type') == 'timeout')
        
        if timeout_count > 3:
            # Increase timeout progressively
            multiplier = 1.0 + (timeout_count * 0.5)  # 1.5x, 2.0x, 2.5x, etc.
            new_timeout = int(default_timeout * multiplier)
            new_timeout = min(300000, new_timeout)  # Max 5 minutes
            
            logger.info(f"Increasing timeout for {domain} to {new_timeout}ms due to {timeout_count} timeouts")
            return new_timeout
        
        return default_timeout
    
    def get_optimal_strategy(self, domain: str) -> Dict[str, Any]:
        """
        Determine optimal scraping strategy for a domain.
        
        Returns strategy recommendations including:
        - Preset to use
        - Whether to use Web Unblocker
        - Proxy rotation frequency
        - Other optimizations
        
        Args:
            domain: Target domain
            
        Returns:
            Strategy dict with recommendations
        """
        attempts = self.domain_configs.get(domain, [])
        
        if not attempts:
            return {
                'preset': 'balanced',
                'use_web_unblocker': False,
                'proxy_rotation': 'per_domain',
                'reason': 'No historical data - using balanced preset'
            }
        
        # Analyze blocking patterns
        blocking_counts = defaultdict(int)
        for attempt in attempts:
            if not attempt.get('success'):
                blocking_counts[attempt.get('blocking_type', 'unknown')] += 1
        
        total_attempts = len(attempts)
        success_rate = sum(1 for a in attempts if a.get('success')) / total_attempts
        
        # Decision tree for strategy
        strategy = {
            'preset': 'balanced',
            'use_web_unblocker': False,
            'proxy_rotation': 'per_domain',
            'increase_delays': False,
            'use_static_fallback': False,
            'reason': ''
        }
        
        # Check for Kasada/advanced protection
        if blocking_counts.get('kasada', 0) > 0 or blocking_counts.get('datadome', 0) > 0:
            strategy['use_web_unblocker'] = True
            strategy['preset'] = 'stealth'
            strategy['reason'] = f"Advanced anti-bot detected ({max(blocking_counts, key=blocking_counts.get)})"
            return strategy
        
        # Check for Cloudflare
        if blocking_counts.get('cloudflare', 0) > 2:
            strategy['preset'] = 'stealth'
            strategy['proxy_rotation'] = 'per_request'
            strategy['reason'] = "Cloudflare detected - using stealth mode with frequent proxy rotation"
            return strategy
        
        # Check for rate limiting
        if blocking_counts.get('rate_limit', 0) > 2:
            strategy['increase_delays'] = True
            strategy['proxy_rotation'] = 'per_request'
            strategy['reason'] = "Rate limiting detected - increasing delays and rotating proxies"
            return strategy
        
        # Check for consistent timeouts
        if blocking_counts.get('timeout', 0) > 2:
            if success_rate < 0.1:
                # Very low success rate with timeouts - try Web Unblocker
                strategy['use_web_unblocker'] = True
                strategy['reason'] = "Consistent timeouts with low success rate - recommending Web Unblocker"
            else:
                # Some success - just need longer timeout (handled by get_optimal_timeout)
                strategy['reason'] = "Timeouts detected - timeout will be automatically increased"
            return strategy
        
        # Check for 403 blocks
        if blocking_counts.get('generic_block', 0) > 2 or blocking_counts.get('ip_ban', 0) > 0:
            strategy['preset'] = 'stealth'
            strategy['proxy_rotation'] = 'per_request'
            strategy['reason'] = "IP blocking detected - using stealth with aggressive proxy rotation"
            return strategy
        
        # If success rate is good, use faster settings
        if success_rate > 0.8:
            strategy['preset'] = 'aggressive'
            strategy['use_static_fallback'] = True
            strategy['reason'] = f"High success rate ({success_rate:.1%}) - using aggressive mode for speed"
            return strategy
        
        # Default balanced approach
        strategy['reason'] = f"Using balanced approach (success rate: {success_rate:.1%})"
        return strategy
    
    def _persist_to_redis(self, domain: str, config_hash: str, stats: Dict[str, Any]):
        """Persist stats to Redis"""
        if not self.redis_cache:
            return
        
        try:
            # Convert sets to lists for JSON serialization
            stats_copy = stats.copy()
            stats_copy['domains'] = list(stats_copy['domains'])
            stats_copy['blocking_types'] = dict(stats_copy['blocking_types'])
            
            key = f"config_learner:stats:{config_hash}"
            self.redis_cache.set(key, json.dumps(stats_copy), ttl=86400 * 30)  # 30 days
            
            # Also store domain-specific index
            domain_key = f"config_learner:domain:{domain}"
            self.redis_cache.sadd(domain_key, config_hash)
            
        except Exception as e:
            logger.warning(f"Failed to persist to Redis: {e}")

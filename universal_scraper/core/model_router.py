"""
Model Router - 3-Tier Model Selection
Routes tasks to appropriate model based on frequency and complexity

Architecture:
A. Router/Classification (fastest, most frequent)
   - Model: gpt-4o-mini or claude-haiku
   - Tasks: "Is this page type X?", "Do we have a template?", "Has layout changed?"
   - Frequency: Every request
   - Latency Target: <100ms

B. Template Generation (occasional, higher value)
   - Model: gpt-4o-mini or claude-sonnet-3.5
   - Tasks: Generate extraction plan, create template spec JSON
   - Frequency: ~5-10% of requests (cache miss)
   - Latency Target: <2s

C. Recovery Mode (rare, hardest)
   - Model: gpt-4o or claude-opus
   - Tasks: Re-derive template when validation fails, handle odd layouts
   - Frequency: <1% of requests
   - Latency Target: <10s (acceptable for rare cases)
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model tier classification"""
    ROUTER = "router"  # Fast classification/routing
    TEMPLATE = "template"  # Template generation
    RECOVERY = "recovery"  # Recovery mode (rare, complex)


class ModelRouter:
    """
    Routes tasks to appropriate model based on tier
    
    Usage:
        router = ModelRouter(
            router_model="gpt-4o-mini",
            template_model="gpt-4o-mini",
            recovery_model="gpt-4o"
        )
        
        # For routing/classification
        model = router.get_model(ModelTier.ROUTER)
        
        # For template generation
        model = router.get_model(ModelTier.TEMPLATE)
        
        # For recovery
        model = router.get_model(ModelTier.RECOVERY)
    """
    
    # Default model configurations
    DEFAULT_MODELS = {
        ModelTier.ROUTER: "gpt-4o-mini",  # Fast, cheap
        ModelTier.TEMPLATE: "gpt-4o-mini",  # Balanced (can upgrade to sonnet-3.5)
        ModelTier.RECOVERY: "gpt-4o"  # Powerful for complex cases
    }
    
    # Model cost per 1M tokens (input/output average)
    MODEL_COSTS = {
        "gpt-4o-mini": 0.15,  # $0.15/1M tokens
        "gpt-4o": 2.50,  # $2.50/1M tokens
        "claude-haiku": 0.25,  # $0.25/1M tokens
        "claude-sonnet-3.5": 3.00,  # $3.00/1M tokens
        "claude-opus": 15.00,  # $15.00/1M tokens
    }
    
    def __init__(
        self,
        router_model: Optional[str] = None,
        template_model: Optional[str] = None,
        recovery_model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize model router
        
        Args:
            router_model: Model for routing/classification (default: gpt-4o-mini)
            template_model: Model for template generation (default: gpt-4o-mini)
            recovery_model: Model for recovery mode (default: gpt-4o)
            api_key: API key (if needed for validation)
        """
        self.models = {
            ModelTier.ROUTER: router_model or self.DEFAULT_MODELS[ModelTier.ROUTER],
            ModelTier.TEMPLATE: template_model or self.DEFAULT_MODELS[ModelTier.TEMPLATE],
            ModelTier.RECOVERY: recovery_model or self.DEFAULT_MODELS[ModelTier.RECOVERY]
        }
        
        self.api_key = api_key
        
        logger.info(f" Model Router initialized:")
        logger.info(f"   Router: {self.models[ModelTier.ROUTER]}")
        logger.info(f"   Template: {self.models[ModelTier.TEMPLATE]}")
        logger.info(f"   Recovery: {self.models[ModelTier.RECOVERY]}")
    
    def get_model(self, tier: ModelTier) -> str:
        """
        Get model name for tier
        
        Args:
            tier: Model tier
            
        Returns:
            Model name string
        """
        return self.models.get(tier, self.DEFAULT_MODELS[tier])
    
    def get_cost_estimate(
        self,
        tier: ModelTier,
        input_tokens: int,
        output_tokens: int = 0
    ) -> float:
        """
        Estimate cost for model call
        
        Args:
            tier: Model tier
            input_tokens: Input token count
            output_tokens: Output token count (default: 0)
            
        Returns:
            Estimated cost in USD
        """
        model = self.get_model(tier)
        cost_per_million = self.MODEL_COSTS.get(model, 1.0)  # Default $1/1M
        
        total_tokens = input_tokens + output_tokens
        cost = (total_tokens / 1_000_000) * cost_per_million
        
        return cost
    
    def should_use_recovery(
        self,
        attempts: int,
        last_quality: float,
        quality_threshold: float = 0.7
    ) -> bool:
        """
        Determine if recovery mode should be used
        
        Args:
            attempts: Number of extraction attempts
            last_quality: Quality score from last attempt
            quality_threshold: Minimum quality threshold
            
        Returns:
            True if recovery mode should be used
        """
        # Use recovery if:
        # 1. Multiple failed attempts (>= 2)
        # 2. Quality below threshold
        if attempts >= 2 and last_quality < quality_threshold:
            return True
        
        return False
    
    def get_tier_for_task(self, task_type: str) -> ModelTier:
        """
        Get appropriate tier for task type
        
        Args:
            task_type: Task type ('classification', 'template', 'recovery', etc.)
            
        Returns:
            Model tier
        """
        task_lower = task_type.lower()
        
        if any(keyword in task_lower for keyword in ['classify', 'route', 'check', 'detect']):
            return ModelTier.ROUTER
        elif any(keyword in task_lower for keyword in ['recover', 'fix', 'retry', 'fallback']):
            return ModelTier.RECOVERY
        else:
            return ModelTier.TEMPLATE  # Default to template generation
    
    def upgrade_model(self, tier: ModelTier, new_model: str) -> bool:
        """
        Upgrade model for tier (for testing/optimization)
        
        Args:
            tier: Model tier
            new_model: New model name
            
        Returns:
            True if upgraded successfully
        """
        if new_model in self.MODEL_COSTS:
            old_model = self.models[tier]
            self.models[tier] = new_model
            logger.info(f" Upgraded {tier.value} model: {old_model} → {new_model}")
            return True
        else:
            logger.warning(f" Unknown model: {new_model}")
            return False




"""
Universal JSON Quality Validator
Fast, LLM-free validation to filter out irrelevant JSON data
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class JSONQualityValidator:
    """
    Performs fast, universal, and LLM-free quality validation on extracted JSON data.
    Checks for field overlap, presence of metadata keywords, and data density.
    
    This is a first-pass filter to reject obvious garbage (tracking data, config, etc.)
    before expensive LLM validation.
    """
    
    def __init__(
        self,
        min_field_overlap_ratio: float = 0.3,
        min_data_density_score: float = 0.2
    ):
        """
        Initialize the JSON Quality Validator
        
        Args:
            min_field_overlap_ratio: Minimum ratio of requested fields found in extracted items
            min_data_density_score: Minimum score for data density (0-1)
        """
        self.min_field_overlap_ratio = min_field_overlap_ratio
        self.min_data_density_score = min_data_density_score
        
        # Metadata/tracking keywords (bad signals)
        self.metadata_keywords = [
            'session', 'token', 'tracking', 'cookie', 'correlation', 'guid', 'config',
            'api_key', 'id_token', 'access_token', 'refresh_token', 'user_id',
            'client_id', 'client_secret', 'timestamp', 'version', 'build_id',
            'environment', 'debug', 'log', 'error', 'status', 'code', 'message',
            'event', 'event_id', 'platform', 'device', 'browser', 'os', 'locale',
            'country', 'region', 'zone', 'ip_address', 'user_agent', 'referrer',
            'utm_source', 'utm_medium', 'utm_campaign', 'gclid', 'fbclid',
            'csrf_token', 'nonce', 'signature', 'hash', 'checksum', 'fingerprint',
            'x_ebay_c', 'correlation_session', 'signed-out', 'signed-in', 'recognized'
        ]
        
        # Data keywords (good signals)
        self.data_keywords = [
            'title', 'name', 'product', 'item', 'price', 'cost', 'amount', 'currency',
            'description', 'content', 'text', 'body', 'author', 'user', 'publisher',
            'date', 'time', 'published', 'created', 'updated', 'url', 'link', 'href',
            'image', 'src', 'photo', 'picture', 'thumbnail', 'category', 'tag', 'type',
            'rating', 'score', 'review', 'comment', 'feedback', 'stock', 'quantity',
            'availability', 'condition', 'shipping', 'delivery', 'location', 'address',
            'phone', 'email', 'contact', 'brand', 'model', 'sku', 'mpn', 'gtin', 'isbn',
            'dimensions', 'weight', 'size', 'color', 'material', 'features', 'specs'
        ]
    
    def validate(
        self,
        extracted_items: List[Dict[str, Any]],
        requested_fields: List[str],
        extraction_context: Optional[str] = None
    ) -> Tuple[bool, str, float]:
        """
        Validate if extracted JSON items are likely to be real data (not metadata/tracking)
        
        Args:
            extracted_items: List of extracted dictionaries
            requested_fields: List of fields user requested
            extraction_context: Optional context about what user wants to extract
        
        Returns:
            Tuple of (is_valid, reason, confidence_score)
        """
        if not extracted_items:
            return False, "No items to validate", 0.0
        
        logger.info(f" Validating {len(extracted_items)} JSON items...")
        
        #  FREQUENCY-BASED VALIDATION (Universal!)
        # Valuable data has HIGH frequency patterns!
        # If JSON extraction returns < 5 items, it's likely garbage metadata
        item_count = len(extracted_items)
        
        if item_count < 5:
            logger.warning(f"    Low frequency: {item_count} items (expected 5+ for real data)")
            logger.warning(f"    Real data appears multiple times (15+ posts, 20+ products, etc.)")
            logger.warning(f"    Single items are usually metadata/tracking/config")
            return False, f"Low frequency: {item_count} items (expected 5+ for real data)", 0.1
        
        logger.info(f"    Good frequency: {item_count} items (likely real data)")
        
        # Collect all keys from all items
        all_keys = set()
        for item in extracted_items[:10]:  # Sample first 10 items
            if isinstance(item, dict):
                all_keys.update(item.keys())
        
        if not all_keys:
            return False, "No keys found in items", 0.0
        
        # Convert keys to lowercase for comparison
        keys_lower = [str(k).lower() for k in all_keys]
        
        # 1. Check for metadata/tracking keywords (BAD)
        metadata_count = sum(
            1 for key in keys_lower 
            for keyword in self.metadata_keywords 
            if keyword in key
        )
        metadata_ratio = metadata_count / len(keys_lower) if keys_lower else 0
        
        # 2. Check for data keywords (GOOD)
        data_count = sum(
            1 for key in keys_lower 
            for keyword in self.data_keywords 
            if keyword in key
        )
        data_ratio = data_count / len(keys_lower) if keys_lower else 0
        
        # 3. Check field overlap with requested fields
        requested_lower = [f.lower() for f in requested_fields]
        overlap_count = sum(1 for key in keys_lower if any(req in key or key in req for req in requested_lower))
        field_overlap_ratio = overlap_count / len(requested_fields) if requested_fields else 0
        
        # 4. Check for actual data values (not None/empty)
        non_null_count = 0
        total_values = 0
        for item in extracted_items[:10]:
            if isinstance(item, dict):
                for value in item.values():
                    total_values += 1
                    if value is not None and value != '' and value != {}:
                        non_null_count += 1
        
        value_density = non_null_count / total_values if total_values > 0 else 0
        
        # Calculate overall confidence score
        # Penalize metadata, reward data keywords and field overlap
        confidence = (
            (1 - metadata_ratio) * 0.25 +  # Less metadata = better
            data_ratio * 0.25 +              # More data keywords = better
            field_overlap_ratio * 0.25 +     # More requested fields = better
            value_density * 0.25             # More non-null values = better
        )
        
        logger.info(f"    JSON validation scores:")
        logger.info(f"      Metadata: {metadata_ratio:.0%}, Field overlap: {field_overlap_ratio:.0%}, Data: {data_ratio:.0%}, Values: {value_density:.0%}")
        
        # Validation logic
        if metadata_ratio > 0.5:
            return False, f"High metadata content: {metadata_ratio:.0%}", confidence
        
        if data_ratio < 0.1 and field_overlap_ratio < 0.2:
            return False, "No data keywords found", confidence
        
        if value_density < 0.3:
            return False, f"Low value density: {value_density:.0%}", confidence
        
        if field_overlap_ratio < self.min_field_overlap_ratio and data_ratio < 0.3:
            return False, f"Insufficient field overlap and data keywords", confidence
        
        # Passed all checks
        logger.info(f"    JSON validation passed (confidence: {confidence:.2f})")
        return True, "Validation passed", confidence
    
    def suggest_fallback(self, failure_reason: str) -> str:
        """
        Suggest a fallback strategy based on failure reason
        
        Args:
            failure_reason: The reason validation failed
        
        Returns:
            Human-readable suggestion
        """
        if "metadata" in failure_reason.lower():
            return "Fall back to HTML extraction (JSON contains tracking/config data)"
        elif "field overlap" in failure_reason.lower():
            return "Fall back to HTML extraction (JSON doesn't match requested fields)"
        elif "value density" in failure_reason.lower():
            return "Fall back to HTML extraction (JSON has too many null values)"
        elif "no data keywords" in failure_reason.lower():
            return "Fall back to HTML extraction (JSON lacks data keywords)"
        else:
            return "Fall back to HTML extraction (JSON quality check failed)"

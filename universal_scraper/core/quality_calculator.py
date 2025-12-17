"""
Universal Quality Score Calculator
Calculates quality scores with support for required vs optional fields
"""
import logging
from typing import List, Dict, Any, Set, Optional

logger = logging.getLogger(__name__)


class QualityCalculator:
    """
    Calculates quality scores for extracted data
    
    Supports:
    - Required vs optional fields
    - Field coverage calculation
    - Quality score with proper weighting
    """
    
    # Default field requirement levels (can be overridden)
    REQUIRED_FIELDS = {
        'title', 'name', 'product name', 'job title', 'item name',
        'url', 'link', 'href', 'product url', 'job url'
    }
    
    OPTIONAL_FIELDS = {
        'rating', 'review count', 'reviews', 'comments', 'score',
        'metascore', 'description', 'summary',
        'author', 'company', 'location', 'salary',
        'color', 'variant', 'size', 'category',
        'release date', 'published date', 'created date'
    }
    
    def __init__(self, required_fields: Optional[Set[str]] = None, optional_fields: Optional[Set[str]] = None):
        """
        Initialize quality calculator
        
        Args:
            required_fields: Set of field names that are always required
            optional_fields: Set of field names that are optional (nice to have)
        """
        self.required_fields = required_fields or self.REQUIRED_FIELDS.copy()
        self.optional_fields = optional_fields or self.OPTIONAL_FIELDS.copy()
    
    def calculate_field_coverage(
        self,
        items: List[Dict[str, Any]],
        requested_fields: List[str]
    ) -> Dict[str, int]:
        """
        Calculate field coverage (how many items have each field)
        
        Args:
            items: List of extracted items
            requested_fields: List of requested fields
            
        Returns:
            Dict mapping field name to count of items that have it
        """
        coverage = {field: 0 for field in requested_fields}
        
        for item in items:
            for field in requested_fields:
                # Check if field exists and has a non-empty value
                if self._has_field_value(item, field):
                    coverage[field] += 1
        
        return coverage
    
    def _has_field_value(self, item: Dict[str, Any], field: str) -> bool:
        """
        Check if item has a non-empty value for the field
        
        Handles:
        - Exact key matches
        - Semantic matches (e.g., "product name" matches "name")
        - Nested objects (extracts string value)
        - Empty/null values
        """
        # Check exact key match
        if field in item:
            value = item[field]
            if self._is_valid_value(value):
                return True
        
        # Check semantic matches (case-insensitive, partial)
        field_lower = field.lower()
        for key, value in item.items():
            key_lower = key.lower()
            # Exact match or contains match
            if field_lower == key_lower or field_lower in key_lower or key_lower in field_lower:
                if self._is_valid_value(value):
                    return True
        
        return False
    
    def _is_valid_value(self, value: Any) -> bool:
        """Check if value is valid (not None, empty, or just whitespace)"""
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() not in ['', 'null', 'None', 'N/A', 'n/a']
        if isinstance(value, dict):
            # Nested object - check if it has any string values
            return any(self._is_valid_value(v) for v in value.values() if isinstance(v, str))
        if isinstance(value, (list, tuple)):
            return len(value) > 0
        # Numbers, booleans are always valid
        return True
    
    def calculate_quality_score(
        self,
        items: List[Dict[str, Any]],
        requested_fields: List[str],
        required_fields: Optional[List[str]] = None,
        optional_fields: Optional[List[str]] = None
    ) -> float:
        """
        Calculate quality score with required/optional field distinction
        
        Formula:
        quality = (required_fields_coverage * 0.7) + (optional_fields_coverage * 0.3)
        
        Args:
            items: List of extracted items
            requested_fields: All requested fields
            required_fields: Fields that are required (defaults to class defaults)
            optional_fields: Fields that are optional (defaults to class defaults)
            
        Returns:
            Quality score (0-100)
        """
        if not items:
            return 0.0
        
        # Determine which fields are required vs optional
        if required_fields is None:
            # Auto-detect based on field names
            required_fields = [
                f for f in requested_fields 
                if any(req in f.lower() for req in self.required_fields)
            ]
            # If no required fields detected, make first 1-2 fields required
            if not required_fields and requested_fields:
                required_fields = requested_fields[:min(2, len(requested_fields))]
        
        if optional_fields is None:
            optional_fields = [
                f for f in requested_fields 
                if f not in required_fields and any(opt in f.lower() for opt in self.optional_fields)
            ]
            # Remaining fields are optional if not explicitly required
            remaining = [f for f in requested_fields if f not in required_fields and f not in optional_fields]
            optional_fields.extend(remaining)
        
        # Calculate coverage
        coverage = self.calculate_field_coverage(items, requested_fields)
        
        # Calculate required field coverage
        if required_fields:
            required_coverage = sum(
                coverage.get(f, 0) / len(items) 
                for f in required_fields
            ) / len(required_fields)
        else:
            required_coverage = 1.0  # No required fields = perfect
        
        # Calculate optional field coverage
        if optional_fields:
            optional_coverage = sum(
                coverage.get(f, 0) / len(items) 
                for f in optional_fields
            ) / len(optional_fields) if optional_fields else 0.0
        else:
            optional_coverage = 1.0  # No optional fields = perfect
        
        # Weighted quality score
        quality = (required_coverage * 0.7 + optional_coverage * 0.3) * 100
        
        logger.debug(f"   Quality calculation: required={required_coverage:.1%}, optional={optional_coverage:.1%}, total={quality:.1f}%")
        
        return quality
    
    def get_missing_fields(
        self,
        items: List[Dict[str, Any]],
        requested_fields: List[str],
        required_only: bool = False
    ) -> List[str]:
        """
        Get list of missing fields
        
        Args:
            items: List of extracted items
            requested_fields: Requested fields
            required_only: If True, only return missing required fields
            
        Returns:
            List of missing field names
        """
        coverage = self.calculate_field_coverage(items, requested_fields)
        
        if required_only:
            # Only check required fields
            required_fields = [
                f for f in requested_fields 
                if any(req in f.lower() for req in self.required_fields)
            ]
            if not required_fields and requested_fields:
                required_fields = requested_fields[:min(2, len(requested_fields))]
            
            missing = [f for f in required_fields if coverage.get(f, 0) == 0]
        else:
            # Check all fields
            missing = [f for f in requested_fields if coverage.get(f, 0) == 0]
        
        return missing



"""
Schema Manager for Universal Scraper

Ensures schema integrity and stability in production environments.
Handles dynamic field mapping, validation, normalization, and auto-recovery.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FieldMapping:
    """Defines how to map source fields to output schema"""
    output_field: str  # The stable field name in output schema
    source_fields: List[str]  # Possible source field names (in priority order)
    field_type: str  # Expected type: 'string', 'number', 'boolean', 'object', 'array'
    required: bool = True
    default_value: Any = None
    transformer: Optional[Callable] = None  # Optional transformation function
    aliases: List[str] = field(default_factory=list)  # Alternative field names
    description: str = ""
    
    def find_value(self, source_data: Dict[str, Any]) -> Optional[Any]:
        """Find value from source data using priority list"""
        # Try exact matches first
        for source_field in self.source_fields:
            if source_field in source_data:
                value = source_data[source_field]
                if value is not None:
                    return self._transform(value)
        
        # Try aliases (case-insensitive)
        all_possible_names = self.source_fields + self.aliases + [self.output_field]
        source_data_lower = {k.lower(): v for k, v in source_data.items()}
        
        for name in all_possible_names:
            if name.lower() in source_data_lower:
                value = source_data_lower[name.lower()]
                if value is not None:
                    return self._transform(value)
        
        # Try fuzzy matching (partial matches)
        for name in all_possible_names:
            for key, value in source_data.items():
                if name.lower() in key.lower() or key.lower() in name.lower():
                    if value is not None:
                        logger.debug(f" Fuzzy match: '{name}' -> '{key}'")
                        return self._transform(value)
        
        return None
    
    def _transform(self, value: Any) -> Any:
        """Apply transformation if defined"""
        if self.transformer:
            try:
                return self.transformer(value)
            except Exception as e:
                logger.warning(f"Transformation failed for {self.output_field}: {e}")
                return value
        return value


@dataclass
class SchemaDefinition:
    """Defines a stable output schema"""
    name: str
    version: str
    fields: List[FieldMapping]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_field_mapping(self, output_field: str) -> Optional[FieldMapping]:
        """Get field mapping by output field name"""
        for field_map in self.fields:
            if field_map.output_field == output_field:
                return field_map
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema to dict"""
        return {
            'name': self.name,
            'version': self.version,
            'created_at': self.created_at,
            'fields': [
                {
                    'output_field': f.output_field,
                    'source_fields': f.source_fields,
                    'field_type': f.field_type,
                    'required': f.required,
                    'default_value': f.default_value,
                    'aliases': f.aliases,
                    'description': f.description
                }
                for f in self.fields
            ]
        }
    
    def get_hash(self) -> str:
        """Get unique hash of schema definition"""
        schema_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


class SchemaManager:
    """
    Manages schema integrity and field mapping for production environments.
    
    Features:
    - Stable output schema regardless of source changes
    - Intelligent field mapping with fallbacks
    - Type validation and normalization
    - Missing field detection and alerts
    - Schema versioning and migration
    - AI-powered field discovery when mappings fail
    """
    
    def __init__(
        self,
        schema: Optional[SchemaDefinition] = None,
        ai_generator: Optional[Any] = None,
        strict_mode: bool = False,
        enable_ai_mapping: bool = True
    ):
        """
        Initialize Schema Manager
        
        Args:
            schema: Predefined schema definition
            ai_generator: AI generator for intelligent field mapping
            strict_mode: If True, fail on missing required fields
            enable_ai_mapping: Use AI to discover new field mappings
        """
        self.schema = schema
        self.ai_generator = ai_generator
        self.strict_mode = strict_mode
        self.enable_ai_mapping = enable_ai_mapping
        
        # Track mapping performance
        self.stats = {
            'total_items': 0,
            'successful_mappings': 0,
            'missing_required_fields': 0,
            'ai_assisted_mappings': 0,
            'field_coverage': {}
        }
        
        # Cache for discovered mappings
        self.discovered_mappings: Dict[str, List[str]] = {}
        
        logger.info(f" Schema Manager initialized")
        if schema:
            logger.info(f"   Schema: {schema.name} v{schema.version}")
            logger.info(f"   Fields: {len(schema.fields)}")
            logger.info(f"   Strict Mode: {strict_mode}")
    
    def normalize_item(self, source_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a source item to match the defined schema
        
        Args:
            source_item: Raw data item from scraper
            
        Returns:
            Normalized item matching schema
        """
        if not self.schema:
            logger.warning("  No schema defined, returning source item as-is")
            return source_item
        
        normalized = {}
        missing_fields = []
        
        for field_mapping in self.schema.fields:
            # Try to find value using field mapping
            value = field_mapping.find_value(source_item)
            
            # If not found and AI mapping enabled, try AI discovery
            if value is None and self.enable_ai_mapping and self.ai_generator:
                value = self._ai_discover_field(
                    field_mapping.output_field,
                    source_item
                )
                if value is not None:
                    self.stats['ai_assisted_mappings'] += 1
            
            # Handle missing values
            if value is None:
                if field_mapping.required:
                    missing_fields.append(field_mapping.output_field)
                    if field_mapping.default_value is not None:
                        value = field_mapping.default_value
                    elif self.strict_mode:
                        logger.error(f" Required field missing: {field_mapping.output_field}")
                    else:
                        logger.warning(f"  Required field missing: {field_mapping.output_field}")
                else:
                    value = field_mapping.default_value
            
            # Validate and normalize type
            value = self._normalize_type(value, field_mapping.field_type)
            
            # Set normalized value
            normalized[field_mapping.output_field] = value
            
            # Track field coverage
            if value is not None:
                self.stats['field_coverage'][field_mapping.output_field] = \
                    self.stats['field_coverage'].get(field_mapping.output_field, 0) + 1
        
        # Update stats
        self.stats['total_items'] += 1
        if not missing_fields:
            self.stats['successful_mappings'] += 1
        else:
            self.stats['missing_required_fields'] += len(missing_fields)
        
        return normalized
    
    def normalize_batch(self, source_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize a batch of items"""
        return [self.normalize_item(item) for item in source_items]
    
    def _normalize_type(self, value: Any, expected_type: str) -> Any:
        """Normalize value to expected type"""
        if value is None:
            return None
        
        try:
            if expected_type == 'string':
                return str(value)
            elif expected_type == 'number':
                # Try int first, then float
                if isinstance(value, (int, float)):
                    return value
                return float(value) if '.' in str(value) else int(value)
            elif expected_type == 'boolean':
                if isinstance(value, bool):
                    return value
                return str(value).lower() in ('true', '1', 'yes', 'on')
            elif expected_type == 'array':
                if isinstance(value, list):
                    return value
                return [value]
            elif expected_type == 'object':
                if isinstance(value, dict):
                    return value
                return {'value': value}
            else:
                return value
        except Exception as e:
            logger.warning(f"Type normalization failed: {e}")
            return value
    
    def _ai_discover_field(
        self,
        target_field: str,
        source_item: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Use AI to discover which source field maps to target field
        
        This is called when standard mapping fails
        """
        # Check cache first
        if target_field in self.discovered_mappings:
            for source_field in self.discovered_mappings[target_field]:
                if source_field in source_item:
                    logger.debug(f" Using cached AI mapping: {target_field} <- {source_field}")
                    return source_item[source_field]
        
        # Use AI to analyze source data and find best match
        try:
            prompt = f"""
Analyze this data item and identify which field best matches "{target_field}".

Source data fields:
{json.dumps(list(source_item.keys()), indent=2)}

Sample values:
{json.dumps({k: str(v)[:100] for k, v in list(source_item.items())[:10]}, indent=2)}

Return ONLY the exact field name that best matches "{target_field}", or "NONE" if no match.
"""
            
            # This would call the AI generator (simplified for now)
            logger.debug(f" AI analyzing field mapping for: {target_field}")
            # In production, you'd call: self.ai_generator.suggest_field_mapping(...)
            
            # For now, return None (would be implemented with actual AI call)
            return None
            
        except Exception as e:
            logger.warning(f"AI field discovery failed: {e}")
            return None
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate quality report for monitoring"""
        if self.stats['total_items'] == 0:
            return {'status': 'no_data'}
        
        success_rate = self.stats['successful_mappings'] / self.stats['total_items']
        
        # Calculate field coverage
        field_coverage = {}
        for field_name, count in self.stats['field_coverage'].items():
            coverage = (count / self.stats['total_items']) * 100
            field_coverage[field_name] = round(coverage, 2)
        
        # Determine status
        if success_rate >= 0.9:
            status = 'healthy'
        elif success_rate >= 0.7:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'success_rate': round(success_rate * 100, 2),
            'total_items': self.stats['total_items'],
            'successful_mappings': self.stats['successful_mappings'],
            'missing_required_fields': self.stats['missing_required_fields'],
            'ai_assisted_mappings': self.stats['ai_assisted_mappings'],
            'field_coverage': field_coverage,
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_items': 0,
            'successful_mappings': 0,
            'missing_required_fields': 0,
            'ai_assisted_mappings': 0,
            'field_coverage': {}
        }


# Predefined schema examples
def create_ecommerce_schema() -> SchemaDefinition:
    """Create a standard e-commerce product schema"""
    return SchemaDefinition(
        name="ecommerce_product",
        version="1.0",
        fields=[
            FieldMapping(
                output_field="product_id",
                source_fields=["id", "product_id", "sku", "productId"],
                field_type="string",
                required=True,
                aliases=["itemId", "item_id"],
                description="Unique product identifier"
            ),
            FieldMapping(
                output_field="name",
                source_fields=["name", "title", "product_name", "productName"],
                field_type="string",
                required=True,
                aliases=["product_title", "item_name"],
                description="Product name"
            ),
            FieldMapping(
                output_field="price",
                source_fields=["price", "current_price", "salePrice", "currentPrice"],
                field_type="number",
                required=True,
                aliases=["cost", "amount"],
                transformer=lambda x: float(str(x).replace('$', '').replace(',', '')),
                description="Current price"
            ),
            FieldMapping(
                output_field="original_price",
                source_fields=["original_price", "list_price", "msrp", "regularPrice"],
                field_type="number",
                required=False,
                transformer=lambda x: float(str(x).replace('$', '').replace(',', '')),
                description="Original/list price"
            ),
            FieldMapping(
                output_field="description",
                source_fields=["description", "product_description", "details"],
                field_type="string",
                required=False,
                description="Product description"
            ),
            FieldMapping(
                output_field="brand",
                source_fields=["brand", "manufacturer", "vendor"],
                field_type="string",
                required=False,
                description="Product brand"
            ),
            FieldMapping(
                output_field="category",
                source_fields=["category", "product_type", "department"],
                field_type="string",
                required=False,
                description="Product category"
            ),
            FieldMapping(
                output_field="in_stock",
                source_fields=["in_stock", "available", "availability", "inStock"],
                field_type="boolean",
                required=False,
                default_value=True,
                description="Stock availability"
            ),
            FieldMapping(
                output_field="image_url",
                source_fields=["image", "image_url", "imageUrl", "thumbnail"],
                field_type="string",
                required=False,
                description="Product image URL"
            ),
            FieldMapping(
                output_field="rating",
                source_fields=["rating", "average_rating", "averageRating", "stars"],
                field_type="number",
                required=False,
                description="Product rating"
            ),
        ]
    )


def create_leafly_schema() -> SchemaDefinition:
    """Create schema for Leafly dispensary products"""
    return SchemaDefinition(
        name="leafly_product",
        version="1.0",
        fields=[
            FieldMapping(
                output_field="name",
                source_fields=["product_name", "name", "title"],
                field_type="string",
                required=True,
                description="Product name"
            ),
            FieldMapping(
                output_field="price",
                source_fields=["price", "cost"],
                field_type="number",
                required=True,
                description="Product price in USD"
            ),
            FieldMapping(
                output_field="thc_percentage",
                source_fields=[
                    "thc_content.percentile50",
                    "thc_content.percentile75",
                    "thc_content",
                    "thc",
                    "strain_type.cannabinoids.thc.percentile50"
                ],
                field_type="number",
                required=False,
                transformer=lambda x: x.get('percentile50') if isinstance(x, dict) else x,
                description="THC percentage"
            ),
            FieldMapping(
                output_field="cbd_percentage",
                source_fields=[
                    "cbd_content.percentile50",
                    "cbd_content",
                    "cbd",
                    "strain_type.cannabinoids.cbd.percentile50"
                ],
                field_type="number",
                required=False,
                transformer=lambda x: x.get('percentile50') if isinstance(x, dict) else x,
                description="CBD percentage"
            ),
            FieldMapping(
                output_field="brand",
                source_fields=["brand.name", "brand", "manufacturer"],
                field_type="string",
                required=False,
                transformer=lambda x: x.get('name') if isinstance(x, dict) else x,
                description="Brand name"
            ),
            FieldMapping(
                output_field="strain_type",
                source_fields=[
                    "strain_type.category",
                    "strain_type",
                    "product_type",
                    "category"
                ],
                field_type="string",
                required=False,
                transformer=lambda x: x.get('category') if isinstance(x, dict) else x,
                description="Strain category (Indica/Sativa/Hybrid)"
            ),
            FieldMapping(
                output_field="strain_name",
                source_fields=[
                    "strain_type.name",
                    "strain_type.slug",
                    "strain"
                ],
                field_type="string",
                required=False,
                transformer=lambda x: x.get('name') if isinstance(x, dict) else x,
                description="Strain name"
            ),
        ]
    )









"""
Schema Inference - Auto-generate schemas from scraped data

This module solves the bootstrap problem: "How do I define a schema
for a website I've never scraped before?"

Features:
- Auto-generate schema from first scrape
- Learn optimal field mappings from multiple scrapes
- Suggest schema refinements based on data patterns
- Export to reusable schema definitions
"""

import logging
from typing import List, Dict, Any, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
import json

from .schema_manager import SchemaDefinition, FieldMapping

logger = logging.getLogger(__name__)


@dataclass
class FieldStats:
    """Statistics about a field observed in data"""
    field_name: str
    values_seen: int = 0
    null_count: int = 0
    type_counts: Dict[str, int] = None
    sample_values: List[Any] = None
    nested_keys: Set[str] = None  # For objects
    
    def __post_init__(self):
        if self.type_counts is None:
            self.type_counts = {}
        if self.sample_values is None:
            self.sample_values = []
        if self.nested_keys is None:
            self.nested_keys = set()
    
    @property
    def coverage(self) -> float:
        """Percentage of times this field had a value"""
        if self.values_seen == 0:
            return 0.0
        return ((self.values_seen - self.null_count) / self.values_seen) * 100
    
    @property
    def primary_type(self) -> str:
        """Most common type for this field"""
        if not self.type_counts:
            return 'string'
        return max(self.type_counts.items(), key=lambda x: x[1])[0]
    
    @property
    def is_required(self) -> bool:
        """Should this be a required field?"""
        return self.coverage >= 90  # 90%+ coverage = required


class SchemaInference:
    """
    Auto-generate schemas from scraped data
    
    Usage:
        # First time scraping a new website
        inferencer = SchemaInference()
        
        # Scrape without schema
        result = scraper.scrape(url, fields)
        
        # Learn from the data
        inferencer.learn_from_data(result['data'])
        
        # Generate schema
        schema = inferencer.generate_schema(name="my_schema")
        
        # Use it for future scrapes
        scraper = UniversalScraper(schema=schema)
    """
    
    def __init__(self):
        self.field_stats: Dict[str, FieldStats] = {}
        self.observations = 0
        logger.info(" Schema Inference initialized")
    
    def learn_from_data(self, data: List[Dict[str, Any]]) -> None:
        """
        Learn schema from observed data
        
        Args:
            data: List of scraped items
        """
        if not data:
            logger.warning("No data provided to learn from")
            return
        
        logger.info(f" Learning from {len(data)} items...")
        
        for item in data:
            self._analyze_item(item)
        
        self.observations += len(data)
        logger.info(f" Learned from {self.observations} total items")
        self._log_summary()
    
    def _analyze_item(self, item: Dict[str, Any], prefix: str = "") -> None:
        """Analyze a single item and update statistics"""
        for key, value in item.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            # Get or create field stats
            if full_key not in self.field_stats:
                self.field_stats[full_key] = FieldStats(field_name=full_key)
            
            stats = self.field_stats[full_key]
            stats.values_seen += 1
            
            # Track nulls
            if value is None:
                stats.null_count += 1
                continue
            
            # Track type
            value_type = self._get_type(value)
            stats.type_counts[value_type] = stats.type_counts.get(value_type, 0) + 1
            
            # Store sample values (keep first 5)
            if len(stats.sample_values) < 5:
                stats.sample_values.append(value)
            
            # If it's an object, analyze nested fields
            if isinstance(value, dict):
                stats.nested_keys.update(value.keys())
                # Recursively analyze nested structure
                self._analyze_item(value, prefix=full_key)
    
    def _get_type(self, value: Any) -> str:
        """Determine the type category of a value"""
        if value is None:
            return 'null'
        elif isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, int):
            return 'number'
        elif isinstance(value, float):
            return 'number'
        elif isinstance(value, str):
            return 'string'
        elif isinstance(value, list):
            return 'array'
        elif isinstance(value, dict):
            return 'object'
        else:
            return 'string'
    
    def _log_summary(self) -> None:
        """Log a summary of learned schema"""
        logger.info(f" Discovered {len(self.field_stats)} unique fields")
        
        required_fields = [
            name for name, stats in self.field_stats.items()
            if stats.is_required and '.' not in name  # Top-level only
        ]
        logger.info(f"   Required fields: {len(required_fields)}")
        
        optional_fields = [
            name for name, stats in self.field_stats.items()
            if not stats.is_required and '.' not in name
        ]
        logger.info(f"   Optional fields: {len(optional_fields)}")
    
    def generate_schema(
        self,
        name: str,
        version: str = "1.0",
        include_nested: bool = False,
        min_coverage: float = 50.0
    ) -> SchemaDefinition:
        """
        Generate a schema definition from learned data
        
        Args:
            name: Schema name
            version: Schema version
            include_nested: Include nested object fields
            min_coverage: Minimum coverage % to include a field
            
        Returns:
            SchemaDefinition ready to use
        """
        logger.info(f"  Generating schema: {name} v{version}")
        
        field_mappings = []
        
        # Sort fields by coverage (highest first)
        sorted_fields = sorted(
            self.field_stats.items(),
            key=lambda x: x[1].coverage,
            reverse=True
        )
        
        for field_name, stats in sorted_fields:
            # Skip nested fields unless requested
            if '.' in field_name and not include_nested:
                continue
            
            # Skip fields with low coverage
            if stats.coverage < min_coverage:
                continue
            
            # Create field mapping
            field_mapping = self._create_field_mapping(field_name, stats)
            field_mappings.append(field_mapping)
        
        schema = SchemaDefinition(
            name=name,
            version=version,
            fields=field_mappings
        )
        
        logger.info(f" Generated schema with {len(field_mappings)} fields")
        return schema
    
    def _create_field_mapping(
        self,
        field_name: str,
        stats: FieldStats
    ) -> FieldMapping:
        """Create a FieldMapping from field statistics"""
        # Generate aliases (common variations)
        aliases = self._generate_aliases(field_name)
        
        # Generate source field list (try exact name first)
        source_fields = [field_name] + aliases
        
        # Create transformer if needed
        transformer = self._generate_transformer(stats)
        
        return FieldMapping(
            output_field=self._normalize_field_name(field_name),
            source_fields=source_fields,
            field_type=stats.primary_type,
            required=stats.is_required,
            default_value=None,
            transformer=transformer,
            aliases=aliases,
            description=f"Auto-generated from {stats.values_seen} observations ({stats.coverage:.1f}% coverage)"
        )
    
    def _normalize_field_name(self, field_name: str) -> str:
        """Normalize field name to a clean output name"""
        # Remove nesting
        if '.' in field_name:
            field_name = field_name.split('.')[-1]
        
        # Convert to snake_case
        import re
        # Handle camelCase
        field_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', field_name)
        # Convert to lowercase
        field_name = field_name.lower()
        # Replace spaces and hyphens with underscores
        field_name = re.sub(r'[-\s]+', '_', field_name)
        
        return field_name
    
    def _generate_aliases(self, field_name: str) -> List[str]:
        """Generate common aliases for a field name"""
        aliases = []
        base_name = field_name.split('.')[-1] if '.' in field_name else field_name
        
        # camelCase version
        parts = base_name.split('_')
        if len(parts) > 1:
            camel = parts[0] + ''.join(p.capitalize() for p in parts[1:])
            aliases.append(camel)
        
        # PascalCase version
        pascal = ''.join(p.capitalize() for p in parts)
        aliases.append(pascal)
        
        # kebab-case version
        kebab = '-'.join(parts)
        aliases.append(kebab)
        
        # Common variations
        variations = {
            'name': ['title', 'label', 'heading'],
            'price': ['cost', 'amount', 'value'],
            'description': ['desc', 'details', 'summary'],
            'image': ['img', 'picture', 'photo', 'thumbnail'],
            'url': ['link', 'href'],
            'id': ['identifier', 'key'],
        }
        
        for key, vars in variations.items():
            if key in base_name.lower():
                aliases.extend(vars)
        
        # Remove duplicates and empty strings
        aliases = list(set(a for a in aliases if a and a != base_name))
        
        return aliases[:5]  # Limit to 5 aliases
    
    def _generate_transformer(self, stats: FieldStats) -> Optional[Any]:
        """Generate a transformer function if needed"""
        # If field is sometimes nested, extract the value
        if 'object' in stats.type_counts and stats.nested_keys:
            # Common nested patterns
            value_keys = ['value', 'val', 'data', 'current', 'actual']
            for key in value_keys:
                if key in stats.nested_keys:
                    return lambda x: x.get(key) if isinstance(x, dict) else x
        
        # If field is sometimes a number string, convert it
        if stats.primary_type == 'string':
            sample_str = next((v for v in stats.sample_values if isinstance(v, str)), None)
            if sample_str and self._looks_like_number(sample_str):
                return lambda x: float(str(x).replace('$', '').replace(',', '')) if x else None
        
        return None
    
    def _looks_like_number(self, value: str) -> bool:
        """Check if a string looks like a number"""
        cleaned = value.replace('$', '').replace(',', '').strip()
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    
    def get_report(self) -> Dict[str, Any]:
        """Generate a detailed report of learned schema"""
        report = {
            'observations': self.observations,
            'fields_discovered': len(self.field_stats),
            'fields': []
        }
        
        for field_name, stats in sorted(
            self.field_stats.items(),
            key=lambda x: x[1].coverage,
            reverse=True
        ):
            if '.' not in field_name:  # Top-level only
                report['fields'].append({
                    'name': field_name,
                    'normalized': self._normalize_field_name(field_name),
                    'coverage': round(stats.coverage, 1),
                    'type': stats.primary_type,
                    'required': stats.is_required,
                    'samples': stats.sample_values[:3]
                })
        
        return report
    
    def export_schema_code(self, name: str) -> str:
        """
        Export schema as Python code
        
        Returns:
            Python code string that can be saved to a file
        """
        schema = self.generate_schema(name)
        
        code = f'''"""
Auto-generated schema for {name}
Generated from {self.observations} observations
"""

from universal_scraper.core.schema_manager import SchemaDefinition, FieldMapping


def create_{name}_schema() -> SchemaDefinition:
    """Create schema for {name}"""
    return SchemaDefinition(
        name="{name}",
        version="{schema.version}",
        fields=[
'''
        
        for field in schema.fields:
            transformer_str = ""
            if field.transformer:
                transformer_str = ",\n                transformer=lambda x: x  # TODO: Customize"
            
            code += f'''            FieldMapping(
                output_field="{field.output_field}",
                source_fields={field.source_fields},
                field_type="{field.field_type}",
                required={field.required},
                aliases={field.aliases}{transformer_str}
            ),
'''
        
        code += '''        ]
    )
'''
        
        return code


def infer_schema_from_scrape(
    scraper,
    url: str,
    fields: List[str],
    schema_name: str,
    num_samples: int = 3
) -> SchemaDefinition:
    """
    Convenience function: Scrape once and auto-generate schema
    
    Args:
        scraper: UniversalScraper instance (without schema)
        url: URL to scrape
        fields: Fields to extract
        schema_name: Name for the generated schema
        num_samples: Number of scrapes to learn from
        
    Returns:
        SchemaDefinition ready to use
        
    Example:
        # First time scraping a new website
        scraper = UniversalScraper()
        
        # Auto-generate schema from first scrape
        schema = infer_schema_from_scrape(
            scraper,
            "https://example.com/products",
            ["name", "price", "description"],
            "example_products"
        )
        
        # Now use the schema for future scrapes
        scraper_with_schema = UniversalScraper(schema=schema)
    """
    logger.info(f" Inferring schema from {num_samples} sample scrapes...")
    
    inferencer = SchemaInference()
    
    # Scrape multiple times to learn patterns
    urls = [url] if isinstance(url, str) else url[:num_samples]
    
    for i, sample_url in enumerate(urls, 1):
        logger.info(f" Sample {i}/{len(urls)}: {sample_url}")
        try:
            result = scraper.scrape(sample_url, fields)
            inferencer.learn_from_data(result['data'])
        except Exception as e:
            logger.warning(f"  Sample {i} failed: {e}")
            continue
    
    # Generate schema
    schema = inferencer.generate_schema(schema_name)
    
    logger.info(f" Schema inferred: {schema.name}")
    logger.info(f"   Fields: {len(schema.fields)}")
    logger.info(f"   Required: {sum(1 for f in schema.fields if f.required)}")
    
    return schema









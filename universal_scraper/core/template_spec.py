"""
Template Spec - Deterministic Template Specification
LLM outputs JSON spec, runtime extractor is deterministic

This ensures reproducible extraction results even with LLM involvement.
The LLM generates a "template spec" (JSON), and the runtime extractor
executes it deterministically (no LLM calls during extraction).
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)


class SelectorType(Enum):
    """Selector types"""
    CSS = "css"
    XPATH = "xpath"
    TEXT = "text"  # Text-based matching
    ATTRIBUTE = "attribute"  # Data attribute


class NormalizerType(Enum):
    """Normalizer types"""
    PARSE_CURRENCY = "parse_currency"
    PARSE_DATE = "parse_date"
    PARSE_NUMBER = "parse_number"
    STRIP_WHITESPACE = "strip_whitespace"
    REMOVE_HTML = "remove_html"
    EXTRACT_TEXT = "extract_text"


@dataclass
class FieldSelector:
    """Selector configuration for a field"""
    field_name: str
    primary: str  # Primary selector (CSS/XPath)
    fallbacks: List[str] = field(default_factory=list)  # Fallback selectors
    selector_type: SelectorType = SelectorType.CSS
    priority: int = 1  # Priority (1 = highest)
    normalizer: Optional[NormalizerType] = None
    validator: Optional[Dict[str, Any]] = None  # Validation rules
    required: bool = False
    default_value: Any = None


@dataclass
class PaginationConfig:
    """Pagination configuration"""
    type: str  # 'url_param', 'next_button', 'infinite_scroll', etc.
    param: Optional[str] = None  # URL parameter name (for url_param)
    next_selector: Optional[str] = None  # Selector for next button
    max_pages: Optional[int] = None  # Maximum pages to scrape


@dataclass
class TemplateSpec:
    """
    Template specification for deterministic extraction
    
    This is what the LLM generates (JSON), and the runtime extractor executes.
    """
    template_id: str
    page_fingerprint_features: Dict[str, Any]  # What LLM saw
    selectors: List[FieldSelector]  # Selectors per field
    pagination: Optional[PaginationConfig] = None
    confidence: float = 0.0  # LLM confidence (0.0-1.0)
    why_these_selectors: str = ""  # LLM reasoning (for debugging)
    version: int = 1
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)"""
        return {
            'template_id': self.template_id,
            'page_fingerprint_features': self.page_fingerprint_features,
            'selectors': [
                {
                    'field_name': s.field_name,
                    'primary': s.primary,
                    'fallbacks': s.fallbacks,
                    'selector_type': s.selector_type.value,
                    'priority': s.priority,
                    'normalizer': s.normalizer.value if s.normalizer else None,
                    'validator': s.validator,
                    'required': s.required,
                    'default_value': s.default_value
                }
                for s in self.selectors
            ],
            'pagination': asdict(self.pagination) if self.pagination else None,
            'confidence': self.confidence,
            'why_these_selectors': self.why_these_selectors,
            'version': self.version,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateSpec':
        """Create from dictionary"""
        selectors = [
            FieldSelector(
                field_name=s['field_name'],
                primary=s['primary'],
                fallbacks=s.get('fallbacks', []),
                selector_type=SelectorType(s.get('selector_type', 'css')),
                priority=s.get('priority', 1),
                normalizer=NormalizerType(s['normalizer']) if s.get('normalizer') else None,
                validator=s.get('validator'),
                required=s.get('required', False),
                default_value=s.get('default_value')
            )
            for s in data.get('selectors', [])
        ]
        
        pagination = None
        if data.get('pagination'):
            pagination = PaginationConfig(**data['pagination'])
        
        return cls(
            template_id=data['template_id'],
            page_fingerprint_features=data.get('page_fingerprint_features', {}),
            selectors=selectors,
            pagination=pagination,
            confidence=data.get('confidence', 0.0),
            why_these_selectors=data.get('why_these_selectors', ''),
            version=data.get('version', 1),
            created_at=data.get('created_at')
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateSpec':
        """Deserialize from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate template spec
        
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        if not self.template_id:
            errors.append("template_id is required")
        
        if not self.selectors:
            errors.append("at least one selector is required")
        
        for selector in self.selectors:
            if not selector.primary:
                errors.append(f"selector for {selector.field_name} missing primary selector")
        
        return (len(errors) == 0, errors)




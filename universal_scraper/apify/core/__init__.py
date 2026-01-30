"""Core scraping modules"""

from .scraper import UniversalScraper
from .html_fetcher import HTMLFetcher
from .hybrid_fetcher import HybridFetcher
from .html_cleaner import SmartHTMLCleaner
from .json_detector import JSONDetector
from .structural_hash import StructuralHashGenerator
from .code_cache import CodeCache
from .ai_generator import AICodeGenerator
from .api_cache import APICache
from .schema_manager import (
 SchemaManager,
 SchemaDefinition,
 FieldMapping,
 create_ecommerce_schema,
 create_leafly_schema
)
from .schema_inference import (
 SchemaInference,
 infer_schema_from_scrape
)
from .context_manager import ContextManager, ExtractionContext
from .data_validator import LLMDataValidator
from .json_analyzer import LLMJsonAnalyzer

# Browser fetcher is optional (requires Camoufox)
try:
 from .browser_fetcher import BrowserFetcher
 BROWSER_AVAILABLE = True
except ImportError:
 BROWSER_AVAILABLE = False

__all__ = [
 "UniversalScraper",
 "HTMLFetcher",
 "HybridFetcher",
 "SmartHTMLCleaner",
 "JSONDetector",
 "StructuralHashGenerator",
 "CodeCache",
 "AICodeGenerator",
 "APICache",
 "SchemaManager",
 "SchemaDefinition",
 "FieldMapping",
 "create_ecommerce_schema",
 "create_leafly_schema",
 "SchemaInference",
 "infer_schema_from_scrape",
 "ContextManager",
 "ExtractionContext",
 "LLMDataValidator",
 "LLMJsonAnalyzer"
]

if BROWSER_AVAILABLE:
 __all__.append("BrowserFetcher")


"""Core scraping modules"""

from .scraper import UniversalScraper
from .html_fetcher import HTMLFetcher
from .html_cleaner import SmartHTMLCleaner
from .json_detector import JSONDetector
from .structural_hash import StructuralHashGenerator
from .code_cache import CodeCache
from .ai_generator import AICodeGenerator

__all__ = [
    "UniversalScraper",
    "HTMLFetcher",
    "SmartHTMLCleaner",
    "JSONDetector",
    "StructuralHashGenerator",
    "CodeCache",
    "AICodeGenerator"
]


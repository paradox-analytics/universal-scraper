"""
Universal Scraper - Main orchestration class
Coordinates all components following the architecture diagram
"""

import logging
import time
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

from .html_fetcher import HTMLFetcher
from .json_detector import JSONDetector
from .html_cleaner import SmartHTMLCleaner
from .structural_hash import StructuralHashGenerator
from .code_cache import CodeCache
from .ai_generator import AICodeGenerator

logger = logging.getLogger(__name__)


class UniversalScraper:
    """
    Universal Web Scraper with JSON-first architecture
    
    Architecture Flow:
    1. Fetch HTML (with CloudScraper + proxies)
    2. Detect JSON sources (priority)
    3. Clean HTML (98% reduction)
    4. Generate structural hash
    5. Check code cache
    6. Generate extraction code (if cache miss)
    7. Execute code and extract data
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        proxy_config: Optional[Dict[str, str]] = None,
        cache_dir: str = "./cache",
        cache_ttl: int = 86400,
        enable_cache: bool = True,
        enable_warming: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Initialize Universal Scraper
        
        Args:
            api_key: API key for AI provider
            model_name: AI model to use (auto-detected if None)
            proxy_config: Proxy configuration dict
            cache_dir: Directory for code cache
            cache_ttl: Cache TTL in seconds
            enable_cache: Enable/disable caching
            enable_warming: Enable session warming
            log_level: Logging level
        """
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize components
        self.html_fetcher = HTMLFetcher(
            proxy_config=proxy_config,
            enable_warming=enable_warming
        )
        
        self.json_detector = JSONDetector()
        self.html_cleaner = SmartHTMLCleaner()
        self.hash_generator = StructuralHashGenerator()
        
        self.code_cache = CodeCache(
            cache_dir=cache_dir,
            ttl=cache_ttl,
            enable_cache=enable_cache
        )
        
        self.ai_generator = AICodeGenerator(
            api_key=api_key,
            model_name=model_name
        )
        
        logger.info("🚀 Universal Scraper initialized")
        logger.info(f"   AI Model: {self.ai_generator.model_name}")
        logger.info(f"   Cache: {'Enabled' if enable_cache else 'Disabled'}")
        logger.info(f"   Proxy: {'Enabled' if proxy_config else 'Disabled'}")
    
    def scrape(
        self,
        url: str,
        fields: List[str],
        force_html: bool = False,
        force_generate: bool = False
    ) -> Dict[str, Any]:
        """
        Scrape data from URL
        
        Args:
            url: Target URL
            fields: Fields to extract
            force_html: Skip JSON detection, use HTML parsing
            force_generate: Skip cache, generate new extraction code
            
        Returns:
            Dict with 'data', 'metadata', 'source' keys
        """
        start_time = time.time()
        
        logger.info(f"🎯 Scraping: {url}")
        logger.info(f"📋 Fields: {', '.join(fields)}")
        
        # Step 1: Fetch HTML
        logger.info("📥 Step 1: Fetching HTML...")
        fetch_result = self.html_fetcher.fetch(url)
        html = fetch_result['html']
        
        # Step 2: Try JSON detection first (unless forced to use HTML)
        json_data = []
        extraction_source = 'html'
        
        if not force_html:
            logger.info("🔍 Step 2: Detecting JSON sources...")
            json_results = self.json_detector.detect_and_extract(html, url)
            
            if json_results['json_found']:
                # Check if JSON is sufficient
                if self.json_detector.is_json_sufficient(json_results, fields):
                    logger.info("✅ JSON sources sufficient, extracting from JSON...")
                    json_data = self.json_detector.extract_from_json(
                        json_results['data'],
                        fields
                    )
                    extraction_source = 'json'
                else:
                    logger.info("⚠️ JSON sources insufficient, falling back to HTML...")
        
        # Step 3: If JSON not sufficient, use HTML extraction
        if not json_data:
            logger.info("🧹 Step 3: Cleaning HTML...")
            clean_result = self.html_cleaner.clean(html)
            cleaned_html = clean_result['html']
            
            logger.info(f"   Reduction: {clean_result['reduction_percent']:.1f}%")
            
            # Step 4: Generate structural hash
            logger.info("🔑 Step 4: Generating structural hash...")
            hash_result = self.hash_generator.generate_hash(cleaned_html)
            structure_hash = hash_result['hash']
            
            # Step 5: Check cache (unless forced to generate)
            extraction_code = None
            code_cached = False
            
            if not force_generate:
                logger.info("💾 Step 5: Checking code cache...")
                cached_entry = self.code_cache.get(structure_hash)
                
                if cached_entry:
                    extraction_code = cached_entry['code']
                    code_cached = True
                    logger.info("✅ Using cached extraction code")
            
            # Step 6: Generate code if needed
            if not extraction_code:
                logger.info("🤖 Step 6: Generating extraction code with AI...")
                gen_result = self.ai_generator.generate_extraction_code(
                    cleaned_html,
                    fields,
                    url
                )
                extraction_code = gen_result['code']
                
                # Cache the generated code
                self.code_cache.set(
                    structure_hash,
                    extraction_code,
                    {
                        'model': gen_result['model_used'],
                        'fields': fields,
                        'url': url
                    }
                )
            
            # Step 7: Execute extraction code
            logger.info("⚡ Step 7: Executing extraction code...")
            json_data = self._execute_extraction_code(
                extraction_code,
                html  # Use original HTML for extraction
            )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        logger.info(f"✅ Extraction complete: {len(json_data)} items in {execution_time:.2f}s")
        
        return {
            'data': json_data,
            'metadata': {
                'url': url,
                'fields': fields,
                'items_extracted': len(json_data),
                'execution_time': execution_time,
                'extraction_source': extraction_source,
                'code_cached': code_cached if extraction_source == 'html' else None,
                'timestamp': time.time()
            },
            'source': extraction_source
        }
    
    def scrape_multiple(
        self,
        urls: List[str],
        fields: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs
        
        Args:
            urls: List of URLs to scrape
            fields: Fields to extract
            **kwargs: Additional arguments passed to scrape()
            
        Returns:
            List of scrape results
        """
        logger.info(f"🎯 Scraping {len(urls)} URLs...")
        
        results = []
        for i, url in enumerate(urls, 1):
            logger.info(f"📍 Processing {i}/{len(urls)}: {url}")
            
            try:
                result = self.scrape(url, fields, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Failed to scrape {url}: {str(e)}")
                results.append({
                    'data': [],
                    'metadata': {
                        'url': url,
                        'error': str(e)
                    },
                    'source': 'error'
                })
        
        logger.info(f"✅ Batch complete: {len(results)} results")
        return results
    
    def _execute_extraction_code(
        self,
        code: str,
        html: str
    ) -> List[Dict[str, Any]]:
        """
        Execute generated extraction code
        
        Args:
            code: Python extraction code
            html: HTML to extract from
            
        Returns:
            List of extracted items
        """
        try:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Create execution namespace
            namespace = {
                'soup': soup,
                'BeautifulSoup': BeautifulSoup,
            }
            
            # Execute the code
            exec(code, namespace)
            
            # Call the extract_data function
            if 'extract_data' in namespace:
                data = namespace['extract_data'](soup)
                
                if not isinstance(data, list):
                    logger.warning("⚠️ extract_data didn't return a list, wrapping...")
                    data = [data] if data else []
                
                return data
            else:
                logger.error("❌ Generated code doesn't have extract_data function")
                return []
                
        except Exception as e:
            logger.error(f"❌ Code execution failed: {str(e)}")
            logger.debug(f"Code was:\n{code}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.code_cache.get_stats()
    
    def clear_cache(self) -> bool:
        """Clear code cache"""
        return self.code_cache.clear()
    
    def export_cache(self, export_path: str) -> bool:
        """Export cache to file"""
        return self.code_cache.export_cache(export_path)
    
    def import_cache(self, import_path: str) -> bool:
        """Import cache from file"""
        return self.code_cache.import_cache(import_path)
    
    def close(self) -> None:
        """Clean up resources"""
        self.html_fetcher.close()
        logger.info("👋 Universal Scraper closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function for simple usage
def scrape(
    url: str,
    fields: List[str],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    proxy_config: Optional[Dict[str, str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for simple scraping
    
    Args:
        url: Target URL
        fields: Fields to extract
        api_key: AI API key
        model_name: AI model name
        proxy_config: Proxy configuration
        **kwargs: Additional arguments
        
    Returns:
        Scrape result dict
    """
    with UniversalScraper(
        api_key=api_key,
        model_name=model_name,
        proxy_config=proxy_config,
        **kwargs
    ) as scraper:
        return scraper.scrape(url, fields)


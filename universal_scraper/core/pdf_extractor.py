"""
PDF Extractor
Extracts structured data from PDF documents using LLM
"""

import io
import logging
import hashlib
import requests
from typing import List, Dict, Any, Optional
import json

try:
    import pymupdf  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

import litellm

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extract structured data from PDFs using LLM
    
    Two-phase approach:
    1. Fast detection: Check if PDF has tables/forms (use pdfplumber)
    2. LLM extraction: Use LLM for flexible extraction
    
    Similar to DirectLLMExtractor but for PDFs
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        max_pages: Optional[int] = None,  # Limit pages to process (None = all)
        use_ocr: bool = False,  # Use OCR for scanned PDFs (requires pytesseract)
        enable_cache: bool = True,
        timeout: int = 30
    ):
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError(
                "PyMuPDF (pymupdf) is required for PDF extraction. "
                "Install with: pip install pymupdf"
            )
        
        self.api_key = api_key
        self.model_name = model_name
        self.max_pages = max_pages
        self.use_ocr = use_ocr
        self.enable_cache = enable_cache
        self.timeout = timeout
        
        logger.info(f" PDF Extractor initialized (model={model_name}, max_pages={max_pages or 'all'})")
    
    async def extract(
        self,
        pdf_url: str,
        fields: List[str],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract structured data from PDF
        
        Args:
            pdf_url: URL to PDF document
            fields: List of fields to extract
            context: Optional context about what to extract
            
        Returns:
            List of extracted items
        """
        logger.info(f" Extracting from PDF: {pdf_url}")
        logger.info(f"   Fields: {', '.join(fields)}")
        
        # Download PDF
        pdf_bytes = await self._download_pdf(pdf_url)
        
        # Extract text from PDF
        text, metadata = self._pdf_to_text(pdf_bytes)
        
        logger.info(f"   Extracted {len(text)} chars from {metadata['page_count']} pages")
        
        # Check if PDF has tables
        has_tables = metadata.get('has_tables', False)
        if has_tables and PDFPLUMBER_AVAILABLE:
            logger.info("    Tables detected - extracting table data")
            table_data = self._extract_tables(pdf_bytes)
            if table_data:
                logger.info(f"   Found {len(table_data)} tables")
                # Include table data in text for LLM
                text += "\n\n=== EXTRACTED TABLES ===\n" + table_data
        
        # Use LLM to extract structured data
        items = await self._extract_with_llm(text, fields, context, pdf_url)
        
        logger.info(f"    Extracted {len(items)} items from PDF")
        
        return items
    
    async def _download_pdf(self, pdf_url: str) -> bytes:
        """Download PDF from URL or read from local file"""
        try:
            # Handle local file:// URLs
            if pdf_url.startswith('file://'):
                file_path = pdf_url.replace('file://', '')
                logger.info(f" Reading local PDF: {file_path}")
                with open(file_path, 'rb') as f:
                    return f.read()
            
            # Handle regular URLs
            response = requests.get(
                pdf_url,
                timeout=self.timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            response.raise_for_status()
            
            return response.content
        except Exception as e:
            logger.error(f" Failed to download PDF: {e}")
            raise
    
    def _pdf_to_text(self, pdf_bytes: bytes) -> tuple[str, Dict[str, Any]]:
        """
        Convert PDF to text preserving layout
        
        Returns:
            Tuple of (text, metadata)
        """
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            
            text_chunks = []
            has_tables = False
            total_pages = len(doc)
            
            # Limit pages if specified
            pages_to_process = min(total_pages, self.max_pages) if self.max_pages else total_pages
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                
                # Extract text with layout preserved
                page_text = page.get_text("text")
                
                # Check for tables (simple heuristic)
                if self._has_table_structure(page_text):
                    has_tables = True
                
                # Add page marker
                text_chunks.append(f"\n\n=== Page {page_num + 1} ===\n{page_text}")
            
            doc.close()
            
            full_text = "".join(text_chunks)
            
            metadata = {
                'page_count': total_pages,
                'pages_processed': pages_to_process,
                'has_tables': has_tables,
                'char_count': len(full_text)
            }
            
            return full_text, metadata
            
        except Exception as e:
            logger.error(f" Failed to extract text from PDF: {e}")
            raise
    
    def _has_table_structure(self, text: str) -> bool:
        """
        Heuristic to detect if text has table-like structure
        """
        lines = text.split('\n')
        if len(lines) < 3:
            return False
        
        # Check if multiple lines have similar structure (whitespace patterns)
        # This is a simple heuristic - tables often have aligned columns
        tab_counts = [line.count('\t') for line in lines if line.strip()]
        
        # If many lines have same tab count, likely a table
        if tab_counts and max(set(tab_counts), key=tab_counts.count) >= 2:
            return True
        
        return False
    
    def _extract_tables(self, pdf_bytes: bytes) -> str:
        """
        Extract tables from PDF using pdfplumber
        """
        if not PDFPLUMBER_AVAILABLE:
            return ""
        
        try:
            table_texts = []
            
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages_to_process = min(len(pdf.pages), self.max_pages) if self.max_pages else len(pdf.pages)
                
                for page_num in range(pages_to_process):
                    page = pdf.pages[page_num]
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if table:
                            # Convert table to text format
                            table_text = f"\n--- Table {page_num + 1}.{table_idx + 1} ---\n"
                            for row in table:
                                # Clean and join cells
                                cells = [str(cell or '').strip() for cell in row]
                                table_text += " | ".join(cells) + "\n"
                            
                            table_texts.append(table_text)
            
            return "\n".join(table_texts)
            
        except Exception as e:
            logger.warning(f"  Failed to extract tables: {e}")
            return ""
    
    async def _extract_with_llm(
        self,
        text: str,
        fields: List[str],
        context: Optional[str],
        pdf_url: str
    ) -> List[Dict[str, Any]]:
        """
        Extract structured data using LLM (similar to DirectLLMExtractor)
        """
        # Truncate text if too long (avoid token limits)
        max_chars = 50000  # ~12,500 tokens for gpt-4o-mini
        if len(text) > max_chars:
            logger.warning(f"     PDF text too long ({len(text)} chars), truncating to {max_chars}")
            text = text[:max_chars] + "\n\n... [truncated] ..."
        
        # Build extraction prompt
        prompt = self._build_extraction_prompt(text, fields, context)
        
        # Call LLM
        try:
            response = await litellm.acompletion(
                model=self.model_name,
                api_key=self.api_key,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
                items = result.get('items', [])
                
                if not items and isinstance(result, list):
                    items = result
                
                return items
                
            except json.JSONDecodeError as e:
                logger.error(f" Failed to parse LLM response as JSON: {e}")
                logger.debug(f"   Response: {content[:500]}")
                return []
        
        except Exception as e:
            logger.error(f" LLM extraction failed: {e}")
            return []
    
    def _build_extraction_prompt(
        self,
        text: str,
        fields: List[str],
        context: Optional[str]
    ) -> str:
        """Build extraction prompt for LLM"""
        
        context_str = f"\n\nContext: {context}" if context else ""
        
        prompt = f"""Extract structured data from this PDF document.

Extract the following fields:
{', '.join(fields)}
{context_str}

PDF Content:
{text}

Instructions:
1. Extract ALL instances of the requested fields
2. Return as JSON with key "items" containing an array of objects
3. Each object should have the requested fields as keys
4. If a field is not found, use null
5. For tables, extract each row as a separate item

Return ONLY valid JSON, no other text.

Example format:
{{
  "items": [
    {{{', '.join([f'"{field}": "value"' for field in fields])}}},
    {{{', '.join([f'"{field}": "value"' for field in fields])}}}
  ]
}}
"""
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """You are a precise document data extractor. 
Extract structured data from documents and return valid JSON.
Focus on accuracy and completeness.
For tables, extract each row as a separate item.
For repeated information, extract all instances."""
    
    def _generate_cache_key(self, pdf_url: str, fields: List[str]) -> str:
        """Generate cache key for PDF extraction"""
        fields_str = ','.join(sorted(fields))
        cache_input = f"{pdf_url}:{fields_str}"
        return f"pdf_{hashlib.md5(cache_input.encode()).hexdigest()}"


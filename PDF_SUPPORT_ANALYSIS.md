# PDF Support Analysis
## Can Universal LLM Scraper Handle PDFs?

**Date:** December 2024  
**Current Status:** ❌ No PDF support (HTML/JSON only)

---

## 🔍 Current Capabilities

### ✅ What Works Now
- HTML pages (static and JavaScript-rendered)
- Embedded JSON data
- JavaScript-heavy sites (with Camoufox)
- Multi-page pagination
- Anti-bot bypass (Kasada, Cloudflare)

### ❌ What Doesn't Work
- PDF documents
- Word documents (.docx)
- Excel files (.xlsx)
- Images (OCR)
- Scanned documents

---

## 🎯 PDF Use Cases

### **Scenario 1: PDF URLs on Web Pages**
**Example:** `https://example.com/document.pdf`

**Current Behavior:**
```python
# This will fail
result = scraper.scrape(
    url="https://example.com/report.pdf",
    fields=["title", "revenue", "date"]
)
# ❌ Error: Cannot parse PDF as HTML
```

### **Scenario 2: Web Page with PDF Links**
**Example:** Page listing PDF reports

**Current Behavior:**
```python
# This will work (extracts links)
result = scraper.scrape(
    url="https://example.com/reports",
    fields=["report_title", "pdf_url"]
)
# ✅ Returns: [{report_title: "Q4 Report", pdf_url: "https://...pdf"}]
# But won't extract data FROM the PDF
```

---

## 🔧 How to Add PDF Support

### **Option 1: LLM-Based PDF Extraction (Recommended)**

**Tools:**
- **LlamaIndex** - Best for document parsing + LLM
- **LangChain** - Alternative, more complex
- **PyMuPDF (fitz)** - For PDF text extraction
- **Marker** - PDF to Markdown (better than PyPDF)

**Implementation:**

```python
import pymupdf  # PyMuPDF
from llama_index import Document, VectorStoreIndex
import litellm

class PDFExtractor:
    """
    LLM-based PDF data extraction
    Similar to Direct LLM for HTML
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def extract_from_pdf(
        self, 
        pdf_url: str, 
        fields: List[str]
    ) -> List[Dict]:
        """
        Extract structured data from PDF using LLM
        
        Steps:
        1. Download PDF
        2. Convert to text/markdown
        3. Pass to LLM with extraction prompt
        4. Return structured data
        """
        # Download PDF
        pdf_bytes = await self._download_pdf(pdf_url)
        
        # Extract text (preserves layout)
        text = self._pdf_to_text(pdf_bytes)
        
        # Option A: Direct LLM extraction (fast)
        result = await self._extract_with_llm(text, fields)
        
        # Option B: RAG approach (better for large PDFs)
        # result = await self._extract_with_rag(pdf_bytes, fields)
        
        return result
    
    def _pdf_to_text(self, pdf_bytes: bytes) -> str:
        """Convert PDF to text preserving layout"""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        
        text_chunks = []
        for page_num, page in enumerate(doc):
            # Extract text with layout
            text = page.get_text("text")
            
            # Extract tables (if any)
            tables = page.find_tables()
            
            # Combine
            text_chunks.append(f"=== Page {page_num + 1} ===\n{text}")
        
        return "\n\n".join(text_chunks)
    
    async def _extract_with_llm(
        self, 
        text: str, 
        fields: List[str]
    ) -> List[Dict]:
        """
        Direct LLM extraction (similar to DirectLLMExtractor)
        """
        prompt = f"""
Extract the following fields from this document:
Fields: {', '.join(fields)}

Document:
{text[:20000]}  # Limit to avoid token limits

Return as JSON array of objects.
"""
        
        response = await litellm.acompletion(
            model="gpt-4o-mini",
            api_key=self.api_key,
            messages=[
                {"role": "system", "content": "You are a document data extractor."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    async def _extract_with_rag(
        self, 
        pdf_bytes: bytes, 
        fields: List[str]
    ) -> List[Dict]:
        """
        RAG approach for large PDFs (100+ pages)
        
        Better for:
        - Large documents
        - Multiple extractions from same PDF
        - Complex queries
        """
        # Create vector index
        from llama_index.core import VectorStoreIndex
        from llama_index.readers.file import PyMuPDFReader
        
        reader = PyMuPDFReader()
        documents = reader.load_data(pdf_bytes)
        
        index = VectorStoreIndex.from_documents(documents)
        
        # Query for each field
        query_engine = index.as_query_engine()
        
        results = []
        for field in fields:
            response = query_engine.query(f"Extract all values for: {field}")
            results.append({field: response.response})
        
        return results


# Usage
pdf_extractor = PDFExtractor(api_key="...")

# Extract from PDF URL
result = await pdf_extractor.extract_from_pdf(
    pdf_url="https://example.com/report.pdf",
    fields=["company_name", "revenue", "quarter", "year"]
)
```

**Pros:**
- ✅ Handles complex PDFs (multi-column, tables)
- ✅ Works with scanned PDFs (with OCR)
- ✅ Flexible (can extract any field)
- ✅ Same LLM approach as HTML scraping

**Cons:**
- ❌ More expensive (LLM calls)
- ❌ Slower (2-10 seconds per PDF)
- ❌ Token limits (20-30 pages max per call)

---

### **Option 2: Traditional PDF Parsing (Faster but Less Flexible)**

**Tools:**
- **pdfplumber** - Best for tables
- **PyPDF2** - Basic text extraction
- **tabula-py** - Tables only
- **camelot** - Advanced table extraction

**Implementation:**

```python
import pdfplumber
import pandas as pd

class TraditionalPDFParser:
    """
    Rule-based PDF parsing (no LLM)
    Good for: Structured PDFs, invoices, forms
    """
    
    def extract_tables(self, pdf_url: str) -> List[pd.DataFrame]:
        """Extract all tables from PDF"""
        response = requests.get(pdf_url)
        
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            tables = []
            for page in pdf.pages:
                # Extract tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)
            
            return tables
    
    def extract_text_by_pattern(
        self, 
        pdf_url: str, 
        patterns: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Extract text using regex patterns
        
        Example:
        patterns = {
            "invoice_number": r"Invoice #:\s*(\w+)",
            "total": r"Total:\s*\$?([\d,]+\.\d{2})"
        }
        """
        import re
        
        response = requests.get(pdf_url)
        
        results = {}
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text()
            
            for field, pattern in patterns.items():
                match = re.search(pattern, full_text)
                if match:
                    results[field] = match.group(1)
        
        return results


# Usage
parser = TraditionalPDFParser()

# Extract tables
tables = parser.extract_tables("https://example.com/data.pdf")

# Extract specific fields with regex
data = parser.extract_text_by_pattern(
    pdf_url="https://example.com/invoice.pdf",
    patterns={
        "invoice_number": r"Invoice #:\s*(\w+)",
        "date": r"Date:\s*([\d/]+)",
        "total": r"Total:\s*\$?([\d,]+\.\d{2})"
    }
)
```

**Pros:**
- ✅ Fast (no LLM calls)
- ✅ Cheap (no API costs)
- ✅ Reliable for structured PDFs

**Cons:**
- ❌ Requires manual pattern definition
- ❌ Doesn't work for complex layouts
- ❌ Not universal (needs customization per PDF type)

---

## 🚀 Recommended Approach

### **Hybrid: Auto-detect PDF Type**

```python
class UniversalDocumentExtractor:
    """
    Unified extractor for web pages AND documents
    Auto-detects format and uses appropriate method
    """
    
    async def extract(
        self, 
        url: str, 
        fields: List[str]
    ) -> Dict[str, Any]:
        """
        Universal extraction (web or document)
        """
        # Detect content type
        content_type = await self._detect_content_type(url)
        
        if content_type == "application/pdf":
            # PDF extraction
            return await self._extract_pdf(url, fields)
        
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Word doc extraction
            return await self._extract_docx(url, fields)
        
        elif content_type == "text/html":
            # Web scraping (existing logic)
            return await self._extract_html(url, fields)
        
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    
    async def _extract_pdf(self, url: str, fields: List[str]) -> Dict:
        """LLM-based PDF extraction"""
        # Check if structured PDF (tables, forms)
        if self._is_structured_pdf(url):
            # Use traditional parser (fast)
            return await self.traditional_parser.extract(url, fields)
        else:
            # Use LLM (flexible)
            return await self.llm_parser.extract(url, fields)
    
    def _is_structured_pdf(self, url: str) -> bool:
        """
        Detect if PDF is structured (tables/forms)
        vs unstructured (text-heavy)
        """
        # Download first page
        # Check for tables, forms, etc.
        # Return True if structured
        pass
```

---

## 📊 Performance Comparison

| Method | Speed | Cost | Accuracy | Use Case |
|--------|-------|------|----------|----------|
| **LLM (Direct)** | 2-5s | $0.01-0.05 | 90-95% | Complex PDFs, any layout |
| **LLM (RAG)** | 5-15s | $0.05-0.20 | 95-98% | Large PDFs (100+ pages) |
| **Traditional (pdfplumber)** | <1s | $0 | 60-80% | Structured PDFs, tables |
| **Traditional (regex)** | <1s | $0 | 40-70% | Forms, invoices (predictable) |

---

## 💰 Cost Analysis

### **LLM-Based PDF Extraction**

**Example: 10-page financial report**
```python
Tokens (input): ~8,000 tokens (10 pages × 800 words/page)
Tokens (output): ~500 tokens (structured data)

Cost (gpt-4o-mini):
- Input: 8,000 × $0.15/1M = $0.0012
- Output: 500 × $0.60/1M = $0.0003
- Total: $0.0015 per PDF

At scale:
- 1,000 PDFs/day: $1.50/day = $45/month
- 10,000 PDFs/day: $15/day = $450/month
```

**Alternative: gpt-4o (more accurate)**
```python
Cost (gpt-4o):
- Input: 8,000 × $2.50/1M = $0.02
- Output: 500 × $10/1M = $0.005
- Total: $0.025 per PDF (17x more expensive)

Only use gpt-4o for:
✅ Complex financial documents
✅ Legal contracts
✅ Medical records
```

---

## 🎯 Quick Implementation Plan

### **Phase 1: Add Basic PDF Support (Week 1)**

```python
# 1. Add PDF detection
# universal_scraper/core/content_detector.py

def detect_content_type(url: str) -> str:
    """Detect if URL is HTML, PDF, DOCX, etc."""
    response = requests.head(url)
    return response.headers.get('Content-Type', 'text/html')


# 2. Add PDF extractor
# universal_scraper/core/pdf_extractor.py

class PDFExtractor:
    async def extract(self, pdf_url: str, fields: List[str]):
        # Download PDF
        # Convert to text
        # Pass to LLM
        # Return structured data
        pass


# 3. Update main scraper
# universal_scraper/core/scraper.py

async def scrape(self, url: str, fields: List[str]):
    content_type = detect_content_type(url)
    
    if 'pdf' in content_type:
        return await self.pdf_extractor.extract(url, fields)
    else:
        # Existing HTML scraping logic
        return await self._scrape_html(url, fields)
```

### **Phase 2: Add Advanced Features (Week 2)**

- OCR for scanned PDFs (pytesseract)
- Table extraction (pdfplumber + LLM)
- Multi-file support (ZIP with PDFs)
- PDF caching (structure analysis)

---

## 🧪 Testing Recommendation

### **Test PDF URLs:**

1. **Simple Text PDF:**
   - `https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf`
   
2. **Financial Report (Tables):**
   - `https://www.sec.gov/files/[example_10k].pdf`
   
3. **Invoice (Structured):**
   - Any invoice PDF with standard format

### **Test Script:**

```python
# test_pdf_extraction.py
import asyncio
from universal_scraper.core.pdf_extractor import PDFExtractor

async def test_pdf():
    extractor = PDFExtractor(api_key="your-key")
    
    # Test 1: Simple PDF
    result = await extractor.extract_from_pdf(
        pdf_url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        fields=["title", "author", "content"]
    )
    print("Simple PDF:", result)
    
    # Test 2: Financial Report
    result = await extractor.extract_from_pdf(
        pdf_url="https://example.com/financial-report.pdf",
        fields=["company", "revenue", "quarter", "year"]
    )
    print("Financial Report:", result)

asyncio.run(test_pdf())
```

---

## ✅ Recommendation

**For your use case:**

1. **If you need PDF support soon:**
   - Add LLM-based PDF extraction (Option 1)
   - Uses same LLM approach as HTML scraping
   - Universal, flexible, accurate
   - Takes 3-5 days to implement

2. **If PDFs are not critical:**
   - Keep current HTML/JSON focus
   - Add PDF support later as separate feature
   - Charge premium for PDF extraction ($0.05 per PDF)

3. **If you need tables from PDFs:**
   - Use pdfplumber + LLM hybrid
   - Fast table extraction + LLM for understanding
   - Best accuracy for financial/tabular data

---

## 📦 Dependencies to Add

```python
# requirements.txt additions
pymupdf>=1.23.0           # PyMuPDF - PDF text extraction
pdfplumber>=0.10.0        # Table extraction
llama-index>=0.9.0        # RAG for large PDFs (optional)
pytesseract>=0.3.10       # OCR for scanned PDFs (optional)
pdf2image>=1.16.0         # Convert PDF to images (for OCR)
marker-pdf>=0.2.0         # Better PDF to Markdown (optional)
```

---

**Last Updated:** December 2024  
**Status:** PDF support can be added in 3-5 days with LLM-based extraction








# Document Processing Fix - NoneType Error

## Issue
Error: `'NoneType' object is not callable`

## Root Cause
The `DirectLLMExtractor` class was calling `self.html_transformer.transform_documents()` without checking if `html_transformer` was None. When langchain isn't installed, `html_transformer` is set to None, causing the error.

## Fix Applied

1. **Added None checks** before calling `html_transformer.transform_documents()`:
   - Line 358: Added check for `self.html_transformer is not None and Document is not None`
   - Line 513: Added check for `self.html_transformer is not None and Document is not None`
   - Line 766: Added check for `self.html_transformer is not None and Document is not None`

2. **Added langchain dependencies** to `requirements.txt`:
   - `langchain-community>=0.0.20`
   - `langchain-core>=0.1.23`

## Files Modified

1. `universal_scraper/core/direct_llm_extractor.py` - Added None checks
2. `requirements.txt` - Added langchain dependencies

## Deployment

A new deployment has been started to apply these fixes. Once complete, document processing should work correctly.

## Testing

After deployment, test with:
1. Upload a PDF or DOCX file
2. Set fields to extract (or leave empty)
3. Click "Process Document"

The error should be resolved and documents should process successfully.





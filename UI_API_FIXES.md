# UI API Request Processing Fixes

## Issues Found & Fixed

### 1. **Document Processing FormData Field Names**
**Problem**: Frontend was sending `useOCR` and `maxPages` but backend expects `use_ocr` and `max_pages` (snake_case)

**Fix**: Updated `DocumentProcessing.tsx` to use correct field names:
- `useOCR` → `use_ocr`
- `maxPages` → `max_pages`

### 2. **Multipart FormData Content-Type Override**
**Problem**: Axios default Content-Type header (`application/json`) was interfering with multipart/form-data requests

**Fix**: Created separate axios instance for document processing that doesn't override Content-Type:
```typescript
const multipartApi = axios.create({
  baseURL: API_BASE_URL,
});
// Let axios set Content-Type automatically with boundary
```

### 3. **Missing Request/Response Logging**
**Problem**: No visibility into what requests were being made or what responses were received

**Fix**: Added comprehensive console logging:
- Request logging: method, URL, headers, API key status
- Response logging: status, data
- Error logging: full error details, stack traces

### 4. **Response Data Processing**
**Problem**: Response data might not be properly extracted or displayed

**Fix**: Added explicit logging and validation:
- Log response before processing
- Validate data is an array
- Log number of results being set

## Changes Made

### `frontend/src/services/api.ts`
- Added request interceptor logging
- Added response interceptor logging  
- Created separate axios instance for multipart requests
- Improved error logging with full details

### `frontend/src/pages/DocumentProcessing.tsx`
- Fixed FormData field names (snake_case)
- Added request logging before sending
- Added response logging after receiving
- Better error handling

### `frontend/src/pages/WebScraping.tsx`
- Added request logging before scraping
- Added response logging after receiving
- Added result count logging
- Better error handling with stack traces

## Debugging

Now when you use the UI, check the browser console (F12 → Console) to see:
1. **Before request**: What's being sent (URL, fields, API key status)
2. **After request**: What was received (status, data)
3. **On error**: Full error details, response data, stack trace

## Testing

To test if requests are working:

1. **Open Browser Console** (F12 → Console)
2. **Try scraping a URL**:
   - Enter URL and fields
   - Click "Scrape"
   - Check console for:
     - "Starting scrape request" log
     - "API Request" log
     - "API Response" log
     - "Scrape response received" log

3. **Try document processing**:
   - Upload a file
   - Click "Process Document"
   - Check console for:
     - "Sending document processing request" log
     - "Document processing response received" log

## Common Issues to Check

1. **API Key Not Set**: Check console for "hasApiKey: false"
2. **CORS Errors**: Check Network tab for CORS errors
3. **404 Errors**: Check if endpoint URL is correct
4. **401 Errors**: API key is missing or invalid
5. **500 Errors**: Backend error - check response.data.detail

## Next Steps

If requests still aren't working:
1. Check browser console for errors
2. Check Network tab to see actual HTTP requests
3. Verify API key is set in localStorage
4. Verify API endpoint URL is correct
5. Check Cloud Run logs for backend errors





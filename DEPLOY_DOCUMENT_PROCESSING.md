# Deploy Document Processing Endpoint

## Issue
The document processing endpoint (`/document-processing/extract`) was added to the code but hasn't been deployed to Cloud Run yet, causing "Not Found" errors.

## Solution
Deploy the updated backend to Cloud Run.

## Deployment Steps

### Option 1: Using the deployment script (Recommended)
```bash
cd /Users/jevon_williams/Dev/universal-scraper
./deploy_to_gcp.sh
```

### Option 2: Manual deployment
```bash
cd /Users/jevon_williams/Dev/universal-scraper

# Set project
gcloud config set project soma-data-467016

# Build and deploy
gcloud builds submit \
    --config=infrastructure/cloudbuild/cloudbuild.yaml
```

## What Will Be Deployed

1. **New Endpoint**: `POST /document-processing/extract`
   - Accepts file uploads (PDF, DOCX, TXT, images)
   - Extracts text and uses LLM to extract structured data

2. **New Dependencies**:
   - PyPDF2 (PDF processing)
   - python-docx (Word documents)
   - pytesseract (OCR)
   - Pillow (Image processing)

3. **Updated Dockerfile**:
   - Includes Tesseract OCR system package

## Verification

After deployment, test the endpoint:
```bash
curl -X POST https://universal-scraper-api-968720932091.us-central1.run.app/document-processing/extract \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@test.pdf" \
  -F "fields=[]" \
  -F "use_ocr=false"
```

## Expected Deployment Time
- Build: ~5-10 minutes
- Deployment: ~2-3 minutes
- Total: ~10-15 minutes





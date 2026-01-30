# 🚀 Google Cloud Platform Deployment Guide

Complete guide for deploying Universal Scraper to Google Cloud Run and Firebase Hosting.

## 📋 Prerequisites

1. **Google Cloud Account** with billing enabled
2. **gcloud CLI** installed: https://cloud.google.com/sdk/docs/install
3. **Docker** installed (for local testing)
4. **Service Account** JSON file (already configured: `auth/soma-data-467016-d0118961514a.json`)

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│  Firebase Hosting (Frontend)        │
│  • React SPA                         │
│  • Free tier: 10GB transfer/month   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Cloud Run (Backend API)            │
│  • FastAPI server                    │
│  • Auto-scales 0 → 10 instances      │
│  • 2GB RAM, 2 CPU                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Universal Scraper Core             │
│  • AI-powered extraction             │
│  • Code caching                     │
│  • JSON-first architecture          │
└─────────────────────────────────────┘
```

## 🔧 Setup Steps

### 1. Authenticate with Google Cloud

```bash
# Set your project
export GCP_PROJECT_ID="soma-data-467016"
gcloud config set project $GCP_PROJECT_ID

# Authenticate using service account
gcloud auth activate-service-account \
  --key-file=auth/soma-data-467016-d0118961514a.json

# Or use your personal account
gcloud auth login
```

### 2. Enable Required APIs

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable firebase.googleapis.com
```

### 3. Deploy Backend (Cloud Run)

#### Option A: Automated Deployment (Recommended)

```bash
./deploy_to_gcp.sh
```

This script will:
- Build the Docker image
- Push to Container Registry
- Deploy to Cloud Run
- Configure auto-scaling

#### Option B: Manual Deployment

```bash
./deploy_to_gcp_manual.sh
```

#### Option C: Using Cloud Build

```bash
gcloud builds submit --config=infrastructure/cloudbuild/cloudbuild.yaml
```

### 4. Get Service URL

After deployment, get your API URL:

```bash
gcloud run services describe universal-scraper-api \
  --region=us-central1 \
  --format='value(status.url)'
```

Example output: `https://universal-scraper-api-xxxxx-uc.a.run.app`

### 5. Test the API

```bash
# Health check
curl https://YOUR-SERVICE-URL/health

# Scrape a URL
curl -X POST https://YOUR-SERVICE-URL/scrape \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_OPENAI_API_KEY' \
  -d '{
    "url": "https://example.com",
    "fields": ["title", "description"]
  }'
```

### 6. Set Environment Variables (Optional)

For production, set secrets in Cloud Run:

```bash
gcloud run services update universal-scraper-api \
  --region=us-central1 \
  --set-secrets="OPENAI_API_KEY=openai-key:latest"
```

Or set via console:
1. Go to Cloud Run → universal-scraper-api → Edit & Deploy New Revision
2. Variables & Secrets → Add Variable
3. Name: `OPENAI_API_KEY`, Value: `your-key`

### 7. Deploy Frontend (Firebase Hosting)

```bash
cd frontend

# Install dependencies (if not done)
npm install

# Build the frontend
npm run build

# Initialize Firebase (first time only)
firebase init hosting

# Deploy
firebase deploy --only hosting
```

## 📊 Configuration

### Cloud Run Settings

- **Memory**: 2GB (configurable in `cloudbuild.yaml`)
- **CPU**: 2 vCPU
- **Timeout**: 300 seconds (5 minutes)
- **Min Instances**: 0 (scales to zero)
- **Max Instances**: 10
- **Concurrency**: 80 requests per instance

### Cost Estimation

For **10,000 requests/month** (30s avg, 2GB RAM):

```
Cloud Run:
  CPU: 300,000 seconds × $0.0000024 = $0.72
  Memory: 300,000 seconds × $0.0000167 = $5.01
  Free tier: -$3.00 (180K CPU-seconds free)
  Total: ~$2.73/month

Firebase Hosting:
  Free tier: 10GB transfer/month
  Total: $0/month (under free tier)

Total: ~$2.73/month for 10K requests
```

## 🔐 Security

### API Key Management

**Option 1: Environment Variables** (Simple)
```bash
gcloud run services update universal-scraper-api \
  --set-env-vars="OPENAI_API_KEY=sk-..."
```

**Option 2: Secret Manager** (Recommended)
```bash
# Create secret
echo -n "sk-..." | gcloud secrets create openai-api-key --data-file=-

# Grant access
gcloud secrets add-iam-policy-binding openai-api-key \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Use in Cloud Run
gcloud run services update universal-scraper-api \
  --set-secrets="OPENAI_API_KEY=openai-api-key:latest"
```

### CORS Configuration

Update `api/main.py` to restrict CORS:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # Your Firebase URL
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 🧪 Testing

### Local Testing

```bash
# Run API locally
cd api
python -m api.main

# Test locally
curl -X POST http://localhost:8080/scrape \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_KEY' \
  -d '{"url": "https://example.com", "fields": ["title"]}'
```

### Test with Docker

```bash
# Build image
docker build -t universal-scraper-api .

# Run container
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=your-key \
  universal-scraper-api

# Test
curl http://localhost:8080/health
```

## 📈 Monitoring

### View Logs

```bash
# Stream logs
gcloud run services logs tail universal-scraper-api \
  --region=us-central1

# View in console
# https://console.cloud.google.com/run/detail/us-central1/universal-scraper-api/logs
```

### Metrics

View in Cloud Console:
- Request count
- Latency
- Error rate
- Instance count

## 🔄 Continuous Deployment

### GitHub Actions (Optional)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: google-github-actions/setup-gcloud@v1
        with:
          service_account_key: ${{ secrets.GCP_SA_KEY }}
          project_id: soma-data-467016
      - run: ./deploy_to_gcp.sh
```

## 🐛 Troubleshooting

### Issue: Build fails

**Solution**: Check Cloud Build logs:
```bash
gcloud builds list --limit=1
gcloud builds log BUILD_ID
```

### Issue: Service won't start

**Solution**: Check container logs:
```bash
gcloud run services logs read universal-scraper-api \
  --region=us-central1 \
  --limit=50
```

### Issue: Out of memory

**Solution**: Increase memory in `cloudbuild.yaml`:
```yaml
--memory 4Gi  # Increase from 2Gi
```

### Issue: Timeout errors

**Solution**: Increase timeout:
```yaml
--timeout 600  # Increase from 300
```

## 📚 Next Steps

1. ✅ **Backend deployed** - Cloud Run API is live
2. 🔄 **Frontend** - Complete React app (see `frontend/` directory)
3. 🔄 **Firebase Hosting** - Deploy frontend
4. 🔄 **Monitoring** - Set up alerts
5. 🔄 **CI/CD** - Automate deployments

## 🔗 Useful Links

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Firebase Hosting](https://firebase.google.com/docs/hosting)
- [Cloud Build](https://cloud.google.com/build/docs)
- [API Documentation](https://YOUR-SERVICE-URL/docs) (Swagger UI)

## 💡 Tips

1. **Start with free tier** - Cloud Run has generous free tier
2. **Use secrets** - Don't hardcode API keys
3. **Monitor costs** - Set up billing alerts
4. **Test locally first** - Use Docker for local testing
5. **Enable logging** - Helps debug issues

---

**Status**: ✅ Backend ready for deployment  
**Next**: Complete frontend and deploy to Firebase Hosting





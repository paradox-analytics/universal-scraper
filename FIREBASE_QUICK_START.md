# 🔥 Firebase Quick Start

## ✅ What's Been Set Up

All Firebase configuration files have been created! Here's what you need to do to connect your Firebase project:

---

## 🚀 Quick Setup (3 Steps)

### Step 1: Update Firebase Project ID

Edit `.firebaserc` and replace `YOUR_FIREBASE_PROJECT_ID` with your actual Firebase project ID:

```bash
# Open .firebaserc and update:
{
  "projects": {
    "default": "your-actual-firebase-project-id"
  }
}
```

### Step 2: Get Firebase Config Values

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project
3. Click ⚙️ **Project Settings** > **General** tab
4. Scroll to **Your apps** section
5. If no web app exists, click **Add app** > **Web** (</> icon)
6. Copy the `firebaseConfig` values

### Step 3: Create Environment File

Create `frontend/.env` file with your Firebase config:

```bash
cd frontend
cat > .env << 'EOF'
# Firebase Configuration
VITE_FIREBASE_API_KEY=your_api_key_here
VITE_FIREBASE_AUTH_DOMAIN=your-project-id.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-project-id
VITE_FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id
VITE_FIREBASE_APP_ID=your_app_id
VITE_FIREBASE_MEASUREMENT_ID=your_measurement_id

# Backend API (already configured)
VITE_API_BASE_URL=https://universal-scraper-api-968720932091.us-central1.run.app
VITE_API_KEY=your_openai_api_key_here
EOF
```

Replace the placeholder values with your actual Firebase config.

---

## 📦 Install Dependencies

```bash
cd frontend
npm install
```

This will install Firebase SDK and all other dependencies.

---

## 🧪 Test Connection

### Test Firebase Connection

Create a test file `frontend/src/test-firebase.ts`:

```typescript
import { db, auth } from './config/firebase';
import { collection, getDocs } from 'firebase/firestore';

// Test Firestore connection
async function testFirebase() {
  try {
    const testCollection = collection(db, 'test');
    const snapshot = await getDocs(testCollection);
    console.log('✅ Firebase connected!', snapshot.size, 'documents');
  } catch (error) {
    console.error('❌ Firebase connection failed:', error);
  }
}

testFirebase();
```

### Test Backend Connection

```typescript
import { apiService } from './services/api';

// Test backend
async function testBackend() {
  try {
    const health = await apiService.healthCheck();
    console.log('✅ Backend connected!', health);
  } catch (error) {
    console.error('❌ Backend connection failed:', error);
  }
}

testBackend();
```

---

## 🚀 Deploy to Firebase Hosting

### Option 1: Use the Deployment Script

```bash
./deploy_frontend.sh
```

### Option 2: Manual Deployment

```bash
# 1. Build frontend
cd frontend
npm run build

# 2. Deploy to Firebase
cd ..
firebase deploy --only hosting
```

---

## 📁 Files Created

- ✅ `firebase.json` - Firebase Hosting configuration
- ✅ `.firebaserc` - Firebase project ID (needs your project ID)
- ✅ `firestore.rules` - Firestore security rules
- ✅ `firestore.indexes.json` - Firestore indexes
- ✅ `frontend/src/config/firebase.ts` - Firebase SDK initialization
- ✅ `frontend/src/config/api.ts` - Backend API configuration
- ✅ `frontend/src/services/api.ts` - API service for backend calls
- ✅ `deploy_frontend.sh` - Deployment script

---

## 🔗 How It Works

```
Frontend (React + Vite)
    ↓
Firebase Hosting (Static Files)
    ↓
Firebase SDK (Firestore, Auth, Storage)
    ↓
API Service (frontend/src/services/api.ts)
    ↓
Cloud Run Backend (https://universal-scraper-api-968720932091.us-central1.run.app)
```

---

## 🆘 Troubleshooting

### Firebase CLI Not Found
```bash
npm install -g firebase-tools
firebase login
```

### Build Fails
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Environment Variables Not Working
- Make sure `.env` file is in `frontend/` directory
- Restart dev server after creating `.env`
- Variables must start with `VITE_` to be exposed to frontend

### Firebase Project Not Found
```bash
firebase projects:list  # List your projects
firebase use <project-id>  # Switch to your project
```

---

## 📚 Next Steps

1. ✅ Update `.firebaserc` with your Firebase project ID
2. ✅ Create `frontend/.env` with Firebase config
3. ✅ Install dependencies: `cd frontend && npm install`
4. ✅ Test locally: `npm run dev`
5. ✅ Deploy: `./deploy_frontend.sh`

---

**Need Help?** Check `FIREBASE_SETUP.md` for detailed instructions.





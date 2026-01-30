# 🔥 Firebase Setup Guide

## ✅ Files Created

1. **`firebase.json`** - Firebase Hosting configuration
2. **`.firebaserc`** - Firebase project configuration
3. **`frontend/src/config/firebase.ts`** - Firebase SDK initialization
4. **`frontend/src/config/api.ts`** - Backend API configuration
5. **`frontend/src/services/api.ts`** - API service for connecting to Cloud Run backend
6. **`firestore.rules`** - Firestore security rules
7. **`firestore.indexes.json`** - Firestore indexes configuration

---

## 🚀 Setup Steps

### 1. Get Your Firebase Project Configuration

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project
3. Click the gear icon ⚙️ > **Project Settings**
4. Scroll down to **Your apps** section
5. If you don't have a web app, click **Add app** > **Web** (</> icon)
6. Copy the configuration values

### 2. Update Firebase Project ID

Edit `.firebaserc` and replace `YOUR_FIREBASE_PROJECT_ID` with your actual Firebase project ID:

```json
{
  "projects": {
    "default": "your-actual-project-id"
  }
}
```

### 3. Create Environment File

Copy the example environment file and fill in your Firebase config:

```bash
cd frontend
cp .env.example .env
```

Then edit `.env` with your Firebase configuration values:

```env
VITE_FIREBASE_API_KEY=AIza...
VITE_FIREBASE_AUTH_DOMAIN=your-project-id.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-project-id
VITE_FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789
VITE_FIREBASE_APP_ID=1:123456789:web:abc123
VITE_FIREBASE_MEASUREMENT_ID=G-XXXXXXXXXX

# Backend API (already configured)
VITE_API_BASE_URL=https://universal-scraper-api-968720932091.us-central1.run.app
VITE_API_KEY=your_openai_api_key_here
```

### 4. Install Firebase SDK (if not already installed)

```bash
cd frontend
npm install firebase
```

### 5. Install Firebase CLI (if not already installed)

```bash
npm install -g firebase-tools
```

### 6. Login to Firebase

```bash
firebase login
```

### 7. Initialize Firebase (if not already done)

```bash
firebase init
```

Select:
- ✅ **Hosting** - Configure files for Firebase Hosting
- ✅ **Firestore** - Set up security rules and indexes

When prompted:
- **What do you want to use as your public directory?** → `frontend/dist`
- **Configure as a single-page app?** → **Yes**
- **Set up automatic builds and deploys with GitHub?** → **No** (or Yes if you want CI/CD)

### 8. Build Frontend

```bash
cd frontend
npm run build
```

### 9. Deploy to Firebase Hosting

```bash
# From project root
firebase deploy --only hosting
```

Or deploy everything:

```bash
firebase deploy
```

---

## 🔗 Connecting Frontend to Backend

The frontend is already configured to connect to your Cloud Run backend:

- **Backend URL**: `https://universal-scraper-api-968720932091.us-central1.run.app`
- **API Service**: `frontend/src/services/api.ts`

### Usage Example

```typescript
import { apiService } from '@/services/api';

// Health check
const health = await apiService.healthCheck();

// Scrape a URL
const result = await apiService.scrape({
  url: 'https://example.com',
  fields: ['title', 'price', 'description'],
  mode: 'hybrid'
});
```

---

## 📁 Project Structure

```
universal-scraper/
├── firebase.json              # Firebase Hosting config
├── .firebaserc                # Firebase project ID
├── firestore.rules            # Firestore security rules
├── firestore.indexes.json     # Firestore indexes
├── frontend/
│   ├── .env                   # Environment variables (create from .env.example)
│   ├── src/
│   │   ├── config/
│   │   │   ├── firebase.ts    # Firebase SDK initialization
│   │   │   └── api.ts         # Backend API config
│   │   └── services/
│   │       └── api.ts         # API service for backend calls
│   └── dist/                  # Build output (for Firebase Hosting)
```

---

## 🔐 Security Notes

### Current Setup (Development)
- Firestore rules allow all read/write (⚠️ **NOT for production**)
- API key stored in environment variables

### For Production
1. **Update Firestore Rules** (`firestore.rules`):
   ```javascript
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       match /{document=**} {
         allow read: if request.auth != null;
         allow write: if request.auth != null && request.auth.uid == resource.data.userId;
       }
     }
   }
   ```

2. **Use Firebase Authentication** instead of API keys in frontend
3. **Set up Cloud Run authentication** for backend
4. **Use environment variables** for sensitive data

---

## 🧪 Testing

### Test Firebase Connection

```typescript
import { db, auth } from '@/config/firebase';

// Test Firestore
import { collection, getDocs } from 'firebase/firestore';
const querySnapshot = await getDocs(collection(db, 'test'));
```

### Test Backend Connection

```typescript
import { apiService } from '@/services/api';

// Health check
const health = await apiService.healthCheck();
console.log('Backend status:', health);
```

---

## 📊 Firebase Services Available

- ✅ **Firebase Hosting** - Static site hosting (FREE tier: 10GB transfer/month)
- ✅ **Firestore** - NoSQL database (FREE tier: 50K reads/day, 20K writes/day)
- ✅ **Firebase Auth** - Authentication (FREE tier: unlimited users)
- ✅ **Firebase Storage** - File storage (FREE tier: 5GB storage, 1GB/day downloads)

---

## 🚀 Next Steps

1. ✅ Firebase configuration files created
2. ⏳ Fill in Firebase project details in `.firebaserc` and `.env`
3. ⏳ Install Firebase SDK: `npm install firebase` (in frontend directory)
4. ⏳ Build frontend: `npm run build`
5. ⏳ Deploy to Firebase: `firebase deploy --only hosting`

---

## 📚 Resources

- [Firebase Documentation](https://firebase.google.com/docs)
- [Firebase Hosting Guide](https://firebase.google.com/docs/hosting)
- [Firestore Security Rules](https://firebase.google.com/docs/firestore/security/get-started)
- [Vite Environment Variables](https://vitejs.dev/guide/env-and-mode.html)

---

**Need Help?** Check the Firebase Console for your project configuration values.





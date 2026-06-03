# 🔥 How to Access Your Firebase Project

## Your Firebase Project Details

- **Project Name**: `universal-scaper`
- **Project ID**: `998106783458`
- **Domain**: `paradoxanalytics.com`

---

## 🌐 Access Firebase Console

### Option 1: Direct URL
Go to: **https://console.firebase.google.com/project/universal-scaper**

### Option 2: Via Firebase Console
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click on your project: **universal-scaper**

---

## 📋 Get Firebase Configuration Values

You need these values to connect your frontend. Here's how to get them:

### Step 1: Open Project Settings
1. In Firebase Console, click the **⚙️ gear icon** (top left)
2. Select **Project Settings**

### Step 2: Get Web App Config
1. Scroll down to **Your apps** section
2. If you don't have a web app yet:
   - Click **Add app** button
   - Select **Web** (</> icon)
   - Register app with nickname: `universal-scraper-web`
   - Click **Register app**
3. Copy the `firebaseConfig` object that appears

It will look like this:
```javascript
const firebaseConfig = {
  apiKey: "AIzaSy...",
  authDomain: "universal-scaper.firebaseapp.com",
  projectId: "universal-scaper",
  storageBucket: "universal-scaper.appspot.com",
  messagingSenderId: "998106783458",
  appId: "1:998106783458:web:abc123def456",
  measurementId: "G-XXXXXXXXXX"
};
```

### Step 3: Create Environment File

Create `frontend/.env` file with these values:

```bash
cd frontend
cat > .env << 'EOF'
# Firebase Configuration
VITE_FIREBASE_API_KEY=AIzaSy... (from firebaseConfig)
VITE_FIREBASE_AUTH_DOMAIN=universal-scaper.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=universal-scaper
VITE_FIREBASE_STORAGE_BUCKET=universal-scaper.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=998106783458
VITE_FIREBASE_APP_ID=1:998106783458:web:abc123def456 (from firebaseConfig)
VITE_FIREBASE_MEASUREMENT_ID=G-XXXXXXXXXX (from firebaseConfig, optional)

# Backend API (already configured)
VITE_API_BASE_URL=https://universal-scraper-api-968720932091.us-central1.run.app
VITE_API_KEY=your_openai_api_key_here
EOF
```

---

## 🔐 Firebase CLI Access

### Login to Firebase CLI
```bash
firebase login
```

This will open a browser window for authentication.

### Use Your Project
```bash
firebase use universal-scaper
```

### Verify Connection
```bash
firebase projects:list
```

You should see `universal-scaper` in the list.

---

## 🚀 Enable Firebase Services

### 1. Enable Firestore Database
1. In Firebase Console, go to **Firestore Database**
2. Click **Create database**
3. Choose **Start in test mode** (for development)
4. Select location (e.g., `us-central1`)
5. Click **Enable**

### 2. Enable Firebase Hosting
1. In Firebase Console, go to **Hosting**
2. Click **Get started**
3. Follow the setup wizard

### 3. Enable Authentication (Optional)
1. In Firebase Console, go to **Authentication**
2. Click **Get started**
3. Enable sign-in methods as needed

---

## 📱 Firebase Console URLs

- **Dashboard**: https://console.firebase.google.com/project/universal-scaper/overview
- **Authentication**: https://console.firebase.google.com/project/universal-scaper/authentication
- **Firestore**: https://console.firebase.google.com/project/universal-scaper/firestore
- **Hosting**: https://console.firebase.google.com/project/universal-scaper/hosting
- **Storage**: https://console.firebase.google.com/project/universal-scaper/storage
- **Functions**: https://console.firebase.google.com/project/universal-scaper/functions

---

## ✅ Quick Checklist

- [x] Project created: `universal-scaper`
- [ ] Firebase CLI logged in: `firebase login`
- [ ] Web app registered in Firebase Console
- [ ] Firestore Database enabled
- [ ] Firebase Hosting enabled
- [ ] Environment file created: `frontend/.env`
- [ ] Dependencies installed: `cd frontend && npm install`

---

## 🧪 Test Connection

After setting up, test the connection:

```bash
# Test Firebase CLI
firebase projects:list

# Test project connection
firebase use universal-scaper
firebase deploy --only hosting --dry-run
```

---

## 📚 Next Steps

1. **Get Firebase Config**: Follow steps above to get your `firebaseConfig`
2. **Create `.env` file**: Copy config values to `frontend/.env`
3. **Install dependencies**: `cd frontend && npm install`
4. **Build & Deploy**: `./deploy_frontend.sh`

---

**Need Help?** The Firebase Console is your best friend - all configuration is done there!





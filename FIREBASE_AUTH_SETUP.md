# Firebase Authentication Setup Guide

## Error: `auth/configuration-not-found`

This error occurs when Firebase Authentication is not properly configured. Follow these steps to fix it:

## Step 1: Enable Firebase Authentication

1. Go to [Firebase Console](https://console.firebase.google.com/project/universal-scaper/authentication)
2. Click on "Authentication" in the left sidebar
3. Click "Get Started" if you haven't enabled Auth yet
4. You should see the "Sign-in method" tab

## Step 2: Enable Sign-In Providers

### Google Sign-In
1. In the "Sign-in method" tab, click on "Google"
2. Toggle "Enable" to ON
3. Enter your project's support email
4. Click "Save"

### Apple Sign-In
1. In the "Sign-in method" tab, click on "Apple"
2. Toggle "Enable" to ON
3. Enter your Apple Services ID (if you have one)
4. Click "Save"

### Email/Password
1. In the "Sign-in method" tab, click on "Email/Password"
2. Toggle "Enable" to ON
3. Toggle "Email link (passwordless sign-in)" if desired
4. Click "Save"

## Step 3: Configure Authorized Domains

1. In the Authentication settings, go to "Settings" tab
2. Scroll to "Authorized domains"
3. Ensure these domains are listed:
   - `universal-scaper.web.app`
   - `universal-scaper.firebaseapp.com`
   - `localhost` (for local development)
   - Your custom domain (if applicable)

## Step 4: Verify Configuration

After enabling the providers, the error should be resolved. The app will automatically use the configured providers.

## Troubleshooting

- **Still seeing the error?** Clear your browser cache and try again
- **Google Sign-In not working?** Make sure the OAuth consent screen is configured in Google Cloud Console
- **Apple Sign-In not working?** Apple Sign-In requires additional setup in Apple Developer Portal

## Quick Setup Command

You can also enable providers via Firebase CLI (if you have the right permissions):

```bash
# Enable Email/Password (this is usually enabled by default)
# Google and Apple need to be enabled via Console due to OAuth setup requirements
```





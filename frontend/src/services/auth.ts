import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithPopup,
  signOut,
  sendPasswordResetEmail,
  sendEmailVerification,
  updateProfile,
  User,
  GoogleAuthProvider,
  OAuthProvider,
} from 'firebase/auth';
import { doc, setDoc, getDoc, updateDoc } from 'firebase/firestore';
import { auth, db } from '../config/firebase';

export interface UserSettings {
  apiKey?: string;
  proxyConfig?: any;
  aiConfig?: any;
  webUnblockerConfig?: any;
  preferences?: {
    theme?: 'light' | 'dark';
    notifications?: boolean;
  };
}

export interface UserProfile {
  uid: string;
  email: string | null;
  displayName: string | null;
  photoURL: string | null;
  emailVerified: boolean;
  mfaEnabled: boolean;
  createdAt: number;
  lastLoginAt: number;
}

// Google Sign In
export const signInWithGoogle = async () => {
  try {
    const provider = new GoogleAuthProvider();
    provider.addScope('email');
    provider.addScope('profile');

    console.log('Attempting Google Sign-In...');
    console.log('Auth instance:', auth);
    console.log('Provider:', provider);

    const result = await signInWithPopup(auth, provider);
    console.log('Sign-in successful:', result.user.email);

    await saveUserProfile(result.user);
    return result;
  } catch (error: any) {
    console.error('Google Sign-In Error Details:', {
      code: error.code,
      message: error.message,
      stack: error.stack,
      customData: error.customData,
      fullError: error
    });
    throw error;
  }
};

// Apple Sign In
export const signInWithApple = async () => {
  const provider = new OAuthProvider('apple.com');
  provider.addScope('email');
  provider.addScope('name');
  const result = await signInWithPopup(auth, provider);
  await saveUserProfile(result.user);
  return result;
};

// Email/Password Sign In
export const signInWithEmail = async (email: string, password: string) => {
  const result = await signInWithEmailAndPassword(auth, email, password);
  await updateUserLastLogin(result.user.uid);
  return result;
};

// Email/Password Sign Up
export const signUpWithEmail = async (
  email: string,
  password: string,
  displayName?: string
) => {
  const result = await createUserWithEmailAndPassword(auth, email, password);

  // Update profile with display name
  if (displayName) {
    await updateProfile(result.user, { displayName });
  }

  // Send email verification
  await sendEmailVerification(result.user);

  // Save user profile
  await saveUserProfile(result.user);

  return result;
};

// Sign Out
export const signOutUser = async () => {
  await signOut(auth);
};

// Password Reset
export const resetPassword = async (email: string) => {
  await sendPasswordResetEmail(auth, email);
};

// Save user profile to Firestore
const saveUserProfile = async (user: User) => {
  const userRef = doc(db, 'users', user.uid);
  const userSnap = await getDoc(userRef);

  const profile: Partial<UserProfile> = {
    uid: user.uid,
    email: user.email,
    displayName: user.displayName,
    photoURL: user.photoURL,
    emailVerified: user.emailVerified,
    mfaEnabled: false, // Will be updated when MFA is enabled
    lastLoginAt: Date.now(),
  };

  if (!userSnap.exists()) {
    // New user
    profile.createdAt = Date.now();
    await setDoc(userRef, profile);
  } else {
    // Update existing user
    await updateDoc(userRef, {
      displayName: user.displayName,
      photoURL: user.photoURL,
      emailVerified: user.emailVerified,
      lastLoginAt: Date.now(),
    });
  }
};

// Update last login time
const updateUserLastLogin = async (uid: string) => {
  const userRef = doc(db, 'users', uid);
  await updateDoc(userRef, {
    lastLoginAt: Date.now(),
  });
};

// Get user settings from Firestore
export const getUserSettings = async (uid: string): Promise<UserSettings | null> => {
  try {
    const settingsRef = doc(db, 'user_settings', uid);
    const settingsSnap = await getDoc(settingsRef);

    if (settingsSnap.exists()) {
      return settingsSnap.data() as UserSettings;
    }
    return null;
  } catch (error) {
    console.error('Error fetching user settings:', error);
    return null;
  }
};

// Save user settings to Firestore
export const saveUserSettings = async (uid: string, settings: UserSettings) => {
  try {
    const settingsRef = doc(db, 'user_settings', uid);
    await setDoc(settingsRef, {
      ...settings,
      updatedAt: Date.now(),
    }, { merge: true });
  } catch (error) {
    console.error('Error saving user settings:', error);
    throw error;
  }
};

// Get user profile
export const getUserProfile = async (uid: string): Promise<UserProfile | null> => {
  try {
    const userRef = doc(db, 'users', uid);
    const userSnap = await getDoc(userRef);

    if (userSnap.exists()) {
      return userSnap.data() as UserProfile;
    }
    return null;
  } catch (error) {
    console.error('Error fetching user profile:', error);
    return null;
  }
};

// Note: Firebase Auth supports phone-based MFA, not TOTP directly
// For TOTP, we'll need to implement a custom solution or use phone MFA
// Phone MFA implementation can be added later when needed

// Get Firebase Auth token for backend API
export const getAuthToken = async (): Promise<string | null> => {
  const user = auth.currentUser;
  if (!user) return null;

  try {
    const token = await user.getIdToken();
    return token;
  } catch (error) {
    console.error('Error getting auth token:', error);
    return null;
  }
};


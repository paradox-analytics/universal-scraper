import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { User, onAuthStateChanged } from 'firebase/auth';
import { auth, DEV_MODE } from '../config/firebase';
import { getUserSettings, getUserProfile, getAuthToken } from '../services/auth';
import { setTokenGetter } from '../services/api';
import type { UserSettings, UserProfile } from '../services/auth';

interface AuthContextType {
  user: User | null;
  userProfile: UserProfile | null;
  userSettings: UserSettings | null;
  loading: boolean;
  refreshSettings: () => Promise<void>;
  refreshProfile: () => Promise<void>;
  getToken: () => Promise<string | null>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [userSettings, setUserSettings] = useState<UserSettings | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (DEV_MODE) {
      setUser({ uid: 'dev-user', email: 'dev@localhost' } as unknown as User);
      setLoading(false);
      return;
    }

    setTokenGetter(getAuthToken);

    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      setUser(currentUser);

      if (currentUser) {
        try {
          const [profile, settings] = await Promise.all([
            getUserProfile(currentUser.uid),
            getUserSettings(currentUser.uid),
          ]);

          setUserProfile(profile);
          setUserSettings(settings);
        } catch (error) {
          console.error('Error loading user data:', error);
        }
      } else {
        setUserProfile(null);
        setUserSettings(null);
      }

      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const refreshSettings = async () => {
    if (user) {
      const settings = await getUserSettings(user.uid);
      setUserSettings(settings);
    }
  };

  const refreshProfile = async () => {
    if (user) {
      const profile = await getUserProfile(user.uid);
      setUserProfile(profile);
    }
  };

  const getToken = async () => {
    return await getAuthToken();
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        userProfile,
        userSettings,
        loading,
        refreshSettings,
        refreshProfile,
        getToken,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}


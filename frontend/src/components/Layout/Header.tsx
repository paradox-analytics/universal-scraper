import { useState, useRef, useEffect } from 'react';
import { BellIcon, UserCircleIcon } from '@heroicons/react/24/outline';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { signOutUser } from '../../services/auth';
import GlobalStatusIndicators from '../Common/GlobalStatusIndicators';

export function Header() {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const { user } = useAuth();
  const navigate = useNavigate();

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowUserMenu(false);
      }
    };

    if (showUserMenu) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showUserMenu]);

  const handleSignOut = async () => {
    try {
      await signOutUser();
      navigate('/login');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  return (
    <header className="bg-gray-900 shadow-sm border-b border-gray-800">
      <div className="flex items-center justify-between h-16 px-6">
        <div className="flex items-center gap-3">
          <img src="/paradocs-icon.svg?v=3" alt="ParaDocs" className="w-8 h-8" key="logo-header" />
          <Link to="/" className="text-lg font-semibold text-gray-50 hover:text-purple-400 transition-colors">
            AI Studio
          </Link>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* Global Status Indicators: AI, Proxy, Web Unblocker */}
          <GlobalStatusIndicators />
          
          <button className="p-2 text-gray-400 hover:text-gray-200 relative transition-colors">
            <BellIcon className="h-6 w-6" />
            <span className="absolute top-1 right-1 h-2 w-2 bg-red-500 rounded-full"></span>
          </button>
          
          <div className="relative" ref={menuRef}>
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center space-x-2 p-2 rounded-lg hover:bg-gray-800 transition-colors"
            >
              {user?.photoURL ? (
                <img
                  src={user.photoURL}
                  alt={user.displayName || 'User'}
                  className="h-8 w-8 rounded-full"
                />
              ) : (
                <UserCircleIcon className="h-6 w-6 text-gray-400" />
              )}
              <span className="text-sm font-medium text-gray-200">
                {user?.displayName || user?.email || 'Account'}
              </span>
            </button>
            
            {showUserMenu && (
              <div className="absolute right-0 mt-2 w-56 bg-gray-800 rounded-lg shadow-lg border border-gray-700 py-1 z-50">
                <div className="px-4 py-2 border-b border-gray-700">
                  <p className="text-sm font-medium text-gray-200">{user?.displayName || 'User'}</p>
                  <p className="text-xs text-gray-400 truncate">{user?.email}</p>
                </div>
                <Link
                  to="/settings"
                  className="block px-4 py-2 text-sm text-gray-200 hover:bg-gray-700"
                  onClick={() => setShowUserMenu(false)}
                >
                  Settings
                </Link>
                <a href="/billing" className="block px-4 py-2 text-sm text-gray-200 hover:bg-gray-700">
                  Billing
                </a>
                <hr className="my-1 border-gray-700" />
                <button
                  onClick={handleSignOut}
                  className="w-full text-left block px-4 py-2 text-sm text-red-400 hover:bg-gray-700"
                >
                  Sign Out
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

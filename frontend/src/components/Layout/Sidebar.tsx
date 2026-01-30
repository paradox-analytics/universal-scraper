import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  HomeIcon,
  GlobeAltIcon,
  DocumentTextIcon,
  Cog6ToothIcon,
  CircleStackIcon,
  ClockIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  Bars3Icon,
  XMarkIcon,
} from '@heroicons/react/24/outline';

const mainNavigation = [
  { name: 'Dashboard', href: '/', icon: HomeIcon },
];

const agentsNavigation = [
  { name: 'Scraper', href: '/web-scraping', icon: GlobeAltIcon },
  { name: 'Document Processor', href: '/document-processing', icon: DocumentTextIcon },
];

const otherNavigation = [
  { name: 'Agent Jobs', href: '/history', icon: ClockIcon },
  { name: 'Cache', href: '/cache', icon: CircleStackIcon },
  { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
];

export function Sidebar() {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(true); // Collapsed by default
  const [agentsExpanded, setAgentsExpanded] = useState(
    location.pathname === '/web-scraping' || location.pathname === '/document-processing'
  );

  const isActive = (href: string) => location.pathname === href;
  const isAgentsActive = agentsNavigation.some(item => isActive(item.href));

  return (
    <div className={`flex flex-col bg-dark-900 border-r border-dark-700 shadow-lg transition-all duration-300 ${collapsed ? 'w-20' : 'w-64'}`}>
      <div className={`flex items-center h-20 border-b border-dark-700 ${collapsed ? 'justify-center px-4' : 'justify-between px-6'}`}>
        {!collapsed && (
          <div className="flex items-center overflow-hidden whitespace-nowrap">
            <img src="/paradocs-icon.svg?v=3" alt="ParaDocs" className="w-8 h-8 mr-3 flex-shrink-0" key="logo-sidebar" />
            <h1 className="text-xl font-bold text-white truncate">ParaDocs</h1>
          </div>
        )}
        {collapsed && (
          <img src="/paradocs-icon.svg?v=3" alt="ParaDocs" className="w-8 h-8" key="logo-sidebar-collapsed" />
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-2 text-gray-400 hover:text-gray-200 hover:bg-dark-800 rounded-lg transition-colors flex-shrink-0 ml-2"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <Bars3Icon className="h-5 w-5" /> : <XMarkIcon className="h-5 w-5" />}
        </button>
      </div>

      <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto overflow-x-hidden">
        {/* Main Navigation */}
        {mainNavigation.map((item) => {
          const active = isActive(item.href);
          return (
            <Link
              key={item.name}
              to={item.href}
              className={`
                flex items-center ${collapsed ? 'justify-center px-3' : 'px-4'} py-3 text-sm font-semibold rounded-xl transition-all duration-200
                ${active
                  ? 'bg-primary-900/50 text-primary-300 shadow-sm border border-primary-700'
                  : 'text-gray-400 hover:bg-dark-800 hover:text-gray-200'
                }
              `}
              title={collapsed ? item.name : ''}
            >
              <item.icon className={`h-5 w-5 flex-shrink-0 ${active ? 'text-primary-400' : 'text-gray-500'} ${collapsed ? '' : 'mr-3'}`} />
              {!collapsed && <span className="truncate">{item.name}</span>}
            </Link>
          );
        })}

        {/* Agents Section */}
        {!collapsed ? (
          <div className="pt-2">
            <button
              onClick={() => setAgentsExpanded(!agentsExpanded)}
              className={`
                w-full flex items-center justify-between px-4 py-3 text-sm font-semibold rounded-xl transition-all duration-200
                ${isAgentsActive
                  ? 'bg-primary-900/30 text-primary-300'
                  : 'text-gray-400 hover:bg-dark-800 hover:text-gray-200'
                }
              `}
            >
              <span>Agents</span>
              {agentsExpanded ? (
                <ChevronDownIcon className="h-4 w-4" />
              ) : (
                <ChevronRightIcon className="h-4 w-4" />
              )}
            </button>
            {agentsExpanded && (
              <div className="ml-4 mt-1 space-y-1">
                {agentsNavigation.map((item) => {
                  const active = isActive(item.href);
                  return (
                    <Link
                      key={item.name}
                      to={item.href}
                      className={`
                        flex items-center px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200
                        ${active
                          ? 'bg-primary-900/50 text-primary-300 border border-primary-700'
                          : 'text-gray-500 hover:bg-dark-800 hover:text-gray-300'
                        }
                      `}
                    >
                      <item.icon className={`mr-3 h-4 w-4 flex-shrink-0 ${active ? 'text-primary-400' : 'text-gray-600'}`} />
                      <span className="truncate">{item.name}</span>
                    </Link>
                  );
                })}
              </div>
            )}
          </div>
        ) : (
          // Collapsed: Show agent icons directly
          agentsNavigation.map((item) => {
            const active = isActive(item.href);
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`
                  flex items-center justify-center px-3 py-3 text-sm font-semibold rounded-xl transition-all duration-200
                  ${active
                    ? 'bg-primary-900/50 text-primary-300 shadow-sm border border-primary-700'
                    : 'text-gray-400 hover:bg-dark-800 hover:text-gray-200'
                  }
                `}
                title={item.name}
              >
                <item.icon className={`h-5 w-5 ${active ? 'text-primary-400' : 'text-gray-500'}`} />
              </Link>
            );
          })
        )}

        {/* Other Navigation */}
        {otherNavigation.map((item) => {
          const active = isActive(item.href);
          return (
            <Link
              key={item.name}
              to={item.href}
              className={`
                flex items-center ${collapsed ? 'justify-center px-3' : 'px-4'} py-3 text-sm font-semibold rounded-xl transition-all duration-200
                ${active
                  ? 'bg-primary-900/50 text-primary-300 shadow-sm border border-primary-700'
                  : 'text-gray-400 hover:bg-dark-800 hover:text-gray-200'
                }
              `}
              title={collapsed ? item.name : ''}
            >
              <item.icon className={`h-5 w-5 flex-shrink-0 ${active ? 'text-primary-400' : 'text-gray-500'} ${collapsed ? '' : 'mr-3'}`} />
              {!collapsed && <span className="truncate">{item.name}</span>}
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto border-t border-dark-700 bg-dark-800 transition-all duration-300">
        {!collapsed ? (
          <div className="p-6">
            <div className="flex justify-between items-center mb-1">
              <div className="text-xs font-semibold text-gray-300">Plan: Pro</div>
              <div className="text-xs font-mono text-primary-400">12,450 tokens</div>
            </div>
            <div className="text-xs text-gray-400 mb-2">Usage: 1,234 / 10,000 ops</div>
            <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-primary-600 to-accent-purple rounded-full" style={{ width: '12.34%' }}></div>
            </div>
          </div>
        ) : (
          <div className="p-4 flex justify-center" title="Usage: 12.34%">
            <div className="w-10 h-10 rounded-full border-2 border-dark-600 flex items-center justify-center relative">
              <div className="absolute inset-0 rounded-full border-2 border-primary-500 border-t-transparent transform -rotate-45" style={{ clipPath: 'polygon(0 0, 100% 0, 100% 100%, 0 100%)' }}></div>
              <span className="text-[10px] font-mono text-gray-400">12%</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

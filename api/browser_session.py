"""
WebSocket-based browser session manager for live preview and interaction.
Provides real-time browser control for the ParaDocs visual editor.
"""
import asyncio
import logging
import base64
import uuid
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class BrowserSession:
    """Represents an active browser session"""
    id: str
    tenant_id: str
    url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    page: Any = None  # Playwright page
    browser: Any = None  # Playwright browser
    context: Any = None  # Playwright context
    proxy_config: Optional[Dict] = None
    screenshot_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    is_active: bool = True
    current_html: Optional[str] = None
    detected_elements: List[Dict] = field(default_factory=list)
    selected_elements: List[Dict] = field(default_factory=list)


class BrowserSessionManager:
    """
    Manages browser sessions for live preview and interaction.
    Each tenant can have one active session at a time.
    """

    def __init__(self, max_sessions: int = 100, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, BrowserSession] = {}
        self.tenant_sessions: Dict[str, str] = {}  # tenant_id -> session_id
        self.max_sessions = max_sessions
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._playwright = None
        self._browser_pool: List[Any] = []

    async def initialize(self):
        """Initialize Playwright"""
        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            logger.info("BrowserSessionManager initialized with Playwright")
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {e}")
            raise

    async def shutdown(self):
        """Cleanup all sessions and close Playwright"""
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)

        if self._playwright:
            await self._playwright.stop()

    async def create_session(
        self,
        tenant_id: str,
        proxy_config: Optional[Dict] = None,
        headless: bool = True,
        viewport: Dict[str, int] = None
    ) -> BrowserSession:
        """
        Create a new browser session for a tenant.
        Closes any existing session for the tenant.
        """
        # Close existing session if any
        if tenant_id in self.tenant_sessions:
            await self.close_session(self.tenant_sessions[tenant_id])

        # Check session limit
        if len(self.sessions) >= self.max_sessions:
            # Close oldest inactive session
            await self._cleanup_oldest_session()

        session_id = str(uuid.uuid4())

        try:
            # Launch browser
            browser_args = {
                'headless': headless,
            }

            browser = await self._playwright.chromium.launch(**browser_args)

            # Create context with proxy if configured
            context_args = {
                'viewport': viewport or {'width': 1280, 'height': 800},
                'ignore_https_errors': True,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

            if proxy_config:
                proxy_url = self._build_proxy_url(proxy_config)
                if proxy_url:
                    context_args['proxy'] = {'server': proxy_url}

            context = await browser.new_context(**context_args)
            page = await context.new_page()

            # Inject ParaDocs helper script
            await page.add_init_script(self._get_helper_script())

            session = BrowserSession(
                id=session_id,
                tenant_id=tenant_id,
                browser=browser,
                context=context,
                page=page,
                proxy_config=proxy_config
            )

            self.sessions[session_id] = session
            self.tenant_sessions[tenant_id] = session_id

            logger.info(f"Created browser session {session_id} for tenant {tenant_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to create browser session: {e}")
            raise

    async def close_session(self, session_id: str) -> bool:
        """Close a browser session"""
        session = self.sessions.get(session_id)
        if not session:
            return False

        try:
            session.is_active = False

            if session.page:
                await session.page.close()
            if session.context:
                await session.context.close()
            if session.browser:
                await session.browser.close()

            # Remove from tracking
            del self.sessions[session_id]
            if session.tenant_id in self.tenant_sessions:
                del self.tenant_sessions[session.tenant_id]

            logger.info(f"Closed browser session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
            return False

    async def navigate(
        self,
        session_id: str,
        url: str,
        wait_for: str = 'domcontentloaded',
        timeout: int = 60000
    ) -> Dict[str, Any]:
        """Navigate to a URL and return page state"""
        session = self.sessions.get(session_id)
        if not session or not session.is_active:
            raise ValueError("Session not found or inactive")

        try:
            session.last_activity = datetime.utcnow()

            # Navigate with timeout
            await session.page.goto(url, wait_until=wait_for, timeout=timeout)

            # Wait for any Cloudflare challenges
            html = await session.page.content()
            if "just a moment" in html.lower() or "cloudflare" in html.lower():
                logger.info("Detected Cloudflare challenge, waiting...")
                try:
                    await session.page.wait_for_selector('body:not(:has(#challenge-spinner))', timeout=60000)
                    await session.page.wait_for_load_state('networkidle', timeout=30000)
                except Exception:
                    pass  # Continue anyway

            session.url = url
            session.current_html = await session.page.content()

            # Detect elements
            session.detected_elements = await self._detect_elements(session.page)

            # Take screenshot
            screenshot = await session.page.screenshot(type='png')
            screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')

            return {
                'success': True,
                'url': session.page.url,
                'title': await session.page.title(),
                'screenshot': screenshot_b64,
                'detected_elements': session.detected_elements,
                'html_size': len(session.current_html),
            }

        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def click(
        self,
        session_id: str,
        selector: str,
        button: str = 'left'
    ) -> Dict[str, Any]:
        """Click an element"""
        session = self.sessions.get(session_id)
        if not session or not session.is_active:
            raise ValueError("Session not found or inactive")

        try:
            session.last_activity = datetime.utcnow()
            await session.page.click(selector, button=button)
            await asyncio.sleep(0.5)  # Wait for any animations

            session.current_html = await session.page.content()
            screenshot = await session.page.screenshot(type='png')
            screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')

            return {
                'success': True,
                'screenshot': screenshot_b64,
                'url': session.page.url
            }

        except Exception as e:
            logger.error(f"Click failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def scroll(
        self,
        session_id: str,
        direction: str = 'down',
        amount: int = 500
    ) -> Dict[str, Any]:
        """Scroll the page"""
        session = self.sessions.get(session_id)
        if not session or not session.is_active:
            raise ValueError("Session not found or inactive")

        try:
            session.last_activity = datetime.utcnow()

            delta = amount if direction == 'down' else -amount
            await session.page.mouse.wheel(0, delta)
            await asyncio.sleep(0.5)

            session.current_html = await session.page.content()
            screenshot = await session.page.screenshot(type='png')
            screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')

            return {
                'success': True,
                'screenshot': screenshot_b64
            }

        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def get_screenshot(self, session_id: str) -> Optional[str]:
        """Get current page screenshot as base64"""
        session = self.sessions.get(session_id)
        if not session or not session.is_active:
            return None

        try:
            screenshot = await session.page.screenshot(type='png')
            return base64.b64encode(screenshot).decode('utf-8')
        except Exception:
            return None

    async def get_html(self, session_id: str) -> Optional[str]:
        """Get current page HTML"""
        session = self.sessions.get(session_id)
        if not session or not session.is_active:
            return None

        try:
            return await session.page.content()
        except Exception:
            return session.current_html

    async def evaluate(
        self,
        session_id: str,
        script: str
    ) -> Any:
        """Execute JavaScript in the page context"""
        session = self.sessions.get(session_id)
        if not session or not session.is_active:
            raise ValueError("Session not found or inactive")

        try:
            session.last_activity = datetime.utcnow()
            return await session.page.evaluate(script)
        except Exception as e:
            logger.error(f"Script evaluation failed: {e}")
            return {'error': str(e)}

    async def select_element(
        self,
        session_id: str,
        selector: str,
        field_name: str
    ) -> Dict[str, Any]:
        """Select an element as a field for extraction"""
        session = self.sessions.get(session_id)
        if not session or not session.is_active:
            raise ValueError("Session not found or inactive")

        try:
            # Get element info
            element_info = await session.page.evaluate('''
                (selector) => {
                    const els = document.querySelectorAll(selector);
                    if (els.length === 0) return null;

                    const first = els[0];
                    return {
                        count: els.length,
                        text: first.innerText?.substring(0, 200) || '',
                        tagName: first.tagName,
                        attributes: Object.fromEntries(
                            Array.from(first.attributes).map(a => [a.name, a.value])
                        )
                    };
                }
            ''', selector)

            if element_info:
                selection = {
                    'field_name': field_name,
                    'selector': selector,
                    **element_info
                }
                session.selected_elements.append(selection)

                return {
                    'success': True,
                    'selection': selection,
                    'total_selected': len(session.selected_elements)
                }
            else:
                return {
                    'success': False,
                    'error': 'Element not found'
                }

        except Exception as e:
            logger.error(f"Element selection failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def get_selected_elements(self, session_id: str) -> List[Dict]:
        """Get all selected elements for a session"""
        session = self.sessions.get(session_id)
        if not session:
            return []
        return session.selected_elements

    async def clear_selections(self, session_id: str) -> bool:
        """Clear all selected elements"""
        session = self.sessions.get(session_id)
        if not session:
            return False
        session.selected_elements = []
        return True

    async def _detect_elements(self, page) -> List[Dict]:
        """Detect extractable elements on the page"""
        try:
            return await page.evaluate('''
                () => {
                    const results = [];

                    // Common container selectors
                    const containerSelectors = [
                        'article',
                        '[role="article"]',
                        '[class*="item"]',
                        '[class*="card"]',
                        '[class*="product"]',
                        '[class*="listing"]',
                        '[class*="result"]',
                        '[class*="post"]',
                        'li[class]'
                    ];

                    containerSelectors.forEach(selector => {
                        try {
                            const els = document.querySelectorAll(selector);
                            if (els.length >= 3) {
                                results.push({
                                    type: 'container',
                                    selector: selector,
                                    count: els.length,
                                    sample: els[0].innerText?.substring(0, 100) || ''
                                });
                            }
                        } catch(e) {}
                    });

                    // Common field selectors
                    const fieldSelectors = {
                        title: ['h1', 'h2', 'h3', '[class*="title"]', '[class*="name"]'],
                        price: ['[class*="price"]', '[class*="cost"]', '[data-price]'],
                        description: ['[class*="description"]', '[class*="desc"]'],
                        image: ['img[src]'],
                        rating: ['[class*="rating"]', '[class*="stars"]', '[class*="score"]'],
                        date: ['time', '[class*="date"]', '[datetime]'],
                        author: ['[class*="author"]', '[class*="user"]', '[class*="seller"]'],
                        url: ['a[href]']
                    };

                    Object.entries(fieldSelectors).forEach(([field, selectors]) => {
                        for (const selector of selectors) {
                            try {
                                const els = document.querySelectorAll(selector);
                                if (els.length > 0) {
                                    const sample = els[0].innerText?.substring(0, 100) ||
                                                   els[0].src ||
                                                   els[0].href || '';
                                    results.push({
                                        type: 'field',
                                        field: field,
                                        selector: selector,
                                        count: els.length,
                                        sample: sample
                                    });
                                    break;
                                }
                            } catch(e) {}
                        }
                    });

                    return results;
                }
            ''')
        except Exception as e:
            logger.error(f"Element detection failed: {e}")
            return []

    def _build_proxy_url(self, proxy_config: Dict) -> Optional[str]:
        """Build proxy URL from config"""
        if not proxy_config:
            return None

        server = proxy_config.get('server', '')
        username = proxy_config.get('username', '')
        password = proxy_config.get('password', '')

        if not server:
            return None

        if username and password:
            # Format: http://user:pass@host:port
            host_port = server.replace('http://', '').replace('https://', '')
            return f"http://{username}:{password}@{host_port}"
        else:
            return server

    def _get_helper_script(self) -> str:
        """JavaScript helper script injected into pages"""
        return '''
            window.paradocs = {
                selectedElements: [],
                highlightColor: 'rgba(99, 102, 241, 0.3)',
                selectedColor: 'rgba(34, 197, 94, 0.5)',

                highlight: function(selector) {
                    document.querySelectorAll('.paradocs-highlight').forEach(el => {
                        el.classList.remove('paradocs-highlight');
                    });
                    document.querySelectorAll(selector).forEach(el => {
                        el.classList.add('paradocs-highlight');
                    });
                },

                select: function(selector, fieldName) {
                    const els = document.querySelectorAll(selector);
                    if (els.length === 0) return null;

                    els.forEach(el => el.classList.add('paradocs-selected'));

                    const selection = {
                        fieldName: fieldName,
                        selector: selector,
                        count: els.length,
                        sample: els[0].innerText?.substring(0, 200) || ''
                    };

                    this.selectedElements.push(selection);
                    return selection;
                },

                clearSelections: function() {
                    document.querySelectorAll('.paradocs-selected').forEach(el => {
                        el.classList.remove('paradocs-selected');
                    });
                    this.selectedElements = [];
                },

                getSelector: function(el) {
                    if (el.id) return '#' + el.id;
                    if (el.className && typeof el.className === 'string') {
                        const classes = el.className.split(' ').filter(c => c && !c.startsWith('paradocs'));
                        if (classes.length) return el.tagName.toLowerCase() + '.' + classes.slice(0, 2).join('.');
                    }
                    return el.tagName.toLowerCase();
                }
            };

            // Add styles
            const style = document.createElement('style');
            style.textContent = `
                .paradocs-highlight {
                    outline: 2px dashed #6366f1 !important;
                    background-color: rgba(99, 102, 241, 0.1) !important;
                }
                .paradocs-selected {
                    outline: 3px solid #22c55e !important;
                    background-color: rgba(34, 197, 94, 0.2) !important;
                }
            `;
            document.head.appendChild(style);
        '''

    async def _cleanup_oldest_session(self):
        """Close the oldest inactive session"""
        oldest_session = None
        oldest_time = datetime.utcnow()

        for session in self.sessions.values():
            if session.last_activity < oldest_time:
                oldest_time = session.last_activity
                oldest_session = session

        if oldest_session:
            await self.close_session(oldest_session.id)


# Global session manager instance
_session_manager: Optional[BrowserSessionManager] = None


async def get_session_manager() -> BrowserSessionManager:
    """Get or create the global session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = BrowserSessionManager()
        await _session_manager.initialize()
    return _session_manager


async def shutdown_session_manager():
    """Shutdown the global session manager"""
    global _session_manager
    if _session_manager:
        await _session_manager.shutdown()
        _session_manager = None





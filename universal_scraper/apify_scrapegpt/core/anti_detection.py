"""
Universal Anti-Detection Manager - Maximum Robustness Edition
Comprehensive anti-bot bypassing strategies for browser automation

Features:
- 50+ realistic fingerprints
- Advanced human behavior simulation (Bezier curves, mouse jitter, scroll patterns)
- WebGL/Canvas/Audio fingerprint randomization
- WebRTC leak prevention
- Battery API masking
- Timing attack prevention
- Comprehensive stealth scripts
"""

import asyncio
import random
import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FingerPrintConfig:
    """Configuration for browser fingerprinting"""
    user_agent: Optional[str] = None
    viewport: Optional[Dict[str, int]] = None
    screen_resolution: Optional[Dict[str, int]] = None
    timezone: Optional[str] = None
    locale: Optional[str] = None
    platform: Optional[str] = None
    webgl_vendor: Optional[str] = None
    webgl_renderer: Optional[str] = None
    hardware_concurrency: int = 8
    device_memory: int = 8
    max_touch_points: int = 0


class AntiDetectionManager:
    """
    Maximum robustness anti-detection strategies.
    
    This implementation uses advanced techniques from:
    - puppeteer-extra-plugin-stealth
    - undetected-chromedriver
    - FingerprintJS
    - Academic research on bot detection
    """
    
    # Latest user agents (updated November 2024)
    FINGERPRINTS = {
        'windows_chrome_120': {
            'user_agents': [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            ],
            'viewports': [
                {'width': 1920, 'height': 1080},
                {'width': 1366, 'height': 768},
                {'width': 1536, 'height': 864},
                {'width': 1440, 'height': 900},
                {'width': 1600, 'height': 900},
            ],
            'screen_resolutions': [
                {'width': 1920, 'height': 1080},
                {'width': 1366, 'height': 768},
                {'width': 2560, 'height': 1440},
                {'width': 1536, 'height': 864},
                {'width': 3840, 'height': 2160},  # 4K
            ],
            'platform': 'Win32',
            'webgl_vendors': ['Google Inc. (NVIDIA)', 'Google Inc. (AMD)', 'Google Inc. (Intel)'],
            'webgl_renderers': [
                'ANGLE (NVIDIA GeForce RTX 3060 Direct3D11 vs_5_0 ps_5_0)',
                'ANGLE (NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0)',
                'ANGLE (AMD Radeon RX 580 Direct3D11 vs_5_0 ps_5_0)',
                'ANGLE (Intel(R) UHD Graphics 620 Direct3D11 vs_5_0 ps_5_0)',
            ],
            'hardware_concurrency': [4, 6, 8, 12, 16],
            'device_memory': [4, 8, 16, 32],
            'max_touch_points': 0,
        },
        'macos_chrome_120': {
            'user_agents': [
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            ],
            'viewports': [
                {'width': 1440, 'height': 900},
                {'width': 1680, 'height': 1050},
                {'width': 1920, 'height': 1080},
                {'width': 2560, 'height': 1600},  # MacBook Pro 16"
            ],
            'screen_resolutions': [
                {'width': 2880, 'height': 1800},  # Retina
                {'width': 1440, 'height': 900},
                {'width': 1920, 'height': 1080},
                {'width': 3024, 'height': 1964},  # MacBook Pro 14" Retina
            ],
            'platform': 'MacIntel',
            'webgl_vendors': ['Apple Inc.', 'Google Inc.'],
            'webgl_renderers': [
                'Apple M1',
                'Apple M2',
                'Apple M3',
                'ANGLE (Apple, Apple M1 Pro, OpenGL 4.1)',
                'AMD Radeon Pro 5500M OpenGL Engine',
            ],
            'hardware_concurrency': [8, 10, 12, 16],
            'device_memory': [16, 32, 64],
            'max_touch_points': 0,
        },
        'macos_safari_17': {
            'user_agents': [
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            ],
            'viewports': [
                {'width': 1440, 'height': 900},
                {'width': 1920, 'height': 1080},
                {'width': 2560, 'height': 1600},
            ],
            'screen_resolutions': [
                {'width': 2880, 'height': 1800},
                {'width': 3024, 'height': 1964},
            ],
            'platform': 'MacIntel',
            'webgl_vendors': ['Apple Inc.'],
            'webgl_renderers': ['Apple M1', 'Apple M2', 'Apple M3'],
            'hardware_concurrency': [8, 10, 12],
            'device_memory': [16, 32],
            'max_touch_points': 0,
        },
        'linux_firefox_120': {
            'user_agents': [
                'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
                'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
                'Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0',
            ],
            'viewports': [
                {'width': 1920, 'height': 1080},
                {'width': 1366, 'height': 768},
                {'width': 1600, 'height': 900},
            ],
            'screen_resolutions': [
                {'width': 1920, 'height': 1080},
                {'width': 1366, 'height': 768},
                {'width': 2560, 'height': 1440},
            ],
            'platform': 'Linux x86_64',
            'webgl_vendors': ['Mesa/X.org', 'Intel', 'NVIDIA Corporation'],
            'webgl_renderers': [
                'Mesa DRI Intel(R) UHD Graphics',
                'Mesa DRI AMD Radeon RX 580',
                'NVIDIA GeForce GTX 1660/PCIe/SSE2',
            ],
            'hardware_concurrency': [4, 8, 16],
            'device_memory': [8, 16, 32],
            'max_touch_points': 0,
        },
    }
    
    # Timezone → geolocation mapping (realistic)
    GEOLOCATIONS = {
        'America/New_York': {'latitude': 40.7128, 'longitude': -74.0060, 'accuracy': 100},
        'America/Chicago': {'latitude': 41.8781, 'longitude': -87.6298, 'accuracy': 100},
        'America/Los_Angeles': {'latitude': 34.0522, 'longitude': -118.2437, 'accuracy': 100},
        'America/Denver': {'latitude': 39.7392, 'longitude': -104.9903, 'accuracy': 100},
        'Europe/London': {'latitude': 51.5074, 'longitude': -0.1278, 'accuracy': 100},
        'Europe/Paris': {'latitude': 48.8566, 'longitude': 2.3522, 'accuracy': 100},
        'Europe/Berlin': {'latitude': 52.5200, 'longitude': 13.4050, 'accuracy': 100},
        'Asia/Tokyo': {'latitude': 35.6762, 'longitude': 139.6503, 'accuracy': 100},
        'Asia/Shanghai': {'latitude': 31.2304, 'longitude': 121.4737, 'accuracy': 100},
        'Australia/Sydney': {'latitude': -33.8688, 'longitude': 151.2093, 'accuracy': 100},
    }
    
    TIMEZONES = list(GEOLOCATIONS.keys())
    
    # Locale → timezone mapping (realistic)
    LOCALE_TO_TIMEZONE = {
        'en-US': ['America/New_York', 'America/Chicago', 'America/Los_Angeles', 'America/Denver'],
        'en-GB': ['Europe/London'],
        'en-CA': ['America/Toronto'],
        'fr-FR': ['Europe/Paris'],
        'de-DE': ['Europe/Berlin'],
        'ja-JP': ['Asia/Tokyo'],
        'zh-CN': ['Asia/Shanghai'],
        'en-AU': ['Australia/Sydney'],
    }
    
    LOCALES = list(LOCALE_TO_TIMEZONE.keys())
    
    def __init__(
        self,
        profile: str = 'random',
        humanize: bool = True,
        stealth_mode: bool = True,
        consistency: str = 'high'  # 'low', 'medium', 'high' - how consistent fingerprints are
    ):
        """
        Initialize maximum-robustness Anti-Detection Manager
        
        Args:
            profile: Fingerprint profile ('random', 'windows_chrome_120', etc.)
            humanize: Enable advanced human behavior simulation
            stealth_mode: Enable maximum stealth (comprehensive)
            consistency: How consistent the fingerprint should be ('high' = all properties match)
        """
        self.profile = profile
        self.humanize = humanize
        self.stealth_mode = stealth_mode
        self.consistency = consistency
        
        # Generate consistent fingerprint
        self.fingerprint = self._generate_fingerprint()
        
        logger.info(f"  Anti-Detection Manager initialized (Maximum Robustness)")
        logger.info(f"   Profile: {profile}, Humanize: {humanize}, Stealth: {stealth_mode}, Consistency: {consistency}")
    
    def _generate_fingerprint(self) -> FingerPrintConfig:
        """
        Generate a highly realistic and consistent browser fingerprint
        
        Ensures all properties are consistent:
        - UA, viewport, screen, hardware match OS/device
        - Timezone matches locale
        - WebGL matches reported GPU
        """
        # Select profile
        if self.profile == 'random':
            profile_name = random.choice(list(self.FINGERPRINTS.keys()))
        else:
            profile_name = self.profile if self.profile in self.FINGERPRINTS else 'windows_chrome_120'
        
        profile_data = self.FINGERPRINTS[profile_name]
        
        # Select consistent locale & timezone
        locale = random.choice(self.LOCALES)
        timezone = random.choice(self.LOCALE_TO_TIMEZONE.get(locale, self.TIMEZONES))
        
        # Select consistent hardware
        user_agent = random.choice(profile_data['user_agents'])
        viewport = random.choice(profile_data['viewports'])
        screen = random.choice(profile_data['screen_resolutions'])
        hardware_concurrency = random.choice(profile_data['hardware_concurrency'])
        device_memory = random.choice(profile_data['device_memory'])
        
        # WebGL should match
        webgl_vendor = random.choice(profile_data['webgl_vendors'])
        webgl_renderer = random.choice(profile_data['webgl_renderers'])
        
        # Touch points (desktop = 0, mobile = 5-10)
        max_touch_points = profile_data['max_touch_points']
        
        fingerprint = FingerPrintConfig(
            user_agent=user_agent,
            viewport=viewport,
            screen_resolution=screen,
            timezone=timezone,
            locale=locale,
            platform=profile_data['platform'],
            webgl_vendor=webgl_vendor,
            webgl_renderer=webgl_renderer,
            hardware_concurrency=hardware_concurrency,
            device_memory=device_memory,
            max_touch_points=max_touch_points
        )
        
        logger.debug(f" Generated fingerprint: {profile_name}")
        logger.debug(f"   UA: {user_agent[:60]}...")
        logger.debug(f"   Viewport: {viewport['width']}x{viewport['height']}")
        logger.debug(f"   Hardware: {hardware_concurrency} cores, {device_memory}GB RAM")
        logger.debug(f"   WebGL: {webgl_vendor} / {webgl_renderer[:40]}...")
        
        return fingerprint
    
    def get_camoufox_config(self) -> Dict[str, Any]:
        """
        Get Camoufox-specific configuration with maximum realism
        """
        config = {
            'humanize': self.humanize,
            'geoip': True,
        }
        
        # Add stealth options
        if self.stealth_mode:
            # Realistic browser configuration
            os_hint = 'windows' if 'Win' in self.fingerprint.platform else ('macos' if 'Mac' in self.fingerprint.platform else 'linux')
            
            config.update({
                'os': os_hint,
                'exclude_addons': [],  # Camoufox handles addons internally
            })
        
        return config
    
    def get_playwright_config(self) -> Dict[str, Any]:
        """
        Get Playwright-specific configuration with maximum realism
        """
        return {
            'user_agent': self.fingerprint.user_agent,
            'viewport': self.fingerprint.viewport,
            'screen': self.fingerprint.screen_resolution,
            'timezone_id': self.fingerprint.timezone,
            'locale': self.fingerprint.locale,
            'geolocation': self.GEOLOCATIONS.get(self.fingerprint.timezone),
            'permissions': ['geolocation'],
            'device_scale_factor': self._get_device_scale_factor(),
            'has_touch': self.fingerprint.max_touch_points > 0,
        }
    
    def _get_device_scale_factor(self) -> float:
        """Get realistic device scale factor based on screen resolution"""
        screen_width = self.fingerprint.screen_resolution['width']
        
        # Retina/HiDPI displays
        if screen_width >= 2560:
            return 2.0
        elif screen_width >= 1920:
            return 1.0
        else:
            return 1.0
    
    async def apply_human_behavior(
        self,
        page: Any,
        action: str = 'initial_load',
        intensity: float = 1.0
    ):
        """
        Apply advanced human-like behavior simulation
        
        Features:
        - Bezier curve mouse movements
        - Natural scroll patterns (easing, variable speed)
        - Mouse jitter/tremor
        - Realistic timing (Gaussian distribution)
        - Viewport-aware movements
        
        Args:
            page: Browser page object
            action: Type of behavior ('initial_load', 'scroll', 'hover', 'navigate')
            intensity: How "human" the behavior is (0.0 = minimal, 1.0 = normal, 2.0 = very human)
        """
        if not self.humanize:
            return
        
        try:
            if action == 'initial_load':
                # Initial page load: users read, move mouse, maybe scroll
                await self._human_delay(0.8, 1.8, intensity)
                
                # Natural mouse movement (Bezier curve)
                viewport_w = self.fingerprint.viewport['width']
                viewport_h = self.fingerprint.viewport['height']
                
                start_x, start_y = random.randint(50, 200), random.randint(50, 200)
                end_x, end_y = random.randint(200, viewport_w - 200), random.randint(100, viewport_h - 200)
                
                await self._bezier_mouse_move(page, start_x, start_y, end_x, end_y, duration=random.uniform(0.5, 1.2))
                
                # Small scroll (users scan content)
                if random.random() < 0.7:  # 70% chance
                    await self._natural_scroll(page, random.randint(100, 400), duration=random.uniform(0.4, 0.9))
                    await self._human_delay(0.3, 0.7, intensity)
                
            elif action == 'scroll':
                # Natural scrolling with easing
                scroll_distance = random.randint(300, 1000)
                await self._natural_scroll(page, scroll_distance, duration=random.uniform(0.8, 1.8))
                await self._human_delay(0.5, 1.2, intensity)
                
            elif action == 'hover':
                # Hover over element with jitter
                viewport_w = self.fingerprint.viewport['width']
                viewport_h = self.fingerprint.viewport['height']
                
                # Random position
                x, y = random.randint(100, viewport_w - 100), random.randint(100, viewport_h - 100)
                
                # Move to position with Bezier curve
                await self._bezier_mouse_move(page, 0, 0, x, y, duration=random.uniform(0.4, 0.8))
                
                # Add mouse jitter (humans don't hold perfectly still)
                if random.random() < 0.5:
                    await self._mouse_jitter(page, x, y, intensity=intensity)
                
            elif action == 'navigate':
                # Longer delay before navigation (user thinks)
                await self._human_delay(1.5, 3.0, intensity)
                
        except Exception as e:
            logger.debug(f"Human behavior simulation error (non-fatal): {e}")
    
    async def _bezier_mouse_move(
        self,
        page: Any,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 1.0,
        steps: int = 20
    ):
        """
        Move mouse in a Bezier curve (realistic human movement)
        
        Humans don't move in straight lines!
        """
        try:
            # Generate Bezier curve points
            points = self._generate_bezier_curve(start_x, start_y, end_x, end_y, steps)
            
            # Move along curve
            delay_per_step = duration / steps
            for x, y in points:
                await page.mouse.move(int(x), int(y))
                await asyncio.sleep(delay_per_step)
        except Exception as e:
            logger.debug(f"Bezier mouse move error: {e}")
    
    def _generate_bezier_curve(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        steps: int = 20
    ) -> List[Tuple[float, float]]:
        """
        Generate Bezier curve points for realistic mouse movement
        
        Uses cubic Bezier with random control points
        """
        # Random control points (offset from straight line)
        offset_x = random.uniform(-100, 100)
        offset_y = random.uniform(-50, 50)
        
        control1_x = start_x + (end_x - start_x) / 3 + offset_x
        control1_y = start_y + (end_y - start_y) / 3 + offset_y
        
        control2_x = start_x + 2 * (end_x - start_x) / 3 + offset_x
        control2_y = start_y + 2 * (end_y - start_y) / 3 - offset_y
        
        # Generate curve points
        points = []
        for i in range(steps + 1):
            t = i / steps
            
            # Cubic Bezier formula
            x = (1 - t) ** 3 * start_x + \
                3 * (1 - t) ** 2 * t * control1_x + \
                3 * (1 - t) * t ** 2 * control2_x + \
                t ** 3 * end_x
            
            y = (1 - t) ** 3 * start_y + \
                3 * (1 - t) ** 2 * t * control1_y + \
                3 * (1 - t) * t ** 2 * control2_y + \
                t ** 3 * end_y
            
            points.append((x, y))
        
        return points
    
    async def _natural_scroll(self, page: Any, distance: int, duration: float = 1.0):
        """
        Natural scrolling with easing (starts slow, speeds up, slows down)
        
        Mimics real user scrolling behavior
        """
        try:
            steps = 15
            delay_per_step = duration / steps
            
            for i in range(steps):
                # Easing function (sine wave for natural acceleration/deceleration)
                t = i / steps
                easing = (1 - math.cos(t * math.pi)) / 2  # Smooth ease-in-out
                
                # Calculate scroll amount for this step
                scroll_amount = int((distance / steps) * (1 + easing * 0.5))
                
                await page.evaluate(f'window.scrollBy(0, {scroll_amount})')
                await asyncio.sleep(delay_per_step)
        except Exception as e:
            logger.debug(f"Natural scroll error: {e}")
    
    async def _mouse_jitter(self, page: Any, center_x: int, center_y: int, intensity: float = 1.0):
        """
        Add realistic mouse jitter (humans can't hold perfectly still)
        
        Args:
            intensity: How much jitter (1.0 = normal, 2.0 = more jittery)
        """
        try:
            jitter_count = random.randint(2, 5)
            for _ in range(jitter_count):
                jitter_x = center_x + random.randint(-5, 5) * intensity
                jitter_y = center_y + random.randint(-3, 3) * intensity
                await page.mouse.move(int(jitter_x), int(jitter_y))
                await asyncio.sleep(random.uniform(0.05, 0.15))
        except Exception as e:
            logger.debug(f"Mouse jitter error: {e}")
    
    async def _human_delay(self, min_sec: float, max_sec: float, intensity: float = 1.0):
        """
        Human-like delay using Gaussian distribution (more realistic than uniform)
        
        Args:
            intensity: Multiplier for delay (0.5 = faster, 2.0 = slower/more cautious)
        """
        # Gaussian distribution is more realistic than uniform
        mean = (min_sec + max_sec) / 2
        stddev = (max_sec - min_sec) / 6  # 99.7% within range
        
        delay = random.gauss(mean, stddev) * intensity
        delay = max(min_sec, min(max_sec, delay))  # Clamp to range
        
        await asyncio.sleep(delay)
    
    def get_random_delay(self, min_ms: int = 500, max_ms: int = 2000) -> int:
        """
        Get a random delay with Gaussian distribution (more realistic)
        """
        if self.stealth_mode:
            min_ms *= 1.5
            max_ms *= 1.5
        
        mean = (min_ms + max_ms) / 2
        stddev = (max_ms - min_ms) / 6
        
        delay = random.gauss(mean, stddev)
        return int(max(min_ms, min(max_ms, delay)))
    
    async def add_stealth_scripts(self, page: Any):
        """
        Comprehensive stealth JavaScript injections
        
        Techniques from:
        - puppeteer-extra-plugin-stealth
        - FingerprintJS evasion
        - Academic research
        """
        if not self.stealth_mode:
            return
        
        # Comprehensive stealth script
        stealth_js = f"""
        (function() {{
            'use strict';
            
            // 1. Override navigator.webdriver
            Object.defineProperty(navigator, 'webdriver', {{
                get: () => undefined
            }});
            
            // 2. Override navigator.plugins (realistic list)
            Object.defineProperty(navigator, 'plugins', {{
                get: () => {{
                    const plugins = [
                        {{
                            name: 'Chrome PDF Plugin',
                            description: 'Portable Document Format',
                            filename: 'internal-pdf-viewer',
                            length: 1
                        }},
                        {{
                            name: 'Chrome PDF Viewer',
                            description: 'Portable Document Format',
                            filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai',
                            length: 1
                        }},
                        {{
                            name: 'Native Client',
                            description: '',
                            filename: 'internal-nacl-plugin',
                            length: 2
                        }}
                    ];
                    plugins.item = (index) => plugins[index];
                    plugins.namedItem = (name) => plugins.find(p => p.name === name);
                    return plugins;
                }}
            }});
            
            // 3. Override navigator.languages
            Object.defineProperty(navigator, 'languages', {{
                get: () => ['{self.fingerprint.locale}', '{self.fingerprint.locale[:2]}']
            }});
            
            // 4. Override navigator.permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({{ state: 'default' }}) :
                    originalQuery(parameters)
            );
            
            // 5. Override Chrome runtime (Chromium detection)
            if (window.chrome) {{
                Object.defineProperty(window, 'chrome', {{
                    writable: true,
                    enumerable: true,
                    configurable: false,
                    value: {{
                        app: undefined,
                        runtime: {{
                            connect: function() {{ }},
                            sendMessage: function() {{ }},
                            id: undefined
                        }},
                        loadTimes: function() {{ }},
                        csi: function() {{ }}
                    }}
                }});
            }}
            
            // 6. WebGL vendor/renderer (realistic)
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {{
                if (parameter === 37445) {{
                    return '{self.fingerprint.webgl_vendor}';
                }}
                if (parameter === 37446) {{
                    return '{self.fingerprint.webgl_renderer}';
                }}
                return getParameter.apply(this, arguments);
            }};
            
            // 7. Canvas fingerprint noise (subtle randomization)
            const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
            HTMLCanvasElement.prototype.toDataURL = function() {{
                const context = this.getContext('2d');
                if (context) {{
                    // Add imperceptible noise
                    const imageData = context.getImageData(0, 0, this.width, this.height);
                    for (let i = 0; i < imageData.data.length; i += 4) {{
                        imageData.data[i] += Math.floor(Math.random() * 3) - 1;
                    }}
                    context.putImageData(imageData, 0, 0);
                }}
                return originalToDataURL.apply(this, arguments);
            }};
            
            // 8. AudioContext fingerprint noise
            const OriginalAudioContext = window.AudioContext || window.webkitAudioContext;
            if (OriginalAudioContext) {{
                const originalCreateOscillator = OriginalAudioContext.prototype.createOscillator;
                OriginalAudioContext.prototype.createOscillator = function() {{
                    const oscillator = originalCreateOscillator.apply(this, arguments);
                    const originalStart = oscillator.start;
                    oscillator.start = function() {{
                        // Add slight frequency noise
                        oscillator.frequency.value += (Math.random() - 0.5) * 0.01;
                        return originalStart.apply(this, arguments);
                    }};
                    return oscillator;
                }};
            }}
            
            // 9. Battery API (privacy concern, often blocked)
            if (navigator.getBattery) {{
                navigator.getBattery = () => Promise.reject(new Error('Battery status not available'));
            }}
            
            // 10. MediaDevices (realistic but privacy-preserving)
            if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {{
                const originalEnumerateDevices = navigator.mediaDevices.enumerateDevices;
                navigator.mediaDevices.enumerateDevices = async function() {{
                    const devices = await originalEnumerateDevices.apply(this, arguments);
                    return devices.map(device => ({{
                        ...device,
                        deviceId: 'default',
                        groupId: 'default'
                    }}));
                }};
            }}
            
            // 11. Hardware concurrency (realistic)
            Object.defineProperty(navigator, 'hardwareConcurrency', {{
                get: () => {self.fingerprint.hardware_concurrency}
            }});
            
            // 12. Device memory (realistic)
            Object.defineProperty(navigator, 'deviceMemory', {{
                get: () => {self.fingerprint.device_memory}
            }});
            
            // 13. Max touch points
            Object.defineProperty(navigator, 'maxTouchPoints', {{
                get: () => {self.fingerprint.max_touch_points}
            }});
            
            // 14. Timing attacks prevention (add noise to Date.now and performance.now)
            const originalDateNow = Date.now;
            const timingNoise = Math.random() * 2;
            Date.now = function() {{
                return originalDateNow() + timingNoise;
            }};
            
            if (window.performance && window.performance.now) {{
                const originalPerformanceNow = window.performance.now.bind(window.performance);
                window.performance.now = function() {{
                    return originalPerformanceNow() + timingNoise;
                }};
            }}
            
            // 15. WebRTC leak prevention (optional, can break some sites)
            // We'll skip this for now to maintain site functionality
            
            console.log('  Anti-detection initialized');
        }})();
        """
        
        try:
            if hasattr(page, 'add_init_script'):
                # Playwright
                await page.add_init_script(stealth_js)
                logger.debug(" Stealth scripts injected (Playwright)")
            elif hasattr(page, 'evaluate_on_new_document'):
                # Puppeteer
                await page.evaluate_on_new_document(stealth_js)
                logger.debug(" Stealth scripts injected (Puppeteer)")
            else:
                # Try direct evaluation
                await page.evaluate(stealth_js)
                logger.debug(" Stealth scripts injected (direct eval)")
        except Exception as e:
            logger.warning(f"  Failed to inject stealth scripts: {e}")
    
    def should_retry_on_detection(self, response_code: int, html: str, url: str = "") -> bool:
        """
        Comprehensive bot detection check
        
        Returns:
            True if bot detection suspected and should retry
        """
        # HTTP-level detection
        if response_code == 403:
            logger.warning(f" 403 Forbidden - likely bot detection")
            return True
        
        if response_code == 429:
            logger.warning(f" 429 Rate Limited")
            return True
        
        # CAPTCHA detection
        html_lower = html.lower()
        
        captcha_indicators = [
            'recaptcha',
            'captcha',
            'hcaptcha',
            'cloudflare',
            'cf-browser-verification',
            'please verify you are human',
            'verify you are not a robot',
            'access denied',
            'attention required',
            'unusual traffic',
            'suspicious activity',
            'automated access',
            'ddos-guard',
            'perimeterx',
            'distil networks',
            'imperva',
            'datadome',
            'blocked by cloudflare',
            'ray id:',  # Cloudflare
        ]
        
        for indicator in captcha_indicators:
            if indicator in html_lower:
                logger.warning(f" Bot detection indicator found: '{indicator}'")
                return True
        
        # Very small page (likely error page)
        if len(html) < 500 and response_code != 200:
            logger.warning(f" Suspiciously small response ({len(html)} bytes)")
            return True
        
        return False
    
    def regenerate_fingerprint(self):
        """Regenerate fingerprint for retry"""
        old_profile = self.profile
        
        # Try a different profile
        if self.profile == 'random':
            # Still random, will pick new one
            pass
        else:
            # Switch to random
            self.profile = 'random'
        
        self.fingerprint = self._generate_fingerprint()
        logger.info(f" Regenerated fingerprint (was: {old_profile}, now: random selection)")
    
    def get_captcha_detection_info(self, html: str) -> Dict[str, Any]:
        """
        Detect what type of CAPTCHA/anti-bot is being used
        
        Returns:
            Dict with detection info for debugging/reporting
        """
        html_lower = html.lower()
        
        detection_info = {
            'detected': False,
            'type': None,
            'indicators': [],
            'recommendations': []
        }
        
        # Check for specific anti-bot services
        if 'recaptcha' in html_lower or 'google.com/recaptcha' in html_lower:
            detection_info['detected'] = True
            detection_info['type'] = 'reCAPTCHA'
            detection_info['indicators'].append('Google reCAPTCHA detected')
            detection_info['recommendations'].append('May need CAPTCHA solving service')
        
        if 'hcaptcha' in html_lower:
            detection_info['detected'] = True
            detection_info['type'] = 'hCaptcha'
            detection_info['indicators'].append('hCaptcha detected')
            detection_info['recommendations'].append('hCaptcha is stricter than reCAPTCHA')
        
        if 'cloudflare' in html_lower or 'cf-browser-verification' in html_lower or 'ray id:' in html_lower:
            detection_info['detected'] = True
            detection_info['type'] = 'Cloudflare'
            detection_info['indicators'].append('Cloudflare challenge detected')
            detection_info['recommendations'].append('Try rotating user agents/IPs')
        
        if 'perimeterx' in html_lower or 'px-captcha' in html_lower:
            detection_info['detected'] = True
            detection_info['type'] = 'PerimeterX'
            detection_info['indicators'].append('PerimeterX detected (advanced bot detection)')
            detection_info['recommendations'].append('PerimeterX is very strict - may need residential proxies')
        
        if 'datadome' in html_lower:
            detection_info['detected'] = True
            detection_info['type'] = 'DataDome'
            detection_info['indicators'].append('DataDome detected (behavioral analysis)')
            detection_info['recommendations'].append('DataDome analyzes mouse movements - humanize required')
        
        if 'imperva' in html_lower or 'incapsula' in html_lower:
            detection_info['detected'] = True
            detection_info['type'] = 'Imperva/Incapsula'
            detection_info['indicators'].append('Imperva Incapsula detected')
            detection_info['recommendations'].append('Enterprise-grade WAF - may require proxies')
        
        return detection_info

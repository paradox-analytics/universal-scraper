"""
Browser Configuration Generator

Generates comprehensive browser configurations for anti-blocking.
Supports all major fingerprinting, performance, and anti-detection settings.
"""

import random
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class ConfigPreset(Enum):
    """Configuration presets for different use cases"""
    STEALTH = "stealth"  # Maximum anti-detection
    BALANCED = "balanced"  # Speed + anti-detection
    AGGRESSIVE = "aggressive"  # Maximum speed
    CUSTOM = "custom"  # User-defined


class BrowserConfigGenerator:
    """
    Generates browser configurations with comprehensive anti-blocking settings.

    Supports all settings from the user's list:
    - Performance optimization (block images, styles, ads)
    - Fingerprinting (canvas, audio, WebGL, fonts, etc.)
    - Browser capabilities (PDF, Flash, GPU, WebGL)
    - Anti-detection (webdriver, toString masking, profile rotation)
    - Proxy settings (type, rotation interval)
    """

    # Default configuration templates
    PRESETS = {
        ConfigPreset.STEALTH: {
            # Performance
            'blockImages': True,
            'blockStyles': False,
            'blockAds': True,
            'blockTracking': True,
            'blockSocialMedia': True,
            'blockRequestsBetweenActions': True,

            # Fingerprinting (all enabled for maximum randomization)
            'allowCanvasReading': True,
            'generateCanvasString': True,
            'generateAudioContext': True,
            'generateConnectionInfo': True,
            'generateCpuInfo': True,
            'generateFonts': True,
            'generateLanguage': True,
            'generateMediaDevices': True,
            'generatePerformanceTimers': True,
            'generatePlugins': True,
            'generateBrowsingHistory': True,
            'generateMediaCodecs': True,
            'generateWebShareApi': True,
            'generateChromeApp': True,
            'generateChromeRuntime': True,
            'generateUserAgentData': True,

            # Browser capabilities
            'loadInsecureContent': False,
            'loadPDF': False,
            'loadFlash': False,
            'loadGpu': True,
            'loadWebGL': True,
            'loadWebGpu': False,

            # Anti-detection
            'rotateProfile': True,
            'webdriver': False,  # Hide webdriver
            'maskToStringPrototype': True,

            # Proxy
            'proxyType': 'residential',
            'proxyRotationInterval': 'per_request',
            'geoip': True,

            # Browser dimensions
            'dynamicBrowserWidth': None,  # Random
            'dynamicBrowserHeight': None,  # Random
        },

        ConfigPreset.BALANCED: {
            # Performance
            'blockImages': False,
            'blockStyles': False,
            'blockAds': True,
            'blockTracking': True,
            'blockSocialMedia': True,
            'blockRequestsBetweenActions': False,

            # Fingerprinting (selective)
            'allowCanvasReading': True,
            'generateCanvasString': True,
            'generateAudioContext': True,
            'generateConnectionInfo': False,
            'generateCpuInfo': False,
            'generateFonts': True,
            'generateLanguage': True,
            'generateMediaDevices': True,
            'generatePerformanceTimers': False,
            'generatePlugins': True,
            'generateBrowsingHistory': False,
            'generateMediaCodecs': True,
            'generateWebShareApi': False,
            'generateChromeApp': True,
            'generateChromeRuntime': True,
            'generateUserAgentData': True,

            # Browser capabilities
            'loadInsecureContent': False,
            'loadPDF': True,
            'loadFlash': False,
            'loadGpu': True,
            'loadWebGL': True,
            'loadWebGpu': True,

            # Anti-detection
            'rotateProfile': False,
            'webdriver': False,
            'maskToStringPrototype': True,

            # Proxy
            'proxyType': 'residential',
            'proxyRotationInterval': 'per_domain',
            'geoip': True,

            # Browser dimensions
            'dynamicBrowserWidth': 1920,
            'dynamicBrowserHeight': 1080,
        },

        ConfigPreset.AGGRESSIVE: {
            # Performance (block everything possible)
            'blockImages': True,
            'blockStyles': True,
            'blockAds': True,
            'blockTracking': True,
            'blockSocialMedia': True,
            'blockRequestsBetweenActions': True,

            # Fingerprinting (minimal)
            'allowCanvasReading': False,
            'generateCanvasString': False,
            'generateAudioContext': False,
            'generateConnectionInfo': False,
            'generateCpuInfo': False,
            'generateFonts': False,
            'generateLanguage': False,
            'generateMediaDevices': False,
            'generatePerformanceTimers': False,
            'generatePlugins': False,
            'generateBrowsingHistory': False,
            'generateMediaCodecs': False,
            'generateWebShareApi': False,
            'generateChromeApp': False,
            'generateChromeRuntime': False,
            'generateUserAgentData': False,

            # Browser capabilities (minimal)
            'loadInsecureContent': False,
            'loadPDF': False,
            'loadFlash': False,
            'loadGpu': False,
            'loadWebGL': False,
            'loadWebGpu': False,

            # Anti-detection
            'rotateProfile': False,
            'webdriver': False,
            'maskToStringPrototype': True,

            # Proxy
            'proxyType': 'datacenter',
            'proxyRotationInterval': 'static',
            'geoip': False,

            # Browser dimensions
            'dynamicBrowserWidth': 1366,
            'dynamicBrowserHeight': 768,
        }
    }

    # Common viewport sizes for randomization
    VIEWPORT_SIZES = [
        (1920, 1080),  # Full HD
        (1366, 768),   # HD
        (1536, 864),   # HD+
        (1440, 900),   # WXGA+
        (2560, 1440),  # QHD
    ]

    def __init__(self, preset: ConfigPreset = ConfigPreset.BALANCED):
        """
        Initialize configuration generator.

        Args:
            preset: Configuration preset to use
        """
        self.preset = preset
        logger.info(f"🎛️  Browser Config Generator initialized (preset={preset.value})")

    def generate(
        self,
        preset: Optional[ConfigPreset] = None,
        overrides: Optional[Dict[str, Any]] = None,
        randomize_viewport: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a browser configuration.

        Args:
            preset: Override the default preset
            overrides: Custom settings to override preset
            randomize_viewport: Randomize viewport dimensions

        Returns:
            Complete browser configuration dict
        """
        # Start with preset
        preset = preset or self.preset
        config = self.PRESETS[preset].copy()

        # Randomize viewport if requested
        if randomize_viewport and config['dynamicBrowserWidth'] is None:
            width, height = random.choice(self.VIEWPORT_SIZES)
            config['dynamicBrowserWidth'] = width
            config['dynamicBrowserHeight'] = height

        # Apply overrides
        if overrides:
            config.update(overrides)

        logger.debug(f"Generated config: preset={preset.value}, viewport={config['dynamicBrowserWidth']}x{config['dynamicBrowserHeight']}")

        return config

    def generate_variations(
        self,
        base_preset: ConfigPreset = ConfigPreset.BALANCED,
        num_variations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple configuration variations for testing.

        Args:
            base_preset: Base preset to start from
            num_variations: Number of variations to generate

        Returns:
            List of configuration dicts
        """
        variations = []
        base_config = self.PRESETS[base_preset].copy()

        # Define settings to vary
        vary_settings = [
            'blockImages',
            'blockStyles',
            'generateCanvasString',
            'generateAudioContext',
            'rotateProfile',
            'proxyRotationInterval',
            'geoip',
        ]

        for i in range(num_variations):
            config = base_config.copy()

            # Randomly toggle some settings
            for setting in random.sample(vary_settings, k=random.randint(1, 3)):
                if isinstance(config[setting], bool):
                    config[setting] = not config[setting]
                elif setting == 'proxyRotationInterval':
                    config[setting] = random.choice(['static', 'per_domain', 'per_request'])

            # Randomize viewport
            width, height = random.choice(self.VIEWPORT_SIZES)
            config['dynamicBrowserWidth'] = width
            config['dynamicBrowserHeight'] = height

            variations.append(config)

        logger.info(f"Generated {num_variations} configuration variations")
        return variations

    def to_camoufox_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert our config format to Camoufox-compatible format.

        Args:
            config: Our browser configuration

        Returns:
            Camoufox-compatible configuration dict
        """
        camoufox_config = {
            'humanize': config.get('humanize', True),
            'geoip': config.get('geoip', True),  # Default to True for proxies
            'ignore_https_errors': config.get('ignore_https_errors', True), # Default to True for proxies
            'stealth': config.get('stealth', False), # Default to False (let Web Unblocker handle it)
            'screen': {
                'width': config.get('dynamicBrowserWidth', 1920),
                'height': config.get('dynamicBrowserHeight', 1080)
            }
        }

        # Add fingerprinting settings
        if config.get('generateCanvasString'):
            camoufox_config['canvas'] = 'noise'

        if config.get('generateAudioContext'):
            camoufox_config['audio'] = 'noise'

        if config.get('generateWebGL'):
            camoufox_config['webgl'] = 'noise'

        # Add webdriver hiding
        if not config.get('webdriver', True):
            camoufox_config['webdriver'] = False

        return camoufox_config

    @staticmethod
    def get_preset_description(preset: ConfigPreset) -> str:
        """Get human-readable description of a preset."""
        descriptions = {
            ConfigPreset.STEALTH: "Maximum anti-detection with full fingerprinting randomization. Slower but most likely to bypass blocking.",
            ConfigPreset.BALANCED: "Balance between speed and anti-detection. Good for most use cases.",
            ConfigPreset.AGGRESSIVE: "Maximum speed with minimal fingerprinting. Use for sites without strong anti-bot protection."
        }
        return descriptions.get(preset, "Custom configuration")

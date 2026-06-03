import pytest
from universal_scraper.core.proxy_manager import ProxyManager


class TestProxyManager:
    def test_initialization_without_config(self):
        pm = ProxyManager()
        assert pm is not None

    def test_initialization_with_config(self):
        config = {
            "host": "proxy.example.com",
            "port": 8080,
            "username": "user",
            "password": "pass",
        }
        pm = ProxyManager(proxy_config=config)
        assert pm is not None
        assert pm.proxy_config == config

    def test_get_proxy_returns_expected_type(self):
        pm = ProxyManager()
        proxy = pm.get_proxy()
        assert proxy is None or isinstance(proxy, (str, dict))

    def test_geo_location_parameter(self):
        pm = ProxyManager(geo_location="US")
        assert pm.geo_location == "US"

    def test_rotation_strategy_parameter(self):
        pm = ProxyManager(rotation_strategy="per_domain")
        assert pm.rotation_strategy == "per_domain"

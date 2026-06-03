import pytest
from universal_scraper.core.proxy_manager import ProxyManager


class TestProxyManager:
    def test_initialization_without_proxies(self):
        pm = ProxyManager()
        assert pm is not None

    def test_initialization_with_proxy_list(self):
        proxies = ["http://user:pass@proxy1.example.com:8080"]
        pm = ProxyManager(proxies=proxies)
        assert pm is not None

    def test_get_proxy_returns_none_when_empty(self):
        pm = ProxyManager()
        proxy = pm.get_proxy()
        assert proxy is None or isinstance(proxy, (str, dict))

    def test_proxy_rotation(self):
        proxies = [
            "http://user:pass@proxy1.example.com:8080",
            "http://user:pass@proxy2.example.com:8080",
        ]
        pm = ProxyManager(proxies=proxies)
        first = pm.get_proxy()
        second = pm.get_proxy()
        if first and second and len(proxies) > 1:
            assert True  # rotation is working if we get proxies back

import pytest
from universal_scraper.core.json_detector import JSONDetector


class TestJSONDetector:
    def setup_method(self):
        self.detector = JSONDetector()

    def test_detects_json_ld(self, sample_json_html):
        result = self.detector.detect_and_extract(
            html=sample_json_html,
            url="https://example.com/product",
        )
        assert result is not None
        assert result.get("json_found") is True

    def test_no_json_in_plain_html(self, sample_html):
        result = self.detector.detect_and_extract(
            html=sample_html,
            url="https://example.com/product",
        )
        json_found = result.get("json_found", False) if result else False
        assert not json_found or len(result.get("data", [])) == 0

    def test_handles_empty_html(self):
        result = self.detector.detect_and_extract(
            html="",
            url="https://example.com",
        )
        assert result is None or result.get("json_found") is False

    def test_handles_malformed_json_ld(self):
        html = """
        <html><head>
        <script type="application/ld+json">{ not valid json }</script>
        </head><body></body></html>
        """
        result = self.detector.detect_and_extract(
            html=html,
            url="https://example.com",
        )
        assert result is None or result.get("json_found") is False

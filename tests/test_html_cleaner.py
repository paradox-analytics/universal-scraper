import pytest
from universal_scraper.core.html_cleaner import SmartHTMLCleaner


class TestSmartHTMLCleaner:
    def setup_method(self):
        self.cleaner = SmartHTMLCleaner()

    def test_removes_script_tags(self):
        html = "<html><body><script>alert('xss')</script><p>Content</p></body></html>"
        result = self.cleaner.clean(html)
        cleaned = result.get("html", "") if isinstance(result, dict) else str(result)
        assert "<script>" not in cleaned
        assert "Content" in cleaned

    def test_removes_style_tags(self):
        html = "<html><body><style>body{color:red}</style><p>Content</p></body></html>"
        result = self.cleaner.clean(html)
        cleaned = result.get("html", "") if isinstance(result, dict) else str(result)
        assert "<style>" not in cleaned
        assert "Content" in cleaned

    def test_preserves_structural_elements(self):
        html = """
        <html><body>
            <h1>Title</h1>
            <div class="product"><span class="price">$10</span></div>
        </body></html>
        """
        result = self.cleaner.clean(html)
        cleaned = result.get("html", "") if isinstance(result, dict) else str(result)
        assert "Title" in cleaned
        assert "$10" in cleaned

    def test_handles_empty_html(self):
        result = self.cleaner.clean("")
        assert result is not None

    def test_significant_size_reduction(self):
        bloated_html = "<html><head>" + "<style>.x{color:red}</style>" * 100
        bloated_html += "</head><body><p>Small content</p></body></html>"
        result = self.cleaner.clean(bloated_html)
        cleaned = result.get("html", "") if isinstance(result, dict) else str(result)
        assert len(cleaned) < len(bloated_html)

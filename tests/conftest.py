import pytest
import os


@pytest.fixture
def api_key():
    return os.environ.get("OPENAI_API_KEY", "test-key-for-unit-tests")


@pytest.fixture
def sample_html():
    return """
    <html>
    <head><title>Test Product Page</title></head>
    <body>
        <div class="product">
            <h1 class="product-title">Widget Pro 3000</h1>
            <span class="price">$49.99</span>
            <span class="brand">Acme Corp</span>
            <p class="description">A high-quality widget for professionals.</p>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def sample_json_html():
    return """
    <html>
    <head>
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": "Widget Pro 3000",
            "brand": {"@type": "Brand", "name": "Acme Corp"},
            "offers": {
                "@type": "Offer",
                "price": "49.99",
                "priceCurrency": "USD"
            }
        }
        </script>
    </head>
    <body>
        <h1>Widget Pro 3000</h1>
    </body>
    </html>
    """

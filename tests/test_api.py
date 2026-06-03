import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestAPIHealth:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200


class TestAPIScrape:
    def test_scrape_requires_auth(self, client):
        response = client.post("/scrape", json={
            "url": "https://example.com",
            "fields": ["title"],
        })
        assert response.status_code in [401, 403, 422]

    def test_scrape_rejects_invalid_payload(self, client):
        response = client.post(
            "/scrape",
            json={},
            headers={"X-Tenant-ID": "test-tenant"},
        )
        assert response.status_code in [400, 422]

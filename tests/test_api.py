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
        assert response.status_code == 401

    def test_scrape_rejects_invalid_payload_without_auth(self, client):
        response = client.post("/scrape", json={})
        assert response.status_code == 401

    def test_scrape_rejects_invalid_payload_with_auth(self, client):
        response = client.post(
            "/scrape",
            json={},
            headers={"X-API-Key": "test-key-for-validation"},
        )
        assert response.status_code == 422

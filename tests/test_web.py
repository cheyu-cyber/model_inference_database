"""Tests for the Web UI gateway endpoints.

The web service uses TestClient internally to call into sub-service
FastAPI apps.  Here we drive the *outer* web service with its own
TestClient, verifying that the gateway wiring is correct.
"""

from fastapi.testclient import TestClient

from messaging import InMemoryBus
import services.web_service as web_mod
import services.upload_service as upload_mod
import services.embedding_service as embedding_mod


def _client(bus: InMemoryBus) -> TestClient:
    web_mod.init_app(bus)
    return TestClient(web_mod.app)


class TestWebUpload:
    def test_upload_returns_image_id(self, bus):
        client = _client(bus)
        resp = client.post(
            "/api/upload",
            files={"file": ("pic.jpg", b"\xff\xd8\xff" + b"\x00" * 64, "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["image_id"]

    def test_upload_triggers_pipeline(self, bus):
        client = _client(bus)
        client.post(
            "/api/upload",
            files={"file": ("x.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        )
        docs = client.get("/api/documents").json()["document_ids"]
        assert len(docs) == 1


class TestWebDocuments:
    def test_list_empty(self, bus):
        client = _client(bus)
        assert client.get("/api/documents").json()["document_ids"] == []

    def test_get_after_upload(self, bus):
        client = _client(bus)
        up = client.post(
            "/api/upload",
            files={"file": ("a.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        ).json()
        doc = client.get(f"/api/documents/{up['image_id']}").json()
        assert doc["image_id"] == up["image_id"]
        assert "annotations" in doc

    def test_get_missing_returns_404(self, bus):
        client = _client(bus)
        assert client.get("/api/documents/nope").status_code == 404


class TestWebSearch:
    def test_search_by_image_id(self, bus):
        client = _client(bus)
        up = client.post(
            "/api/upload",
            files={"file": ("s.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        ).json()
        resp = client.post(
            "/api/search",
            json={"image_id": up["image_id"], "top_k": 1},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) >= 1
        assert results[0]["similarity"] > 0.99

    def test_search_by_vector(self, bus):
        client = _client(bus)
        client.post(
            "/api/upload",
            files={"file": ("v.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        )
        # Get the stored vector via the embeddings endpoint
        docs = client.get("/api/documents").json()["document_ids"]
        image_id = docs[0].replace("doc_", "")
        emb = client.get(f"/api/embeddings/default/{image_id}").json()

        resp = client.post(
            "/api/search",
            json={"vector": emb["vector"], "top_k": 1},
        )
        assert resp.status_code == 200
        assert resp.json()["results"][0]["image_id"] == image_id

    def test_search_no_params_returns_400(self, bus):
        client = _client(bus)
        assert client.post("/api/search", json={}).status_code == 400


class TestWebSchemas:
    def test_list_schemas_initially_empty(self, bus):
        client = _client(bus)
        assert client.get("/api/schemas").json()["schemas"] == []

    def test_register_schema(self, bus):
        client = _client(bus)
        resp = client.post(
            "/api/schemas",
            json={"name": "test-schema", "dimensions": 8, "metric": "cosine"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "test-schema"
        schemas = client.get("/api/schemas").json()["schemas"]
        assert any(s["name"] == "test-schema" for s in schemas)

    def test_stats_after_upload(self, bus):
        client = _client(bus)
        client.post(
            "/api/upload",
            files={"file": ("st.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        )
        stats = client.get("/api/stats").json()
        assert stats["total_embeddings"] == 1


class TestWebHTML:
    def test_root_returns_html(self, bus):
        client = _client(bus)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Semantic Image Database" in resp.text

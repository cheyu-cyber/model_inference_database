"""Tests for the Web UI gateway endpoints.

The web service calls downstream services via ``httpx``.  The
``http_gateway_client`` fixture (see conftest.py) routes those calls
through a MockTransport into each service's in-process ASGI app, so
tests exercise the real HTTP code path without real sockets.
"""


class TestWebUpload:
    def test_upload_returns_image_id(self, http_gateway_client):
        resp = http_gateway_client.post(
            "/api/upload",
            files={"file": ("pic.jpg", b"\xff\xd8\xff" + b"\x00" * 64, "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["image_id"]

    def test_upload_triggers_pipeline(self, http_gateway_client):
        http_gateway_client.post(
            "/api/upload",
            files={"file": ("x.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        )
        docs = http_gateway_client.get("/api/documents").json()["document_ids"]
        assert len(docs) == 1


class TestWebDocuments:
    def test_list_empty(self, http_gateway_client):
        assert http_gateway_client.get("/api/documents").json()["document_ids"] == []

    def test_get_after_upload(self, http_gateway_client):
        up = http_gateway_client.post(
            "/api/upload",
            files={"file": ("a.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        ).json()
        doc = http_gateway_client.get(f"/api/documents/{up['image_id']}").json()
        assert doc["image_id"] == up["image_id"]
        assert "annotations" in doc

    def test_get_missing_returns_404(self, http_gateway_client):
        assert http_gateway_client.get("/api/documents/nope").status_code == 404


class TestWebSearch:
    def test_search_by_image_id(self, http_gateway_client):
        up = http_gateway_client.post(
            "/api/upload",
            files={"file": ("s.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        ).json()
        resp = http_gateway_client.post(
            "/api/search",
            json={"image_id": up["image_id"], "top_k": 1},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) >= 1
        assert results[0]["similarity"] > 0.99

    def test_search_by_vector(self, http_gateway_client):
        http_gateway_client.post(
            "/api/upload",
            files={"file": ("v.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        )
        docs = http_gateway_client.get("/api/documents").json()["document_ids"]
        image_id = docs[0].replace("doc_", "")
        emb = http_gateway_client.get(f"/api/embeddings/default/{image_id}").json()

        resp = http_gateway_client.post(
            "/api/search",
            json={"vector": emb["vector"], "top_k": 1},
        )
        assert resp.status_code == 200
        assert resp.json()["results"][0]["image_id"] == image_id

    def test_search_no_params_returns_400(self, http_gateway_client):
        assert http_gateway_client.post("/api/search", json={}).status_code == 400


class TestWebSchemas:
    def test_list_schemas_initially_empty(self, http_gateway_client):
        assert http_gateway_client.get("/api/schemas").json()["schemas"] == []

    def test_register_schema(self, http_gateway_client):
        resp = http_gateway_client.post(
            "/api/schemas",
            json={"name": "test-schema", "dimensions": 8},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "test-schema"
        schemas = http_gateway_client.get("/api/schemas").json()["schemas"]
        assert any(s["name"] == "test-schema" for s in schemas)

    def test_stats_after_upload(self, http_gateway_client):
        http_gateway_client.post(
            "/api/upload",
            files={"file": ("st.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        )
        stats = http_gateway_client.get("/api/stats").json()
        assert stats["total_embeddings"] == 1


class TestWebHTML:
    def test_root_returns_html(self, http_gateway_client):
        resp = http_gateway_client.get("/")
        assert resp.status_code == 200
        assert "Semantic Image Database" in resp.text

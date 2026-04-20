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
    def test_search_by_text_finds_synonyms(self, http_gateway_client):
        """Upload two images then query via text — see integration test for
        the seeded-tags variant; here we just verify the endpoint shape."""
        http_gateway_client.post(
            "/api/upload",
            files={"file": ("a.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        )
        resp = http_gateway_client.post(
            "/api/search",
            json={"query_text": "pedestrians and 4 wheeler", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data and "query_vector" in data

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
        # Stub inference randomises tags, so similarity is ≥ 0 and the
        # uploaded image is the top match against its own vector.
        results = resp.json()["results"]
        assert len(results) >= 1
        assert results[0]["image_id"] == up["image_id"]

    def test_search_no_params_returns_400(self, http_gateway_client):
        assert http_gateway_client.post("/api/search", json={}).status_code == 400

    def test_vocabulary_endpoint(self, http_gateway_client):
        vocab = http_gateway_client.get("/api/vocabulary").json()
        names = [c["name"] for c in vocab["categories"]]
        assert "person" in names
        assert "four_wheeler" in names


class TestWebSchemas:
    def test_default_semantic_schema_visible(self, http_gateway_client):
        schemas = http_gateway_client.get("/api/schemas").json()["schemas"]
        assert any(s["name"] == "semantic" for s in schemas)

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

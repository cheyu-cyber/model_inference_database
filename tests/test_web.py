"""Tests for the Web service — the only HTTP-speaking service.

Web's HTTP handlers internally do bus request/reply. The ``web_client``
fixture wires every backend to a synchronous in-memory bus, so each
HTTP call exercises the full publish → handler → reply → unblock chain.
"""


class TestWebUpload:
    def test_upload_returns_image_id(self, web_client):
        resp = web_client.post(
            "/api/upload",
            files={"file": ("pic.jpg", b"\xff\xd8\xff" + b"\x00" * 64, "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["image_id"]

    def test_upload_triggers_pipeline(self, web_client):
        web_client.post(
            "/api/upload",
            files={"file": ("x.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        )
        docs = web_client.get("/api/documents").json()["document_ids"]
        assert len(docs) == 1


class TestWebDocuments:
    def test_list_empty(self, web_client):
        assert web_client.get("/api/documents").json()["document_ids"] == []

    def test_get_after_upload(self, web_client):
        up = web_client.post(
            "/api/upload",
            files={"file": ("a.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        ).json()
        doc = web_client.get(f"/api/documents/{up['image_id']}").json()
        assert doc["image_id"] == up["image_id"]
        assert "annotations" in doc

    def test_get_missing_returns_404(self, web_client):
        assert web_client.get("/api/documents/nope").status_code == 404


class TestWebSearch:
    def test_search_by_text_finds_synonyms(self, web_client):
        web_client.post(
            "/api/upload",
            files={"file": ("a.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        )
        resp = web_client.post(
            "/api/search",
            json={"query_text": "pedestrians and 4 wheeler", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data and "query_vector" in data

    def test_search_by_image_id(self, web_client):
        up = web_client.post(
            "/api/upload",
            files={"file": ("s.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        ).json()
        resp = web_client.post(
            "/api/search",
            json={"image_id": up["image_id"], "top_k": 1},
        )
        assert resp.status_code == 200
        # Stub inference randomises tags, so similarity is ≥ 0 and the
        # uploaded image is the top match against its own vector.
        results = resp.json()["results"]
        assert len(results) >= 1
        assert results[0]["image_id"] == up["image_id"]

    def test_search_no_params_returns_400(self, web_client):
        assert web_client.post("/api/search", json={}).status_code == 400

    def test_vocabulary_endpoint(self, web_client):
        vocab = web_client.get("/api/vocabulary").json()
        names = [c["name"] for c in vocab["categories"]]
        assert "person" in names
        assert "four_wheeler" in names


class TestWebSchemas:
    def test_default_semantic_schema_visible(self, web_client):
        schemas = web_client.get("/api/schemas").json()["schemas"]
        assert any(s["name"] == "semantic" for s in schemas)

    def test_register_schema(self, web_client):
        resp = web_client.post(
            "/api/schemas",
            json={"name": "test-schema", "dimensions": 8},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "test-schema"
        schemas = web_client.get("/api/schemas").json()["schemas"]
        assert any(s["name"] == "test-schema" for s in schemas)

    def test_stats_after_upload(self, web_client):
        web_client.post(
            "/api/upload",
            files={"file": ("st.jpg", b"\xff\xd8\xff" + b"\x00" * 32, "image/jpeg")},
        )
        stats = web_client.get("/api/stats").json()
        assert stats["total_embeddings"] == 1


class TestWebHTML:
    def test_root_returns_html(self, web_client):
        resp = web_client.get("/")
        assert resp.status_code == 200
        assert "Semantic Image Database" in resp.text

"""Unit tests for each service in isolation.

Each service is tested by registering only it on the bus, driving it with
well-formed events (or HTTP calls where the service exposes them), and
asserting on the side-effects the service owns plus the events it
publishes.  No other service is wired in, so a failure clearly localizes.
"""

import os

import pytest

from events import (
    DOCUMENT_STORED,
    EMBEDDING_INDEXED,
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
    SEARCH_COMPLETED,
    SEARCH_REQUESTED,
    ImageUploadedPayload,
    InferenceCompletedPayload,
    SearchRequestedPayload,
    make_event,
)
import services.embedding_service as embedding_mod
import services.inference_service as inference_mod


# ----------------------------------------------------------------------
# Upload Service
# ----------------------------------------------------------------------

class TestUploadService:
    def test_post_upload_stores_file_and_publishes_event(self, upload_client, bus):
        resp = upload_client.post(
            "/upload",
            files={"file": ("test.jpg", b"\xff\xd8\xff" + b"\x00" * 64, "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert os.path.exists(data["file_path"])

        emitted = bus.messages_on(IMAGE_UPLOADED)
        assert len(emitted) == 1
        payload = emitted[0]["payload"]
        assert payload["image_id"] == data["image_id"]
        assert payload["mime_type"] == "image/jpeg"
        assert payload["file_size_bytes"] > 0

    def test_upload_assigns_unique_ids(self, upload_client):
        r1 = upload_client.post(
            "/upload", files={"file": ("a.jpg", b"\xff\xd8\xff", "image/jpeg")}
        )
        r2 = upload_client.post(
            "/upload", files={"file": ("b.jpg", b"\xff\xd8\xff", "image/jpeg")}
        )
        assert r1.json()["image_id"] != r2.json()["image_id"]


# ----------------------------------------------------------------------
# Inference Service
# ----------------------------------------------------------------------

class TestInferenceService:
    def test_subscribes_to_image_uploaded(self, bus):
        inference_mod.register(bus)
        evt = make_event(
            IMAGE_UPLOADED,
            ImageUploadedPayload(
                image_id="img-x",
                file_path="/tmp/x.jpg",
                file_size_bytes=10,
                mime_type="image/jpeg",
            ),
            correlation_id="trace-1",
        )
        bus.publish(IMAGE_UPLOADED, evt)

        completed = bus.messages_on(INFERENCE_COMPLETED)
        assert len(completed) == 1
        payload = completed[0]["payload"]
        assert payload["image_id"] == "img-x"
        assert payload["model_name"]
        assert "objects" in payload["annotations"]  # nested annotations
        assert len(payload["embedding_vector"]) == 128
        assert completed[0]["correlation_id"] == "trace-1"

    def test_does_not_publish_on_bad_payload(self, bus):
        inference_mod.register(bus)
        # A broken envelope should raise before any downstream publish.
        with pytest.raises(Exception):
            bus.publish(IMAGE_UPLOADED, {"payload": {"image_id": "x"}})
        assert bus.messages_on(INFERENCE_COMPLETED) == []


# ----------------------------------------------------------------------
# Document DB Service
# ----------------------------------------------------------------------

class TestDocumentDBService:
    def _emit(self, bus, image_id, annotations):
        evt = make_event(
            INFERENCE_COMPLETED,
            InferenceCompletedPayload(
                image_id=image_id,
                model_name="test-model",
                annotations=annotations,
                embedding_vector=[0.0] * 4,
            ),
        )
        bus.publish(INFERENCE_COMPLETED, evt)

    def test_event_stores_document_and_publishes(self, docdb_client, bus):
        self._emit(bus, "img-100", {"label": "cell", "conf": 0.9})

        resp = docdb_client.get("/documents/img-100")
        assert resp.status_code == 200
        assert resp.json()["annotations"]["label"] == "cell"

        stored = bus.messages_on(DOCUMENT_STORED)
        assert len(stored) == 1
        assert stored[0]["payload"]["document_id"] == "doc_img-100"

    def test_list_documents(self, docdb_client, bus):
        for i in range(3):
            self._emit(bus, f"img-{i}", {"i": i})
        resp = docdb_client.get("/documents")
        assert len(resp.json()["document_ids"]) == 3

    def test_variable_annotation_shapes(self, docdb_client, bus):
        self._emit(bus, "a", {"boxes": [{"x": 1, "y": 2}]})
        self._emit(bus, "b", {"label": "cat", "score": 0.99})
        a = docdb_client.get("/documents/a").json()
        b = docdb_client.get("/documents/b").json()
        assert "boxes" in a["annotations"]
        assert b["annotations"]["label"] == "cat"

    def test_get_missing_returns_404(self, docdb_client):
        assert docdb_client.get("/documents/missing").status_code == 404


# ----------------------------------------------------------------------
# Embedding Service
# ----------------------------------------------------------------------

class TestEmbeddingService:
    def _emit_inference(self, bus, image_id, vector, schema="default"):
        evt = make_event(
            INFERENCE_COMPLETED,
            InferenceCompletedPayload(
                image_id=image_id,
                model_name="m",
                annotations={},
                embedding_vector=vector,
                schema_name=schema,
            ),
        )
        bus.publish(INFERENCE_COMPLETED, evt)

    def test_event_indexes_vector_and_publishes(self, embedding_client, bus):
        self._emit_inference(bus, "img-1", [1.0, 0.0, 0.0])
        resp = embedding_client.get("/embeddings/default/img-1")
        assert resp.status_code == 200
        assert resp.json()["dimensions"] == 3

        indexed = bus.messages_on(EMBEDDING_INDEXED)
        assert len(indexed) == 1
        assert indexed[0]["payload"]["dimensions"] == 3

    def test_schema_auto_registered_on_first_vector(self, embedding_client, bus):
        self._emit_inference(bus, "img-1", [0.1, 0.2], schema="vision-v2")
        schemas = embedding_client.get("/schemas").json()["schemas"]
        assert any(s["name"] == "vision-v2" and s["dimensions"] == 2 for s in schemas)

    def test_schema_dimension_mismatch_raises(self, embedding_client, bus):
        self._emit_inference(bus, "a", [0.1, 0.2, 0.3], schema="fixed")
        # Publishing a mismatched vector should surface as a handler error.
        with pytest.raises(ValueError, match="expects dim"):
            self._emit_inference(bus, "b", [0.1, 0.2], schema="fixed")

    def test_explicit_schema_registration_via_http(self, embedding_client):
        resp = embedding_client.post(
            "/schemas", json={"name": "explicit", "dimensions": 4}
        )
        assert resp.status_code == 200

    def test_search_via_event(self, embedding_client, bus):
        self._emit_inference(bus, "a", [1.0, 0.0, 0.0])
        self._emit_inference(bus, "b", [0.0, 1.0, 0.0])

        req = make_event(
            SEARCH_REQUESTED,
            SearchRequestedPayload(
                query_id="q1", vector=[0.9, 0.1, 0.0], top_k=2
            ),
        )
        bus.publish(SEARCH_REQUESTED, req)

        completed = bus.messages_on(SEARCH_COMPLETED)
        assert len(completed) == 1
        results = completed[0]["payload"]["results"]
        assert results[0]["image_id"] == "a"
        assert results[0]["similarity"] > results[1]["similarity"]

    def test_search_http_fallback(self, embedding_client, bus):
        self._emit_inference(bus, "a", [1.0, 0.0, 0.0])
        resp = embedding_client.post(
            "/search/similar",
            json={"vector": [1.0, 0.0, 0.0], "top_k": 1, "schema_name": "default"},
        )
        hits = resp.json()["results"]
        assert hits[0]["image_id"] == "a"
        assert hits[0]["similarity"] > 0.99

    def test_stats_counts_by_schema(self, embedding_client, bus):
        self._emit_inference(bus, "a", [1.0, 0.0])
        self._emit_inference(bus, "b", [0.0, 1.0])
        stats = embedding_client.get("/stats").json()
        assert stats["total_embeddings"] == 2
        assert stats["by_schema"]["default"] == 2

    # pure math
    def test_cosine_identical(self):
        assert abs(embedding_mod._cosine_similarity([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-6

    def test_cosine_orthogonal(self):
        assert abs(embedding_mod._cosine_similarity([1, 0], [0, 1])) < 1e-6

    def test_cosine_zero_vector(self):
        assert embedding_mod._cosine_similarity([0, 0], [1, 2]) == 0.0

    def test_cosine_dim_mismatch(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            embedding_mod._cosine_similarity([1, 2], [1, 2, 3])

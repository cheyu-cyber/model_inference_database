"""Unit tests for each service in isolation.

Each service is tested by registering only it on the bus, driving it with
well-formed events (or HTTP calls where the service exposes them), and
asserting on the side-effects the service owns plus the events it
publishes.  No other service is wired in, so a failure clearly localizes.
"""

import os

import pytest

from DB.model_inference_database.events import (
    DOCUMENT_STORED,
    EMBEDDING_INDEXED,
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
    SEARCH_COMPLETED,
    SEARCH_REQUESTED,
    VECTOR_COMPUTED,
    VECTOR_SEARCH_REQUESTED,
    ImageUploadedPayload,
    InferenceCompletedPayload,
    SearchRequestedPayload,
    VectorComputedPayload,
    make_event,
)
import DB.model_inference_database.services.embedding_service as embedding_mod
import DB.model_inference_database.services.inference_service as inference_mod
import DB.model_inference_database.services.vector_service as vector_mod


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
        obj = payload["annotations"]["objects"][0]
        assert set(obj.keys()) >= {"box", "contours", "tags"}
        assert completed[0]["correlation_id"] == "trace-1"

    def test_does_not_publish_on_bad_payload(self, bus):
        inference_mod.register(bus)
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
# Embedding Service — semantic vocabulary + tag/text → vector
# ----------------------------------------------------------------------

class TestSemanticVocabulary:
    def test_synonyms_map_to_same_vector(self):
        v_person = embedding_mod.semantic_vector(["person"])
        v_people = embedding_mod.semantic_vector(["people"])
        v_pedestrian = embedding_mod.semantic_vector(["pedestrian"])
        assert v_person == v_people == v_pedestrian

    def test_different_categories_produce_different_vectors(self):
        v_person = embedding_mod.semantic_vector(["person"])
        v_car = embedding_mod.semantic_vector(["car"])
        assert v_person != v_car

    def test_vector_is_unit_length(self):
        import math
        v = embedding_mod.semantic_vector(["person", "car", "bicycle"])
        mag = math.sqrt(sum(x * x for x in v))
        assert abs(mag - 1.0) < 1e-9

    def test_unknown_terms_are_ignored(self):
        v = embedding_mod.semantic_vector(["gibberish", "zzz"])
        assert v == [0.0] * embedding_mod.SEMANTIC_DIM

    def test_text_to_vector_handles_free_form(self):
        v_text = embedding_mod.text_to_vector("pedestrians and 4 wheeler")
        v_tags = embedding_mod.semantic_vector(["person", "car"])
        assert v_text == v_tags

    def test_text_to_vector_matches_bigram_synonyms(self):
        v = embedding_mod.text_to_vector("four wheeler stop sign")
        expected = embedding_mod.semantic_vector(["car", "stop sign"])
        assert v == expected


class TestEmbeddingService:
    """Embedding alone — registers only itself on the bus and asserts on
    the events it emits. Vector ownership lives in the next test class."""

    def _register_embedding_only(self, bus):
        embedding_mod.register(bus)

    def _emit_inference(self, bus, image_id, tags):
        evt = make_event(
            INFERENCE_COMPLETED,
            InferenceCompletedPayload(
                image_id=image_id,
                model_name="m",
                annotations={"objects": [{"tags": tags}]},
            ),
        )
        bus.publish(INFERENCE_COMPLETED, evt)

    def test_inference_event_emits_vector_computed(self, bus):
        self._register_embedding_only(bus)
        self._emit_inference(bus, "img-1", ["person", "car"])

        emitted = bus.messages_on(VECTOR_COMPUTED)
        assert len(emitted) == 1
        payload = emitted[0]["payload"]
        assert payload["image_id"] == "img-1"
        assert payload["schema_name"] == "semantic"
        assert len(payload["vector"]) == embedding_mod.SEMANTIC_DIM

    def test_search_request_emits_vector_search_requested(self, bus):
        self._register_embedding_only(bus)
        req = make_event(
            SEARCH_REQUESTED,
            SearchRequestedPayload(
                query_id="q1",
                query_text="pedestrians and 4 wheeler",
                top_k=3,
            ),
        )
        bus.publish(SEARCH_REQUESTED, req)

        forwarded = bus.messages_on(VECTOR_SEARCH_REQUESTED)
        assert len(forwarded) == 1
        payload = forwarded[0]["payload"]
        assert payload["query_id"] == "q1"
        assert payload["top_k"] == 3
        assert payload["vector"] == embedding_mod.text_to_vector(
            "pedestrians and 4 wheeler"
        )

    def test_vocabulary_endpoint(self, embedding_client):
        vocab = embedding_client.get("/vocabulary").json()
        assert vocab["dimensions"] == embedding_mod.SEMANTIC_DIM
        names = [c["name"] for c in vocab["categories"]]
        assert "person" in names and "four_wheeler" in names

    def test_embed_text_endpoint(self, embedding_client):
        resp = embedding_client.post(
            "/embed/text", json={"text": "pedestrians and 4 wheeler"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["dimensions"] == embedding_mod.SEMANTIC_DIM
        assert body["vector"] == embedding_mod.semantic_vector(["person", "car"])

    def test_embed_tags_endpoint(self, embedding_client):
        resp = embedding_client.post("/embed/tags", json={"tags": ["dog", "tree"]})
        assert resp.status_code == 200
        assert resp.json()["vector"] == embedding_mod.semantic_vector(["dog", "tree"])


# ----------------------------------------------------------------------
# Vector Service — FAISS index + similarity search
# ----------------------------------------------------------------------

class TestVectorService:
    def _emit_vector(self, bus, image_id, vector, schema_name="semantic"):
        evt = make_event(
            VECTOR_COMPUTED,
            VectorComputedPayload(
                image_id=image_id, schema_name=schema_name, vector=vector
            ),
        )
        bus.publish(VECTOR_COMPUTED, evt)

    def test_vector_event_indexes_and_publishes(self, vector_client, bus):
        v = embedding_mod.semantic_vector(["person", "car"])
        self._emit_vector(bus, "img-1", v)

        resp = vector_client.get("/embeddings/semantic/img-1")
        assert resp.status_code == 200
        assert resp.json()["dimensions"] == embedding_mod.SEMANTIC_DIM

        indexed = bus.messages_on(EMBEDDING_INDEXED)
        assert len(indexed) == 1
        assert indexed[0]["payload"]["schema_name"] == "semantic"

    def test_explicit_schema_registration_via_http(self, vector_client):
        resp = vector_client.post(
            "/schemas", json={"name": "explicit", "dimensions": 4}
        )
        assert resp.status_code == 200
        names = [s["name"] for s in vector_client.get("/schemas").json()["schemas"]]
        assert "explicit" in names

    def test_search_http_returns_top_hit(self, vector_client, bus):
        self._emit_vector(bus, "a", embedding_mod.semantic_vector(["person", "car"]))
        self._emit_vector(bus, "b", embedding_mod.semantic_vector(["dog", "tree"]))

        query = embedding_mod.text_to_vector("pedestrian in a bus")
        resp = vector_client.post(
            "/search/similar",
            json={"vector": query, "top_k": 2},
        )
        hits = resp.json()["results"]
        assert hits[0]["image_id"] == "a"
        assert hits[0]["similarity"] > hits[-1]["similarity"]

    def test_search_via_event(self, vector_client, bus):
        self._emit_vector(bus, "a", embedding_mod.semantic_vector(["person", "car"]))
        self._emit_vector(bus, "b", embedding_mod.semantic_vector(["dog"]))

        req = make_event(
            VECTOR_SEARCH_REQUESTED,
            {
                "query_id": "q1",
                "schema_name": "semantic",
                "vector": embedding_mod.text_to_vector("pedestrians and 4 wheeler"),
                "top_k": 2,
            },
        )
        bus.publish(VECTOR_SEARCH_REQUESTED, req)

        completed = bus.messages_on(SEARCH_COMPLETED)
        assert len(completed) == 1
        results = completed[0]["payload"]["results"]
        assert results[0]["image_id"] == "a"

    def test_reupsert_overwrites_existing_image(self, vector_client, bus):
        self._emit_vector(bus, "img-x", embedding_mod.semantic_vector(["person"]))
        self._emit_vector(bus, "img-x", embedding_mod.semantic_vector(["car"]))

        stats = vector_client.get("/stats").json()
        assert stats["by_schema"]["semantic"] == 1

        stored = vector_client.get("/embeddings/semantic/img-x").json()["vector"]
        assert stored == embedding_mod.semantic_vector(["car"])

    def test_stats_counts_by_schema(self, vector_client, bus):
        self._emit_vector(bus, "a", embedding_mod.semantic_vector(["person"]))
        self._emit_vector(bus, "b", embedding_mod.semantic_vector(["car"]))
        stats = vector_client.get("/stats").json()
        assert stats["total_embeddings"] == 2
        assert stats["by_schema"]["semantic"] == 2

    # pure math
    def test_cosine_identical(self):
        assert abs(vector_mod.cosine_similarity([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-6

    def test_cosine_orthogonal(self):
        assert abs(vector_mod.cosine_similarity([1, 0], [0, 1])) < 1e-6

    def test_cosine_zero_vector(self):
        assert vector_mod.cosine_similarity([0, 0], [1, 2]) == 0.0

    def test_cosine_dim_mismatch(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            vector_mod.cosine_similarity([1, 2], [1, 2, 3])

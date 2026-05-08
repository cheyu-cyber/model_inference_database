"""Unit tests for each backend service in isolation.

Every backend is now a pure pub/sub daemon — no HTTP. Each service is
exercised by registering only it on the bus, driving it with well-formed
events, and asserting on (a) the side-effects the service owns, and
(b) the events it publishes.
"""

import math

import pytest

from DB.model_inference_database.events import (
    DOCUMENT_GET_COMPLETED,
    DOCUMENT_GET_REQUESTED,
    DOCUMENT_STORED,
    DOCUMENTS_LIST_COMPLETED,
    DOCUMENTS_LIST_REQUESTED,
    EMBED_TAGS_COMPLETED,
    EMBED_TAGS_REQUESTED,
    EMBED_TEXT_COMPLETED,
    EMBED_TEXT_REQUESTED,
    EMBEDDING_GET_COMPLETED,
    EMBEDDING_GET_REQUESTED,
    EMBEDDING_INDEXED,
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
    SCHEMA_CREATE_COMPLETED,
    SCHEMA_CREATE_REQUESTED,
    SCHEMAS_LIST_COMPLETED,
    SCHEMAS_LIST_REQUESTED,
    SEARCH_COMPLETED,
    SEARCH_REQUESTED,
    STATS_COMPLETED,
    STATS_REQUESTED,
    VECTOR_COMPUTED,
    VECTOR_SEARCH_REQUESTED,
    VOCABULARY_COMPLETED,
    VOCABULARY_REQUESTED,
    DocumentGetRequestedPayload,
    DocumentsListRequestedPayload,
    EmbedTagsRequestedPayload,
    EmbedTextRequestedPayload,
    EmbeddingGetRequestedPayload,
    ImageUploadedPayload,
    InferenceCompletedPayload,
    SchemaCreateRequestedPayload,
    SchemasListRequestedPayload,
    SearchRequestedPayload,
    StatsRequestedPayload,
    VectorComputedPayload,
    VocabularyRequestedPayload,
    make_event,
)
import DB.model_inference_database.services.document_db_service as docdb_mod
import DB.model_inference_database.services.embedding_service as embedding_mod
import DB.model_inference_database.services.inference_service as inference_mod
import DB.model_inference_database.services.vector_service as vector_mod


def _last(bus, topic: str) -> dict:
    msgs = bus.messages_on(topic)
    assert msgs, f"no message on {topic}"
    return msgs[-1]["payload"]


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
# Document DB Service — pure pub/sub
# ----------------------------------------------------------------------

class TestDocumentDBService:
    def _emit_inference(self, bus, image_id, annotations):
        bus.publish(
            INFERENCE_COMPLETED,
            make_event(
                INFERENCE_COMPLETED,
                InferenceCompletedPayload(
                    image_id=image_id,
                    model_name="test-model",
                    annotations=annotations,
                ),
            ),
        )

    def _request_get(self, bus, image_id):
        bus.publish(
            DOCUMENT_GET_REQUESTED,
            make_event(
                DOCUMENT_GET_REQUESTED,
                DocumentGetRequestedPayload(image_id=image_id),
            ),
        )

    def _request_list(self, bus):
        bus.publish(
            DOCUMENTS_LIST_REQUESTED,
            make_event(DOCUMENTS_LIST_REQUESTED, DocumentsListRequestedPayload()),
        )

    def test_event_stores_document_and_publishes(self, bus):
        docdb_mod.register(bus)
        self._emit_inference(bus, "img-100", {"label": "cell", "conf": 0.9})

        # document.stored fired
        stored = bus.messages_on(DOCUMENT_STORED)
        assert len(stored) == 1
        assert stored[0]["payload"]["document_id"] == "doc_img-100"

        # documents.get returns the stored doc
        self._request_get(bus, "img-100")
        reply = _last(bus, DOCUMENT_GET_COMPLETED)
        assert reply["error"] is None
        assert reply["document"]["annotations"]["label"] == "cell"

    def test_list_documents(self, bus):
        docdb_mod.register(bus)
        for i in range(3):
            self._emit_inference(bus, f"img-{i}", {"i": i})
        self._request_list(bus)
        reply = _last(bus, DOCUMENTS_LIST_COMPLETED)
        assert len(reply["document_ids"]) == 3

    def test_variable_annotation_shapes(self, bus):
        docdb_mod.register(bus)
        self._emit_inference(bus, "a", {"boxes": [{"x": 1, "y": 2}]})
        self._emit_inference(bus, "b", {"label": "cat", "score": 0.99})
        self._request_get(bus, "a")
        a = _last(bus, DOCUMENT_GET_COMPLETED)
        self._request_get(bus, "b")
        b = _last(bus, DOCUMENT_GET_COMPLETED)
        assert "boxes" in a["document"]["annotations"]
        assert b["document"]["annotations"]["label"] == "cat"

    def test_get_missing_returns_error(self, bus):
        docdb_mod.register(bus)
        self._request_get(bus, "missing")
        reply = _last(bus, DOCUMENT_GET_COMPLETED)
        assert reply["document"] is None
        assert reply["error"] == "not_found"


# ----------------------------------------------------------------------
# Embedding Service — vocabulary + tag/text → vector
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
    """Embedding alone — registers only itself and asserts on its events."""

    def _emit_inference(self, bus, image_id, tags):
        bus.publish(
            INFERENCE_COMPLETED,
            make_event(
                INFERENCE_COMPLETED,
                InferenceCompletedPayload(
                    image_id=image_id,
                    model_name="m",
                    annotations={"objects": [{"tags": tags}]},
                ),
            ),
        )

    def test_inference_event_emits_vector_computed(self, bus):
        embedding_mod.register(bus)
        self._emit_inference(bus, "img-1", ["person", "car"])

        emitted = bus.messages_on(VECTOR_COMPUTED)
        assert len(emitted) == 1
        payload = emitted[0]["payload"]
        assert payload["image_id"] == "img-1"
        assert payload["schema_name"] == "semantic"
        assert len(payload["vector"]) == embedding_mod.SEMANTIC_DIM

    def test_search_request_emits_vector_search_requested(self, bus):
        embedding_mod.register(bus)
        bus.publish(
            SEARCH_REQUESTED,
            make_event(
                SEARCH_REQUESTED,
                SearchRequestedPayload(
                    query_id="q1",
                    query_text="pedestrians and 4 wheeler",
                    top_k=3,
                ),
            ),
        )

        forwarded = bus.messages_on(VECTOR_SEARCH_REQUESTED)
        assert len(forwarded) == 1
        payload = forwarded[0]["payload"]
        assert payload["query_id"] == "q1"
        assert payload["top_k"] == 3
        assert payload["vector"] == embedding_mod.text_to_vector(
            "pedestrians and 4 wheeler"
        )

    def test_vocabulary_request_replies(self, bus):
        embedding_mod.register(bus)
        bus.publish(
            VOCABULARY_REQUESTED,
            make_event(VOCABULARY_REQUESTED, VocabularyRequestedPayload()),
        )
        reply = _last(bus, VOCABULARY_COMPLETED)
        assert reply["dimensions"] == embedding_mod.SEMANTIC_DIM
        names = [c["name"] for c in reply["categories"]]
        assert "person" in names and "four_wheeler" in names

    def test_embed_text_request_replies(self, bus):
        embedding_mod.register(bus)
        bus.publish(
            EMBED_TEXT_REQUESTED,
            make_event(
                EMBED_TEXT_REQUESTED,
                EmbedTextRequestedPayload(text="pedestrians and 4 wheeler"),
            ),
        )
        reply = _last(bus, EMBED_TEXT_COMPLETED)
        assert reply["dimensions"] == embedding_mod.SEMANTIC_DIM
        assert reply["vector"] == embedding_mod.semantic_vector(["person", "car"])

    def test_embed_text_rejects_empty(self, bus):
        embedding_mod.register(bus)
        bus.publish(
            EMBED_TEXT_REQUESTED,
            make_event(
                EMBED_TEXT_REQUESTED,
                EmbedTextRequestedPayload(text=""),
            ),
        )
        reply = _last(bus, EMBED_TEXT_COMPLETED)
        assert reply["error"] == "text is required"

    def test_embed_tags_request_replies(self, bus):
        embedding_mod.register(bus)
        bus.publish(
            EMBED_TAGS_REQUESTED,
            make_event(
                EMBED_TAGS_REQUESTED,
                EmbedTagsRequestedPayload(tags=["dog", "tree"]),
            ),
        )
        reply = _last(bus, EMBED_TAGS_COMPLETED)
        assert reply["vector"] == embedding_mod.semantic_vector(["dog", "tree"])


# ----------------------------------------------------------------------
# Vector Service — FAISS index + similarity search
# ----------------------------------------------------------------------

class TestVectorService:
    def _emit_vector(self, bus, image_id, vector, schema_name="semantic"):
        bus.publish(
            VECTOR_COMPUTED,
            make_event(
                VECTOR_COMPUTED,
                VectorComputedPayload(
                    image_id=image_id, schema_name=schema_name, vector=vector
                ),
            ),
        )

    def test_vector_event_indexes_and_publishes(self, bus):
        vector_mod.register(bus)
        v = embedding_mod.semantic_vector(["person", "car"])
        self._emit_vector(bus, "img-1", v)

        # embedding.indexed fired
        indexed = bus.messages_on(EMBEDDING_INDEXED)
        assert len(indexed) == 1
        assert indexed[0]["payload"]["schema_name"] == "semantic"

        # embeddings.get reply hydrates the vector
        bus.publish(
            EMBEDDING_GET_REQUESTED,
            make_event(
                EMBEDDING_GET_REQUESTED,
                EmbeddingGetRequestedPayload(
                    schema_name="semantic", image_id="img-1"
                ),
            ),
        )
        reply = _last(bus, EMBEDDING_GET_COMPLETED)
        assert reply["dimensions"] == embedding_mod.SEMANTIC_DIM

    def test_explicit_schema_registration_via_event(self, bus):
        vector_mod.register(bus)
        bus.publish(
            SCHEMA_CREATE_REQUESTED,
            make_event(
                SCHEMA_CREATE_REQUESTED,
                SchemaCreateRequestedPayload(name="explicit", dimensions=4),
            ),
        )
        reply = _last(bus, SCHEMA_CREATE_COMPLETED)
        assert reply["error"] is None
        assert reply["name"] == "explicit"

        bus.publish(
            SCHEMAS_LIST_REQUESTED,
            make_event(SCHEMAS_LIST_REQUESTED, SchemasListRequestedPayload()),
        )
        listed = _last(bus, SCHEMAS_LIST_COMPLETED)
        names = [s["name"] for s in listed["schemas"]]
        assert "explicit" in names

    def test_search_via_event(self, bus):
        vector_mod.register(bus)
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

    def test_reupsert_overwrites_existing_image(self, bus):
        vector_mod.register(bus)
        self._emit_vector(bus, "img-x", embedding_mod.semantic_vector(["person"]))
        self._emit_vector(bus, "img-x", embedding_mod.semantic_vector(["car"]))

        bus.publish(
            STATS_REQUESTED,
            make_event(STATS_REQUESTED, StatsRequestedPayload()),
        )
        stats = _last(bus, STATS_COMPLETED)
        assert stats["by_schema"]["semantic"] == 1

        bus.publish(
            EMBEDDING_GET_REQUESTED,
            make_event(
                EMBEDDING_GET_REQUESTED,
                EmbeddingGetRequestedPayload(
                    schema_name="semantic", image_id="img-x"
                ),
            ),
        )
        got = _last(bus, EMBEDDING_GET_COMPLETED)
        assert got["vector"] == embedding_mod.semantic_vector(["car"])

    def test_stats_counts_by_schema(self, bus):
        vector_mod.register(bus)
        self._emit_vector(bus, "a", embedding_mod.semantic_vector(["person"]))
        self._emit_vector(bus, "b", embedding_mod.semantic_vector(["car"]))

        bus.publish(
            STATS_REQUESTED,
            make_event(STATS_REQUESTED, StatsRequestedPayload()),
        )
        stats = _last(bus, STATS_COMPLETED)
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

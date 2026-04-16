"""Unit tests for event envelope + payload schemas.

These tests are the "message unit tests" the assignment calls for.  They
check the *shape contract* of every topic independently of any service,
so a schema regression is caught before it breaks the pipeline.
"""

import pytest
from pydantic import ValidationError

from events import (
    ALL_TOPICS,
    DOCUMENT_STORED,
    EMBEDDING_INDEXED,
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
    SEARCH_COMPLETED,
    SEARCH_REQUESTED,
    DocumentStoredPayload,
    EmbeddingIndexedPayload,
    EventEnvelope,
    ImageUploadedPayload,
    InferenceCompletedPayload,
    SearchCompletedPayload,
    SearchHit,
    SearchRequestedPayload,
    make_event,
)
from events.schemas import PAYLOAD_SCHEMAS, validate_payload


class TestEnvelope:
    def test_envelope_autofills_ids(self):
        env = EventEnvelope(event_type="image.uploaded", payload={"k": "v"})
        assert env.event_id
        assert env.correlation_id
        assert env.timestamp.endswith("+00:00")

    def test_make_event_accepts_pydantic_payload(self):
        payload = ImageUploadedPayload(
            image_id="abc",
            file_path="/tmp/a.jpg",
            file_size_bytes=10,
            mime_type="image/jpeg",
        )
        env = make_event(IMAGE_UPLOADED, payload, correlation_id="c-1")
        assert env["event_type"] == IMAGE_UPLOADED
        assert env["correlation_id"] == "c-1"
        assert env["payload"]["image_id"] == "abc"

    def test_make_event_accepts_dict_payload(self):
        env = make_event(IMAGE_UPLOADED, {
            "image_id": "abc",
            "file_path": "/a",
            "file_size_bytes": 1,
            "mime_type": "image/jpeg",
        })
        assert env["payload"]["image_id"] == "abc"


class TestPayloadSchemas:
    def test_all_topics_have_a_schema(self):
        assert set(PAYLOAD_SCHEMAS.keys()) == set(ALL_TOPICS)

    def test_image_uploaded_valid(self):
        p = ImageUploadedPayload(
            image_id="x", file_path="/a.jpg", file_size_bytes=100, mime_type="image/jpeg"
        )
        assert p.image_id == "x"

    def test_image_uploaded_rejects_missing_field(self):
        with pytest.raises(ValidationError):
            ImageUploadedPayload(image_id="x", file_path="/a")  # type: ignore[call-arg]

    def test_inference_completed_allows_nested_annotations(self):
        p = InferenceCompletedPayload(
            image_id="x",
            model_name="m",
            annotations={"objects": [{"label": "cell", "attrs": {"area": 1}}]},
            embedding_vector=[0.1] * 4,
        )
        assert p.annotations["objects"][0]["attrs"]["area"] == 1
        assert p.schema_name == "default"

    def test_search_requested_defaults(self):
        p = SearchRequestedPayload(query_id="q", vector=[0.0])
        assert p.top_k == 5
        assert p.schema_name == "default"

    def test_search_completed_holds_hits(self):
        p = SearchCompletedPayload(
            query_id="q",
            schema_name="default",
            results=[SearchHit(image_id="a", similarity=0.9)],
        )
        assert p.results[0].similarity == 0.9


class TestValidateDispatch:
    def test_roundtrip_all_topics(self):
        samples = {
            IMAGE_UPLOADED: ImageUploadedPayload(
                image_id="x", file_path="/a", file_size_bytes=1, mime_type="image/jpeg"
            ),
            INFERENCE_COMPLETED: InferenceCompletedPayload(
                image_id="x", model_name="m", annotations={}, embedding_vector=[0.0]
            ),
            DOCUMENT_STORED: DocumentStoredPayload(
                document_id="d", image_id="x", model_name="m"
            ),
            EMBEDDING_INDEXED: EmbeddingIndexedPayload(
                image_id="x", schema_name="default", dimensions=4
            ),
            SEARCH_REQUESTED: SearchRequestedPayload(query_id="q", vector=[0.0]),
            SEARCH_COMPLETED: SearchCompletedPayload(
                query_id="q", schema_name="default", results=[]
            ),
        }
        for topic, payload in samples.items():
            env = make_event(topic, payload)
            validated = validate_payload(topic, env["payload"])
            assert validated.model_dump() == payload.model_dump()

    def test_validate_unknown_topic(self):
        with pytest.raises(KeyError):
            validate_payload("not.a.topic", {})

    def test_validate_rejects_bad_payload(self):
        with pytest.raises(ValidationError):
            validate_payload(IMAGE_UPLOADED, {"image_id": "x"})

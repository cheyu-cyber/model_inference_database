"""Integration tests — every service subscribed to one shared bus.

A single ``InMemoryBus`` fans events through Upload → Inference →
Document DB + Embedding.  Because the in-memory bus is synchronous,
these tests read like a request/response scenario but exercise exactly
the same handler code that Redis drives in production.
"""

import os

from fastapi.testclient import TestClient

from events import (
    DOCUMENT_STORED,
    EMBEDDING_INDEXED,
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
    SEARCH_COMPLETED,
    SEARCH_REQUESTED,
    SearchRequestedPayload,
    make_event,
)
import services.document_db_service as docdb_mod
import services.embedding_service as embedding_mod
import services.upload_service as upload_mod


def _upload(bus, filename="test.jpg", content=b"\xff\xd8\xff" + b"\x00" * 64):
    client = TestClient(upload_mod.app)
    resp = client.post(
        "/upload", files={"file": (filename, content, "image/jpeg")}
    )
    assert resp.status_code == 200
    return resp.json()


class TestFullPipeline:
    def test_upload_fans_through_every_service(self, wired_bus):
        data = _upload(wired_bus)
        image_id = data["image_id"]

        # The full event chain fired synchronously during /upload.
        assert len(wired_bus.messages_on(IMAGE_UPLOADED)) == 1
        assert len(wired_bus.messages_on(INFERENCE_COMPLETED)) == 1
        assert len(wired_bus.messages_on(DOCUMENT_STORED)) == 1
        assert len(wired_bus.messages_on(EMBEDDING_INDEXED)) == 1

        # Document persisted — annotations have the {box, contours, tags} shape.
        doc_resp = TestClient(docdb_mod.app).get(f"/documents/{image_id}")
        assert doc_resp.status_code == 200
        obj = doc_resp.json()["annotations"]["objects"][0]
        assert set(obj.keys()) >= {"box", "contours", "tags"}

        # Vector indexed.
        emb_resp = TestClient(embedding_mod.app).get(
            f"/embeddings/default/{image_id}"
        )
        assert emb_resp.status_code == 200
        assert emb_resp.json()["dimensions"] == 128

    def test_correlation_id_propagates_through_chain(self, wired_bus, generator):
        generator.emit_image_uploaded(image_id="trace-target", correlation_id="c-xyz")
        inferred = wired_bus.messages_on(INFERENCE_COMPLETED)[0]
        stored = wired_bus.messages_on(DOCUMENT_STORED)[0]
        indexed = wired_bus.messages_on(EMBEDDING_INDEXED)[0]
        assert inferred["correlation_id"] == "c-xyz"
        assert stored["correlation_id"] == "c-xyz"
        assert indexed["correlation_id"] == "c-xyz"

    def test_search_event_roundtrip(self, wired_bus, generator):
        # Index two vectors via the upload pipeline.
        generator.emit_image_uploaded(image_id="img-a")
        generator.emit_image_uploaded(image_id="img-b")

        # Retrieve one of the stored vectors to use as a query.
        vec = TestClient(embedding_mod.app).get(
            "/embeddings/default/img-a"
        ).json()["vector"]

        req = make_event(
            SEARCH_REQUESTED,
            SearchRequestedPayload(query_id="q-int", vector=vec, top_k=2),
        )
        wired_bus.publish(SEARCH_REQUESTED, req)

        completed = wired_bus.messages_on(SEARCH_COMPLETED)
        assert len(completed) == 1
        top = completed[0]["payload"]["results"][0]
        assert top["image_id"] == "img-a"
        assert top["similarity"] > 0.99

    def test_multiple_uploads_independent(self, wired_bus):
        a = _upload(wired_bus, "a.jpg")
        b = _upload(wired_bus, "b.jpg")
        assert a["image_id"] != b["image_id"]

        docs = TestClient(docdb_mod.app).get("/documents").json()["document_ids"]
        assert len(docs) == 2

        stats = TestClient(embedding_mod.app).get("/stats").json()
        assert stats["total_embeddings"] == 2

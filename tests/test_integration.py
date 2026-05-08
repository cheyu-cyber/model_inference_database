"""Integration tests — every backend subscribed to one shared bus.

A single ``InMemoryBus`` fans events through every service synchronously,
so these tests read like a request/response scenario but exercise exactly
the same handler code that Redis drives in production.

Web is the only HTTP entry point: uploads enter as ``POST /api/upload``
and every read goes through Web's bus request/reply paths.
"""

from DB.model_inference_database.events import (
    DOCUMENT_STORED,
    EMBEDDING_INDEXED,
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
    SEARCH_COMPLETED,
    SEARCH_REQUESTED,
    InferenceCompletedPayload,
    SearchRequestedPayload,
    make_event,
)


def _upload(client, filename="test.jpg", content=b"\xff\xd8\xff" + b"\x00" * 64):
    resp = client.post(
        "/api/upload", files={"file": (filename, content, "image/jpeg")}
    )
    assert resp.status_code == 200
    return resp.json()


class TestFullPipeline:
    def test_upload_fans_through_every_service(self, web_client, wired_bus):
        data = _upload(web_client)
        image_id = data["image_id"]

        # The full event chain fired synchronously during /api/upload.
        assert len(wired_bus.messages_on(IMAGE_UPLOADED)) == 1
        assert len(wired_bus.messages_on(INFERENCE_COMPLETED)) == 1
        assert len(wired_bus.messages_on(DOCUMENT_STORED)) == 1
        assert len(wired_bus.messages_on(EMBEDDING_INDEXED)) == 1

        # Document persisted — reachable via the bus through Web.
        doc = web_client.get(f"/api/documents/{image_id}").json()
        obj = doc["annotations"]["objects"][0]
        assert set(obj.keys()) >= {"box", "contours", "tags"}

        # Vector indexed under the built-in "semantic" schema.
        emb = web_client.get(f"/api/embeddings/semantic/{image_id}").json()
        assert emb["dimensions"]

    def test_correlation_id_propagates_through_chain(self, wired_bus, generator):
        generator.emit_image_uploaded(image_id="trace-target", correlation_id="c-xyz")
        inferred = wired_bus.messages_on(INFERENCE_COMPLETED)[0]
        stored = wired_bus.messages_on(DOCUMENT_STORED)[0]
        indexed = wired_bus.messages_on(EMBEDDING_INDEXED)[0]
        assert inferred["correlation_id"] == "c-xyz"
        assert stored["correlation_id"] == "c-xyz"
        assert indexed["correlation_id"] == "c-xyz"

    def test_text_search_finds_synonyms_end_to_end(self, wired_bus):
        """Seed two images then search by text.

        Searching for "pedestrians and 4 wheeler" must score the
        person+car image higher than the dog+tree image, even though
        none of those query words appear verbatim in the tags.
        """
        def _seed(image_id: str, tags: list[str]) -> None:
            wired_bus.publish(
                INFERENCE_COMPLETED,
                make_event(
                    INFERENCE_COMPLETED,
                    InferenceCompletedPayload(
                        image_id=image_id,
                        model_name="stub",
                        annotations={"objects": [{"tags": tags}]},
                    ),
                ),
            )

        _seed("img-city",   ["person", "car"])
        _seed("img-forest", ["dog", "tree"])

        req = make_event(
            SEARCH_REQUESTED,
            SearchRequestedPayload(
                query_id="q-int",
                query_text="pedestrians and 4 wheeler",
                top_k=2,
            ),
        )
        wired_bus.publish(SEARCH_REQUESTED, req)

        completed = wired_bus.messages_on(SEARCH_COMPLETED)
        assert len(completed) == 1
        results = completed[0]["payload"]["results"]
        assert results[0]["image_id"] == "img-city"
        assert results[0]["similarity"] > results[1]["similarity"]

    def test_multiple_uploads_independent(self, web_client, wired_bus):
        a = _upload(web_client, "a.jpg")
        b = _upload(web_client, "b.jpg")
        assert a["image_id"] != b["image_id"]

        docs = web_client.get("/api/documents").json()["document_ids"]
        assert len(docs) == 2

        stats = web_client.get("/api/stats").json()
        assert stats["total_embeddings"] == 2

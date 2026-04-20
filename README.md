# Semantic Image Database System

An event-driven microservice system for model-inferenced image data.
Upload an image, a (stub) YOLO-style model emits nested annotations,
the Document DB stores them, the Embedding Service vectorizes tags
into a semantic space, and the Web UI lets users ask natural-language
questions like *"pictures with pedestrians and a 4 wheeler"* and get
back matching images.

Services communicate over a **Redis pub/sub message bus** for writes
and **HTTP** for reads.  Each service owns its own data store.

## Two user journeys

**1. Upload → annotate → index (async, fully pub/sub)**

```
 Browser  ─POST /api/upload─▶  Upload  ─file on disk─▶  publishes image.uploaded
                                                               │
   Inference  ─subscribes image.uploaded─▶  runs YOLO stub ─▶  publishes inference.completed
                                                               │
   ┌───────────────────────────────────────────────────────────┴───────────┐
   ▼                                                                       ▼
DocDB (subscribes, writes JSON, publishes document.stored)   Embedding (subscribes,
                                                              builds semantic vector from
                                                              tags, publishes embedding.indexed)
```

**2. Text search → semantic retrieval**

```
 Browser: "pedestrians and a 4 wheeler"
            │
            ▼
 Web  ─POST /api/search {query_text}─▶  Embedding
                                          │
                                          ├─ tokenize + match synonyms via vocabulary
                                          │   ("pedestrians" → person, "4 wheeler" → car)
                                          └─ cosine-search the index, return hits
            ▲
            └── images whose tags include person+car rank highest
```

A parallel event-driven path exists: Web can publish `search.requested`
and the Embedding Service replies with `search.completed` — same handler
code, same result, just async.

## Project layout

```
model_inference_database/
├── events/
│   ├── topics.py       # Channel-name constants
│   └── schemas.py      # Pydantic payload + envelope schemas
├── messaging/
│   ├── bus.py          # MessageBus, InMemoryBus, RedisBus
│   └── generator.py    # Deterministic EventGenerator (contracts, replay, fault injection)
├── services/
│   ├── upload_service.py       # :8001  POST /upload → publishes image.uploaded
│   ├── inference_service.py    # subscribes image.uploaded → publishes inference.completed
│   ├── document_db_service.py  # :8003  subscribes inference.completed, owns JSON docs
│   ├── embedding_service.py    # :8004  schema manager + semantic vector index
│   └── web_service.py          # :8000  HTML UI + HTTP gateway (httpx to others)
├── web/index.html              # Interactive UI — upload, browse, semantic search
├── tests/                      # 70 tests
└── requirements.txt
```

## Services & ownership

| Service     | Owns                                        | Publishes                               | Subscribes                             |
|-------------|---------------------------------------------|-----------------------------------------|----------------------------------------|
| Upload      | Raw image files on disk                     | `image.uploaded`                        | —                                      |
| Inference   | nothing (stateless transform)               | `inference.completed`                   | `image.uploaded`                       |
| Document DB | Annotation documents (JSON, one per image)  | `document.stored`                       | `inference.completed`                  |
| Embedding   | Vector schemas + vector index + vocabulary  | `embedding.indexed`, `search.completed` | `inference.completed`, `search.requested` |
| Web         | nothing — front-end gateway                 | —                                       | —                                      |

Rule: exactly one service owns each data store.  Others read over HTTP
or react to events.

### Why a document DB?
Annotations are variable and nested.  A YOLO detector emits
`{box, contours, tags}` per object; a classifier emits a flat label
map; tomorrow's model emits something else.  A JSON document store
accepts any shape — no schema migrations when model output changes.

### Why the embedding service is also a schema manager and a vocabulary
*Schemas* (`name`, `dimensions`) reject shape drift at index time so a
new model can't silently corrupt the vector space of an old one.  The
default schema `"semantic"` has one dimension per category in a
hand-curated vocabulary that maps synonyms (`person` / `people` /
`pedestrian`) to the same axis.  This is what makes text queries work:
indexing and query-vectorization use the *same* vocabulary, so
`"pedestrians"` naturally matches images tagged `person`.  Swapping in
a real text-embedding model (e.g. sentence-transformers) is a
single-function replacement.

## Event (message) contracts

Every message is wrapped in an `EventEnvelope`:

```json
{
  "event_id":        "uuid",
  "event_type":      "image.uploaded",
  "timestamp":       "2026-04-18T…+00:00",
  "correlation_id":  "uuid (threads a request across services)",
  "payload":         { ... topic-specific schema ... }
}
```

| Topic                 | Payload                                           |
|-----------------------|---------------------------------------------------|
| `image.uploaded`      | `{image_id, file_path, file_size_bytes, mime_type}` |
| `inference.completed` | `{image_id, model_name, annotations, schema_name}` |
| `document.stored`     | `{document_id, image_id, model_name}`              |
| `embedding.indexed`   | `{image_id, schema_name, dimensions}`              |
| `search.requested`    | `{query_id, schema_name, vector?, query_text?, top_k}` |
| `search.completed`    | `{query_id, schema_name, results: [{image_id, similarity}]}` |

Definitions: [events/schemas.py](events/schemas.py).

## Running

```bash
pip install -r requirements.txt
redis-server &

# one terminal per service
python services/document_db_service.py   # :8003
python services/embedding_service.py     # :8004
python services/inference_service.py     # subscriber daemon (no port)
python services/upload_service.py        # :8001
python services/web_service.py           # :8000  ← browse here
```

### Configuration (environment variables)

| Variable                | Default                       |
|-------------------------|-------------------------------|
| `BUS_BACKEND`           | `redis` (`memory` for tests)  |
| `REDIS_URL`             | `redis://localhost:6379/0`    |
| `UPLOAD_SERVICE_URL`    | `http://localhost:8001`       |
| `DOCDB_SERVICE_URL`     | `http://localhost:8003`       |
| `EMBEDDING_SERVICE_URL` | `http://localhost:8004`       |
| `UPLOAD_STORAGE_DIR`    | `./data/uploads`              |
| `DOCDB_STORAGE_DIR`     | `./data/documents`            |
| `MODEL_NAME`            | `stub-yolo-v1`                |

## Web UI

At <http://localhost:8000/>:

* **Upload** — drag-and-drop; triggers the whole pipeline asynchronously.
* **Documents** — live table.  *View* shows the JSON annotation, *Find similar*
  uses that image's vector as the query.
* **Semantic Search** — free-form text like *"pedestrians and 4 wheeler"*.
* **Schemas / Stats** — the built-in `semantic` schema plus any user-registered ones.

## Testing

```bash
pytest tests/ -v
```

Tests run without Redis — `InMemoryBus` dispatches synchronously.

| File                  | Scope                                                |
|-----------------------|------------------------------------------------------|
| `test_events.py`      | Every payload schema + envelope round-trip           |
| `test_bus.py`         | Bus semantics + fault injection                      |
| `test_generator.py`   | Deterministic generator, replay, fault injection     |
| `test_services.py`    | Each service in isolation, plus the semantic vocabulary |
| `test_web.py`         | Web UI gateway endpoints via MockTransport           |
| `test_integration.py` | Full pipeline + end-to-end text-search (“pedestrians and 4 wheeler” → `person`+`car`) |

### Testing strategy

* **Message unit tests** — `validate_payload(topic, …)` round-trips every
  topic against its Pydantic schema.
* **Deterministic replay** — `EventGenerator(seed=n)` produces byte-identical
  payloads run-to-run.
* **Fault injection** — `InMemoryBus.inject_fault(topic, exc)` simulates
  broker failures.
* **Service isolation** — wire only the service under test; failures localize.
* **Integration** — `wired_bus` fixture registers every service on one bus
  and the full chain fires synchronously.

## Design justification

* **Async-by-default**: upload returns as soon as the file is on disk;
  everything else is driven by events.  No request blocks on inference.
* **One owner per store**: Upload→files, DocDB→JSON, Embedding→vectors+vocab.
  Others read-only.
* **Bus abstraction**: services depend on `MessageBus`, not `redis` directly.
  Swap the backend (in-memory for tests, Redis for prod) without touching service code.
* **Semantic vocabulary lives with the index**: query-time tokenization
  uses the same synonym table as indexing, so text search and tag search
  share a vector space.

# Semantic Image Database System

An event-driven microservice system that manages model-inferenced
semantic data (annotations) and image embeddings.  Services communicate
over a **Redis pub/sub message bus**; each service owns its own data.

## Project layout

```
model_inference_database/
├── events/                 # Topic names + Pydantic message schemas
│   ├── topics.py
│   └── schemas.py
├── messaging/              # Bus abstraction + test event generator
│   ├── bus.py              # MessageBus, InMemoryBus, RedisBus
│   └── generator.py        # Deterministic EventGenerator (fault injection, replay)
├── services/
│   ├── upload_service.py       # POST /upload → file to disk → image.uploaded
│   ├── inference_service.py    # image.uploaded → model → inference.completed
│   ├── document_db_service.py  # inference.completed → JSON doc + document.stored
│   ├── embedding_service.py    # inference.completed → vector index + embedding.indexed
│   │                           # search.requested → search.completed
│   └── web_service.py          # HTML UI + REST API gateway (single entry point)
├── web/
│   └── index.html          # Interactive UI (upload, browse, search, schemas)
├── tests/                  # 61 tests — events, bus, generator, services, web, integration
└── requirements.txt
```

## Architecture

The Web service is a user-facing HTTP gateway.  It talks to each service
over HTTP (no module imports).  Internal write-path coordination still
flows over Redis pub/sub.

```
                   Browser (web/index.html)
                             │
                   GET /  POST /api/upload  POST /api/search  …
                             │
                     ┌───────▼────────┐
                     │  Web Service   │  (httpx → each downstream URL)
                     └───┬────┬────┬──┘
                  HTTP   │    │    │   HTTP
              ┌──────────┘    │    └──────────┐
              ▼               ▼               ▼
       Upload :8001     DocDB :8003     Embedding :8004
          │                  ▲                ▲  ▲
          │ publishes        │ subscribes     │  │
          │ image.uploaded   │ inference.*    │  │
          ▼                  │                │  │
   ┌──────────────────── Redis pub/sub ───────┴──┴──────┐
   │                                                    │
   │  image.uploaded → Inference → inference.completed  │
   │  inference.completed → DocDB  +  Embedding         │
   │  document.stored / embedding.indexed (audit)       │
   │  search.requested ↔ search.completed               │
   │                                                    │
   └────────────────────────────────────────────────────┘
                          ▲
                          │ subscriber daemon
                     Inference Service
```

## Services & ownership

| Service            | Owns                                | Publishes             | Subscribes             |
|--------------------|-------------------------------------|-----------------------|------------------------|
| Upload             | Raw image files on disk             | `image.uploaded`      | —                      |
| Inference          | nothing (stateless)                 | `inference.completed` | `image.uploaded`       |
| Document DB        | Annotation documents (JSON)         | `document.stored`     | `inference.completed`  |
| Embedding          | Vector schemas + vector index       | `embedding.indexed`, `search.completed` | `inference.completed`, `search.requested` |
| Web (gateway)      | nothing — HTTP + HTML front-end     | `search.requested` (via API) | —               |

**Rule**: exactly one service owns each data store.  Others observe via events.

### Why a document DB?
Annotations are variable and nested — a detector emits bounding boxes with
per-object attribute dicts; a classifier emits a flat label map.  A JSON
document store accepts whatever the inference service produces.  No
schema migrations when the model output changes.

### Why the embedding service doubles as a schema manager
Different models emit vectors of different dimensionality.  The embedding
service owns named vector *schemas* (`name`, `dimensions`, `metric`) so
shape drift is rejected at index time instead of silently corrupting the
index.  A new model registers a new schema; old and new vectors coexist.

## Event (message) contracts

Every message is wrapped in an `EventEnvelope`:

```json
{
  "event_id":        "uuid",
  "event_type":      "image.uploaded",
  "timestamp":       "2026-04-16T…+00:00",
  "correlation_id":  "uuid (threads a request across services)",
  "payload":         { ... topic-specific schema ... }
}
```

Topic → payload schema map (see [events/schemas.py](events/schemas.py)):

| Topic                 | Payload schema               |
|-----------------------|------------------------------|
| `image.uploaded`      | `ImageUploadedPayload`       |
| `inference.completed` | `InferenceCompletedPayload`  |
| `document.stored`     | `DocumentStoredPayload`      |
| `embedding.indexed`   | `EmbeddingIndexedPayload`    |
| `search.requested`    | `SearchRequestedPayload`     |
| `search.completed`    | `SearchCompletedPayload`     |

## Running

Each service is its own process.  The Web service reaches the others
over HTTP — URLs are configurable via environment variables.

```bash
pip install -r requirements.txt
redis-server &

# One terminal per service
python services/document_db_service.py   # :8003
python services/embedding_service.py     # :8004
python services/inference_service.py     # subscriber daemon (no port)
python services/upload_service.py        # :8001
python services/web_service.py           # :8000  ← open this in a browser
```

Then open <http://localhost:8000>.

### Configuration (environment variables)

| Variable                | Default                       |
|-------------------------|-------------------------------|
| `BUS_BACKEND`           | `redis` (`memory` available for tests) |
| `REDIS_URL`             | `redis://localhost:6379/0`    |
| `UPLOAD_SERVICE_URL`    | `http://localhost:8001`       |
| `DOCDB_SERVICE_URL`     | `http://localhost:8003`       |
| `EMBEDDING_SERVICE_URL` | `http://localhost:8004`       |
| `UPLOAD_STORAGE_DIR`    | `./data/uploads`              |
| `DOCDB_STORAGE_DIR`     | `./data/documents`            |
| `MODEL_NAME`            | `stub-classifier-v1`          |

## Web UI

The UI at `http://localhost:8000/` supports:

* **Upload** — drag-and-drop or click-to-browse; triggers the full pipeline.
* **Documents** — live table of stored annotation documents.  *View* shows
  the full JSON; *Find similar* runs a similarity search using that image's
  embedding.
* **Search** — by image ID (look up its stored vector) or by raw vector.
* **Schemas / Stats** — register new vector schemas and inspect the index.

## Testing

```bash
pytest tests/ -v
```

Tests run without Redis — the `InMemoryBus` dispatches synchronously so
assertions run immediately after each publish.

| File                       | Scope                                               |
|----------------------------|-----------------------------------------------------|
| `test_events.py`           | Message schemas — contract per topic                |
| `test_bus.py`              | Bus semantics + fault injection                     |
| `test_generator.py`        | Deterministic generator, replay, fault injection    |
| `test_services.py`         | Each service in isolation (events in, events out)   |
| `test_web.py`              | Web UI gateway endpoints                            |
| `test_integration.py`      | Full pipeline: Upload → Inference → DocDB + Embedding |

### Testing strategy

* **Message unit tests** — every topic has a Pydantic payload schema.
  `validate_payload(topic, data)` is called round-trip in
  `test_events.py::TestValidateDispatch`.
* **Deterministic replay** — `EventGenerator(seed=n)` produces byte-identical
  payloads across runs; see `test_generator.py`.
* **Fault injection** — `InMemoryBus.inject_fault(topic, exc)` simulates
  broker failures; see `test_bus.py` and `test_generator.py`.
* **Service isolation** — each test wires only the service under test to
  the bus so failures localize.
* **Integration** — `tests/conftest.py::wired_bus` registers all services
  on one bus; the full chain fires synchronously.

## Design justification

* **Async-by-default**: upload, inference, storage, indexing, retrieval
  do not need one blocking call chain.  Pub/sub decouples timing so the
  upload HTTP call returns as soon as the file is on disk.
* **One owner per data store**: Upload owns files, DocDB owns JSON docs,
  Embedding owns the vector index.  Everyone else reads through events
  or read-only HTTP.
* **Bus abstraction**: services depend on `MessageBus`, not Redis.
  `InMemoryBus` for tests, `RedisBus` for production — same contract.

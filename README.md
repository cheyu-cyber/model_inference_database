# Semantic Image Database System

## Architecture Overview

A microservice system that manages model-inferenced semantic data and images.
Each service is a **standalone FastAPI application** that communicates with
other services exclusively via **HTTP API calls** — no shared in-process state.

## Why a Document Database?

Inference annotations are **variable and nested**. A microscopy image might
produce 3 detected objects each with different attribute sets; a classification
model returns flat labels. A document model (JSON documents) stores each
annotation as-is. No schema migration needed when model output format changes.

## Services & Ports

| Service            | Port | Owns                          |
|--------------------|------|-------------------------------|
| Upload Service     | 8001 | Raw image files (filesystem)  |
| Inference Service  | 8002 | Nothing (stateless transform) |
| Document DB Service| 8003 | Annotation documents (JSON)   |
| Embedding Service  | 8004 | Vector index (in-memory)      |

**Rule**: Each service owns its data store. Others access it via API calls.

## API Endpoints

### Upload Service (`:8001`)
| Method | Path      | Description                    |
|--------|-----------|--------------------------------|
| POST   | `/upload` | Upload an image file           |

### Inference Service (`:8002`)
| Method | Path     | Description                     |
|--------|----------|---------------------------------|
| POST   | `/infer` | Run model inference on an image |

### Document DB Service (`:8003`)
| Method | Path                   | Description                 |
|--------|------------------------|-----------------------------|
| POST   | `/documents`           | Store annotation document   |
| GET    | `/documents`           | List all document IDs       |
| GET    | `/documents/{image_id}`| Get document by image ID    |

### Embedding Service (`:8004`)
| Method | Path                    | Description                  |
|--------|-------------------------|------------------------------|
| POST   | `/embeddings`           | Store an embedding vector    |
| GET    | `/embeddings/{image_id}`| Get vector by image ID       |
| POST   | `/search/similar`       | Cosine similarity search     |
| GET    | `/stats`                | Index size                   |

## API Call Flow

```
Client
  │
  POST /upload (file)
  │
  ▼
Upload Service (:8001)
  │  stores file
  │  POST /infer ──────────▶ Inference Service (:8002)
  │                              │  runs model
  │                              ├── POST /documents ──▶ Document DB (:8003)
  │                              └── POST /embeddings ─▶ Embedding  (:8004)
  │
  ◀── returns image_id
```

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Start each service (separate terminals)
python services/document_db_service.py   # port 8003
python services/embedding_service.py     # port 8004
python services/inference_service.py     # port 8002
python services/upload_service.py        # port 8001

# Or with uvicorn
uvicorn services.upload_service:app --port 8001 --reload
```

### Configuration (environment variables)

| Variable                | Default                  |
|-------------------------|--------------------------|
| `UPLOAD_STORAGE_DIR`    | `./data/uploads`         |
| `INFERENCE_SERVICE_URL` | `http://localhost:8002`  |
| `DOCDB_SERVICE_URL`     | `http://localhost:8003`  |
| `DOCDB_STORAGE_DIR`     | `./data/documents`       |
| `EMBEDDING_SERVICE_URL` | `http://localhost:8004`  |
| `MODEL_NAME`            | `stub-classifier-v1`    |

## Testing

```bash
# Run all tests (no running services needed — httpx calls are mocked)
pytest tests/ -v
```

### Testing Strategy

- **Service tests**: Each service API tested in isolation via FastAPI TestClient
- **Integration tests**: Full pipeline (upload → inference → docdb + embedding) with
  inter-service HTTP calls intercepted and routed to real TestClients
- **Math tests**: Cosine similarity edge cases (identical, orthogonal, zero, mismatch)

"""Redis pub/sub topic (channel) name constants.

Topics are the contract between services. Publishing and subscribing services
must agree on the topic name. Keeping the names here (and nowhere else)
prevents typos and provides a single source of truth for the system's
event vocabulary.

The system is **pure pub/sub** — every cross-service interaction (read or
write) flows over the bus. Each former HTTP read endpoint has been split
into a request/reply topic pair (``foo.requested`` → ``foo.completed``),
matched by the envelope's ``correlation_id`` (which every handler
already propagates).

Pipeline (write path)
---------------------
image.uploaded            pub: Web Service          sub: Inference
inference.completed       pub: Inference            sub: Document DB, Embedding
document.stored           pub: Document DB          sub: (audit / tests)
vector.computed           pub: Embedding            sub: Vector
embedding.indexed         pub: Vector               sub: (audit / tests)

Search path
-----------
search.requested          pub: Web                  sub: Embedding
vector.search.requested   pub: Embedding            sub: Vector
search.completed          pub: Vector               sub: Web (reply)

Read paths (request/reply, all initiated by Web)
------------------------------------------------
documents.list.requested  / documents.list.completed   ↔ Document DB
documents.get.requested   / documents.get.completed    ↔ Document DB
embeddings.get.requested  / embeddings.get.completed   ↔ Vector
schemas.list.requested    / schemas.list.completed     ↔ Vector
schemas.create.requested  / schemas.create.completed   ↔ Vector
stats.requested           / stats.completed            ↔ Vector
vocabulary.requested      / vocabulary.completed       ↔ Embedding
embed.text.requested      / embed.text.completed       ↔ Embedding
embed.tags.requested      / embed.tags.completed       ↔ Embedding
"""

# ── Pipeline (write path) ─────────────────────────────────────────────
IMAGE_UPLOADED = "image.uploaded"
INFERENCE_COMPLETED = "inference.completed"
DOCUMENT_STORED = "document.stored"
VECTOR_COMPUTED = "vector.computed"
EMBEDDING_INDEXED = "embedding.indexed"

# ── Search path ───────────────────────────────────────────────────────
SEARCH_REQUESTED = "search.requested"
VECTOR_SEARCH_REQUESTED = "vector.search.requested"
SEARCH_COMPLETED = "search.completed"

# ── Read request/reply pairs ──────────────────────────────────────────
DOCUMENTS_LIST_REQUESTED = "documents.list.requested"
DOCUMENTS_LIST_COMPLETED = "documents.list.completed"
DOCUMENT_GET_REQUESTED = "documents.get.requested"
DOCUMENT_GET_COMPLETED = "documents.get.completed"

EMBEDDING_GET_REQUESTED = "embeddings.get.requested"
EMBEDDING_GET_COMPLETED = "embeddings.get.completed"

SCHEMAS_LIST_REQUESTED = "schemas.list.requested"
SCHEMAS_LIST_COMPLETED = "schemas.list.completed"
SCHEMA_CREATE_REQUESTED = "schemas.create.requested"
SCHEMA_CREATE_COMPLETED = "schemas.create.completed"

STATS_REQUESTED = "stats.requested"
STATS_COMPLETED = "stats.completed"

VOCABULARY_REQUESTED = "vocabulary.requested"
VOCABULARY_COMPLETED = "vocabulary.completed"

EMBED_TEXT_REQUESTED = "embed.text.requested"
EMBED_TEXT_COMPLETED = "embed.text.completed"
EMBED_TAGS_REQUESTED = "embed.tags.requested"
EMBED_TAGS_COMPLETED = "embed.tags.completed"


ALL_TOPICS = (
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
    DOCUMENT_STORED,
    VECTOR_COMPUTED,
    EMBEDDING_INDEXED,
    SEARCH_REQUESTED,
    VECTOR_SEARCH_REQUESTED,
    SEARCH_COMPLETED,
    DOCUMENTS_LIST_REQUESTED,
    DOCUMENTS_LIST_COMPLETED,
    DOCUMENT_GET_REQUESTED,
    DOCUMENT_GET_COMPLETED,
    EMBEDDING_GET_REQUESTED,
    EMBEDDING_GET_COMPLETED,
    SCHEMAS_LIST_REQUESTED,
    SCHEMAS_LIST_COMPLETED,
    SCHEMA_CREATE_REQUESTED,
    SCHEMA_CREATE_COMPLETED,
    STATS_REQUESTED,
    STATS_COMPLETED,
    VOCABULARY_REQUESTED,
    VOCABULARY_COMPLETED,
    EMBED_TEXT_REQUESTED,
    EMBED_TEXT_COMPLETED,
    EMBED_TAGS_REQUESTED,
    EMBED_TAGS_COMPLETED,
)

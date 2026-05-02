"""Redis pub/sub topic (channel) name constants.

Topics are the contract between services. Publishing and subscribing services
must agree on the topic name. Keeping the names here (and nowhere else)
prevents typos and provides a single source of truth for the system's
event vocabulary.

Publish / subscribe map
-----------------------
image.uploaded            pub: Upload Service       sub: Inference Service
inference.completed       pub: Inference Service    sub: Document DB, Embedding
document.stored           pub: Document DB Service  sub: Web UI (audit), tests
vector.computed           pub: Embedding Service    sub: Vector Service
embedding.indexed         pub: Vector Service       sub: Web UI (audit), tests
search.requested          pub: Web UI               sub: Embedding Service
vector.search.requested   pub: Embedding Service    sub: Vector Service
search.completed          pub: Vector Service       sub: Web UI
"""

IMAGE_UPLOADED = "image.uploaded"
INFERENCE_COMPLETED = "inference.completed"
DOCUMENT_STORED = "document.stored"
VECTOR_COMPUTED = "vector.computed"
EMBEDDING_INDEXED = "embedding.indexed"
SEARCH_REQUESTED = "search.requested"
VECTOR_SEARCH_REQUESTED = "vector.search.requested"
SEARCH_COMPLETED = "search.completed"

ALL_TOPICS = (
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
    DOCUMENT_STORED,
    VECTOR_COMPUTED,
    EMBEDDING_INDEXED,
    SEARCH_REQUESTED,
    VECTOR_SEARCH_REQUESTED,
    SEARCH_COMPLETED,
)

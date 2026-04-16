"""Redis pub/sub topic (channel) name constants.

Topics are the contract between services. Publishing and subscribing services
must agree on the topic name. Keeping the names here (and nowhere else)
prevents typos and provides a single source of truth for the system's
event vocabulary.

Publish / subscribe map
-----------------------
image.uploaded         pub: Upload Service        sub: Inference Service
inference.completed    pub: Inference Service     sub: Document DB, Embedding
document.stored        pub: Document DB Service   sub: Web UI (audit), tests
embedding.indexed      pub: Embedding Service     sub: Web UI (audit), tests
search.requested       pub: Web UI                sub: Embedding Service
search.completed       pub: Embedding Service     sub: Web UI
"""

IMAGE_UPLOADED = "image.uploaded"
INFERENCE_COMPLETED = "inference.completed"
DOCUMENT_STORED = "document.stored"
EMBEDDING_INDEXED = "embedding.indexed"
SEARCH_REQUESTED = "search.requested"
SEARCH_COMPLETED = "search.completed"

ALL_TOPICS = (
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
    DOCUMENT_STORED,
    EMBEDDING_INDEXED,
    SEARCH_REQUESTED,
    SEARCH_COMPLETED,
)

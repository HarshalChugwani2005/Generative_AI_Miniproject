"""
rag.py - Retrieval-Augmented Generation pipeline.

Steps:
  1. Load the knowledge base (data/faq.md).
  2. Split it into ~500 character chunks with 50 characters of overlap.
  3. Embed the chunks with the local sentence-transformers model
     "all-MiniLM-L6-v2" (free, runs on CPU, no API key).
  4. Store the vectors in a FAISS index. The default ShopEasy index is saved
     to the vectorstore/ folder and loaded from disk on later runs; it is
     rebuilt automatically if faq.md (or the chunk/embedding settings) change.
  5. For each question, retrieve the top-k most similar chunks.
"""

import hashlib
import json
import logging
import warnings
from functools import lru_cache
from pathlib import Path

# langchain-community prints a "being sunset" DeprecationWarning on import.
# Its FAISS wrapper still works fine, so we hide that notice.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
warnings.filterwarnings("ignore", message=".*langchain-community.*")

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

import config  # noqa: E402

logger = logging.getLogger(__name__)

# Small JSON file stored next to the index that records what it was built from.
META_FILE = "index_meta.json"


# --------------------------------------------------------------------------
# Loading & chunking
# --------------------------------------------------------------------------
def load_knowledge_base(path: Path = config.FAQ_PATH) -> str:
    """Read the FAQ markdown file as UTF-8 text (explicit encoding for Windows)."""
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found at {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Knowledge base file {path} is empty")
    return text


def split_into_chunks(text: str, source: str = config.FAQ_PATH.name) -> list[Document]:
    """
    Split text into overlapping chunks.

    The separators are ordered so the splitter prefers to cut at a new
    question ("### Q:") first, then at section headings, paragraphs, lines
    and finally words. This keeps each Q&A pair together in one chunk, and
    still works well for ordinary documents without headings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n### ", "\n## ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.create_documents([text], metadatas=[{"source": source}])

    # Record a title for each chunk (used to show "Sources" in the UI).
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["title"] = _extract_title(chunk.page_content)
    return chunks


def _extract_title(chunk_text: str) -> str:
    """Return the first heading/question line of a chunk, for display purposes."""
    for line in chunk_text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip().removeprefix("Q:").strip()
    first = chunk_text.strip().splitlines()[0] if chunk_text.strip() else "Document excerpt"
    return first[:80] + ("…" if len(first) > 80 else "")


# --------------------------------------------------------------------------
# Embeddings & vector store
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Create the local embedding model (cached, so it is loaded only once).

    The model (~90 MB) is downloaded from Hugging Face the first time and then
    cached, so later runs work offline.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine-similarity friendly
    )


def _fingerprint(text: str) -> str:
    """Hash of the FAQ content + settings. If it changes, the index is rebuilt."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(f"|{config.EMBEDDING_MODEL}|{config.CHUNK_SIZE}|{config.CHUNK_OVERLAP}".encode())
    return h.hexdigest()


def _index_is_fresh(index_dir: Path, fingerprint: str) -> bool:
    """True if a saved index exists and was built from the same FAQ + settings."""
    meta_path = index_dir / META_FILE
    if not (index_dir / "index.faiss").exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("fingerprint") == fingerprint


def build_or_load_vectorstore(
    index_dir: Path = config.VECTORSTORE_DIR, force_rebuild: bool = False
) -> FAISS:
    """
    Return the FAISS vector store for the default ShopEasy knowledge base.

    Loads the cached index from `index_dir` when it is up to date, otherwise
    (re)builds it from faq.md and saves it to disk.
    """
    text = load_knowledge_base()
    fingerprint = _fingerprint(text)
    embeddings = get_embeddings()

    if not force_rebuild and _index_is_fresh(index_dir, fingerprint):
        try:
            logger.info("Loading cached FAISS index from %s", index_dir)
            # allow_dangerous_deserialization is required by LangChain to load the
            # pickled docstore. It is safe here because we created the file ourselves.
            return FAISS.load_local(
                str(index_dir), embeddings, allow_dangerous_deserialization=True
            )
        except Exception as exc:  # corrupted/incompatible cache -> rebuild
            logger.warning("Could not load cached index (%s). Rebuilding.", exc)

    logger.info("Building FAISS index from %s", config.FAQ_PATH)
    chunks = split_into_chunks(text)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(index_dir))
        (index_dir / META_FILE).write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "num_chunks": len(chunks),
                    "embedding_model": config.EMBEDDING_MODEL,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Saved FAISS index with %d chunks", len(chunks))
    except OSError as exc:  # read-only file systems (some cloud hosts) still work
        logger.warning("Could not save FAISS index to disk: %s", exc)
    return vectorstore


def count_chunks(vectorstore: FAISS) -> int:
    """Number of chunks stored in a FAISS index."""
    return int(vectorstore.index.ntotal)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def retrieve(vectorstore: FAISS, query: str, k: int = config.TOP_K) -> list[Document]:
    """Return the k chunks most similar to the query."""
    if not query.strip():
        return []
    return vectorstore.similarity_search(query, k=k)


def format_context(docs: list[Document]) -> str:
    """Join retrieved chunks into one context string for the prompt."""
    if not docs:
        return "(no relevant information found in the knowledge base)"
    return "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)


def build_search_query(question: str, previous_user_message: str | None) -> str:
    """
    Improve retrieval for short follow-up questions.

    A follow-up like "and how long does that take?" has little meaning on its
    own, so for short questions we also include the previous user message in
    the search query.
    """
    if previous_user_message and len(question.split()) <= 6:
        return f"{previous_user_message} {question}"
    return question


if __name__ == "__main__":
    # Quick manual test:  python rag.py
    logging.basicConfig(level=logging.INFO)
    store = build_or_load_vectorstore()
    print("chunks:", count_chunks(store))
    for q in ["How long does a refund take?", "Can I pay with UPI?", "My OTP is not coming"]:
        print(f"\n=== {q}")
        for d in retrieve(store, q):
            print(" -", d.metadata.get("title"))

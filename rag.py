"""Embedding-based RAG for the Sports Chatbot.

Documents are kept in a small JSON catalogue for readability and backup. Their
embeddings are stored in a persisted local FAISS index, which is used for every
retrieval query.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

KNOWLEDGE_BASE_PATH = Path("sports_knowledge_db.json")
VECTOR_STORE_DIR = Path("sports_vector_store")
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "sports.index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVAL = 3


class RAGSystem:
    """Store document embeddings in FAISS and retrieve them semantically."""

    _embedding_model = None

    def __init__(self) -> None:
        self.documents: List[Dict] = []
        self.index = None
        self.load_knowledge_base()
        self._load_or_rebuild_index()

    @classmethod
    def _get_embedding_model(cls):
        """Load the embedding model only when RAG is initialised."""
        if cls._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Embedding RAG requires sentence-transformers. Run: "
                    "pip install -r requirements.txt"
                ) from exc
            # The model is downloaded during setup. Loading from the local cache
            # keeps normal retrieval independent of Hugging Face network access.
            cls._embedding_model = SentenceTransformer(
                EMBEDDING_MODEL,
                
            )
        return cls._embedding_model

    @staticmethod
    def _get_faiss():
        try:
            import faiss
            return faiss
        except ImportError as exc:
            raise RuntimeError(
                "Embedding RAG requires faiss-cpu. Run: pip install -r requirements.txt"
            ) from exc

    def load_knowledge_base(self) -> None:
        """Load source documents, or initialise the catalogue with starter facts."""
        try:
            if KNOWLEDGE_BASE_PATH.exists():
                with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as file_handle:
                    self.documents = json.load(file_handle).get("documents", [])
                if self.documents:
                    return
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not load knowledge base: {exc}")

        self.documents = self._create_default_sports_knowledge()
        self.save_knowledge_base()

    @staticmethod
    def _create_default_sports_knowledge() -> List[Dict]:
        facts = [
            "The NBA (National Basketball Association) consists of 30 teams divided into Eastern and Western conferences.",
            "The NFL (National Football League) has 32 teams. The Super Bowl is the championship game played in February.",
            "The Premier League is the top tier of English football with 20 teams competing each season.",
            "Cricket has multiple formats: Test (5 days), ODI (One Day International, 50 overs per side), and T20 (20 overs per side).",
            "The FIFA World Cup is held every 4 years and is the largest international football tournament.",
            "Tennis Grand Slam tournaments are Wimbledon, US Open, Australian Open, and French Open.",
            "Formula 1 season typically runs from March to December with races on different circuits worldwide.",
        ]
        return [
            {
                "id": index,
                "content": fact,
                "source": "default_sports_knowledge",
                "metadata": {"type": "sports_fact"},
            }
            for index, fact in enumerate(facts)
        ]

    def _embed(self, texts: List[str]) -> np.ndarray:
        model = self._get_embedding_model()
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype="float32")
        # Inner product over unit vectors is cosine similarity.
        self._get_faiss().normalize_L2(vectors)
        return vectors

    def _load_or_rebuild_index(self) -> None:
        faiss = self._get_faiss()
        if FAISS_INDEX_PATH.exists():
            try:
                index = faiss.read_index(str(FAISS_INDEX_PATH))
                if index.ntotal == len(self.documents):
                    self.index = index
                    return
            except Exception as exc:
                print(f"Could not load FAISS index; rebuilding it: {exc}")
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Create and persist a FAISS index from every document chunk."""
        faiss = self._get_faiss()
        if not self.documents:
            self.index = None
            return
        vectors = self._embed([document.get("content", "") for document in self.documents])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

    def save_knowledge_base(self) -> None:
        """Save the source-document catalogue; vectors live in sports_vector_store/."""
        with KNOWLEDGE_BASE_PATH.open("w", encoding="utf-8") as file_handle:
            json.dump({"documents": self.documents}, file_handle, ensure_ascii=False, indent=2)

    def add_documents(self, documents: List[Dict]) -> None:
        if not documents:
            return
        max_id = max((document.get("id", 0) for document in self.documents), default=0)
        for offset, document in enumerate(documents, start=1):
            document["id"] = max_id + offset
            document.setdefault("metadata", {})
        self.documents.extend(documents)
        self.save_knowledge_base()
        self._rebuild_index()

    def add_text(self, text: str, source: str = "manual_input") -> None:
        if text.strip():
            self.add_documents([{
                "content": text.strip(),
                "source": source,
                "metadata": {"type": "user_generated"},
            }])

    @staticmethod
    def _chunks(content: str):
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for start in range(0, len(content), step):
            chunk = content[start:start + CHUNK_SIZE].strip()
            if chunk:
                yield chunk

    def add_text_file(self, file_path: str, source_name: Optional[str] = None) -> None:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            content = file_handle.read()
        self.add_documents([{
            "content": chunk,
            "source": f"file:{source_name or Path(file_path).name}",
            "metadata": {"type": "user_generated"},
        } for chunk in self._chunks(content)])

    def add_pdf_file(self, file_path: str, source_name: Optional[str] = None) -> None:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF uploads require pypdf. Run: pip install -r requirements.txt") from exc
        reader = PdfReader(file_path)
        content = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.add_documents([{
            "content": chunk,
            "source": f"pdf:{source_name or Path(file_path).name}",
            "metadata": {"type": "user_generated"},
        } for chunk in self._chunks(content)])

    def retrieve_with_scores(self, query: str, k: int = K_RETRIEVAL) -> List[Tuple[Dict, float]]:
        if not query.strip() or self.index is None:
            return []
        scores, positions = self.index.search(self._embed([query]), min(k, len(self.documents)))
        return [
            (self.documents[position], float(score))
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]

    def retrieve(self, query: str, k: int = K_RETRIEVAL) -> List[Dict]:
        return [document for document, _ in self.retrieve_with_scores(query, k)]

    def clear_knowledge_base(self) -> None:
        self.documents = self._create_default_sports_knowledge()
        self.save_knowledge_base()
        self._rebuild_index()


class SportsDataIntegrator:
    """Placeholder for future real-time sports API integrations."""

    @staticmethod
    def fetch_sports_data(sport: str, query: str) -> Optional[str]:
        return f"Sports data for {sport}: {query}"

    @staticmethod
    def add_api_data_to_knowledge_base(rag_system: RAGSystem, sport: str) -> None:
        data = SportsDataIntegrator.fetch_sports_data(sport, "general information")
        if data:
            rag_system.add_text(data, source=f"{sport}_api")


rag_system: Optional[RAGSystem] = None


def initialize_rag() -> RAGSystem:
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem()
    return rag_system


def get_rag_context(query: str, k: int = K_RETRIEVAL) -> str:
    return "\n".join(f"- {document.get('content', '')}" for document in initialize_rag().retrieve(query, k))


def get_rag_context_with_scores(query: str, k: int = K_RETRIEVAL) -> str:
    return "\n".join(
        f"- {document.get('content', '')} (similarity: {score:.2f})"
        for document, score in initialize_rag().retrieve_with_scores(query, k)
    )

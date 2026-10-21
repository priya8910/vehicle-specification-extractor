from __future__ import annotations

from pathlib import Path
from typing import Any


class ChromaVectorStore:
    def __init__(
        self,
        *,
        persist_directory: str | Path,
        collection_name: str,
        embedding_model: str,
        create: bool,
    ) -> None:
        try:
            import chromadb
            from chromadb.errors import NotFoundError
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Vector search dependencies are unavailable. Install this "
                "project's requirements."
            ) from error

        self.persist_directory = Path(persist_directory).expanduser().resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except NotFoundError:
            if not create:
                raise FileNotFoundError(
                    f"No index exists for this manual in {self.persist_directory}. "
                    "Run the ingest command first."
                )
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Vehicle workshop manual chunks",
                    "embedding_model": embedding_model,
                    "hnsw:space": "cosine",
                },
            )

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        valid_chunks = [
            chunk
            for chunk in chunks
            if chunk.get("id") and str(chunk.get("text", "")).strip()
        ]
        if not valid_chunks:
            return 0

        ids = [str(chunk["id"]) for chunk in valid_chunks]
        documents = [str(chunk["text"]).strip() for chunk in valid_chunks]
        metadatas = [
            self._sanitize_metadata(dict(chunk.get("metadata", {})))
            for chunk in valid_chunks
        ]
        embeddings = self.embedding_model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(ids)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.embedding_model.encode(
            [query.strip()],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        query_options: dict[str, Any] = {
            "query_embeddings": query_embedding,
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_options["where"] = where
        results = self.collection.query(**query_options)

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        retrieved: list[dict[str, Any]] = []
        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            clean_metadata = metadata or {}
            retrieved.append(
                {
                    "id": str(chunk_id),
                    "text": str(document or ""),
                    "metadata": clean_metadata,
                    "distance": float(distance),
                    "similarity": score,
                    "source": {
                        "file_name": clean_metadata.get("source_file", "Unknown PDF"),
                        "page_number": clean_metadata.get("page_number"),
                        "section": clean_metadata.get(
                            "section",
                            "Unlabeled section",
                        ),
                    },
                }
            )
        return retrieved

    @staticmethod
    def _sanitize_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, str | int | float | bool]:
        clean: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                clean[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean

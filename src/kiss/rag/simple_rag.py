# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Simple and elegant RAG system for document storage and retrieval using in-memory vector store."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import model


class SimpleRAG:
    """Simple and elegant RAG system for document storage and retrieval.

    This class provides a clean interface for:
    - Storing documents with embeddings in an in-memory vector store
    - Querying similar documents using vector similarity search
    - Managing document collections

    The implementation uses numpy for efficient vector operations and cosine similarity
    for finding the most relevant documents.

    Example:
        ```python
        from kiss.rag import SimpleRAG
        from kiss.core.models import OpenAIModel

        # Initialize RAG system with a model
        model = OpenAIModel("gpt-4")
        rag = SimpleRAG(model=model)

        # Add documents
        documents = [
            {
                "id": "1",
                "text": "Python is a programming language",
                "metadata": {"topic": "programming"},
            },
            {"id": "2", "text": "Machine learning uses algorithms", "metadata": {"topic": "ML"}},
        ]
        rag.add_documents(documents)

        # Query similar documents
        results = rag.query("What is Python?", top_k=2)
        for result in results:
            print(f"Text: {result['text']}, Score: {result['score']}")
        ```
    """

    def __init__(
        self,
        model_name: str,
        metric: str = "cosine",
        embedding_model_name: str | None = None,
    ):
        """Initialize the RAG system.

        Args:
            model_name: Model name to use for the LLM provider.
            metric: Distance metric to use - "cosine" or "l2" (default: "cosine").
            embedding_model_name: Optional specific model name for embeddings.
                                If None, uses model_name or provider default.
        """
        self.metric = metric
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self._model = model(model_name)  # Cache model instance
        self.documents: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a given text.

        Args:
            text: Text to generate embedding for.

        Returns:
            NumPy array representing the embedding vector.
        """
        try:
            # Use specific embedding model if provided, otherwise let provider decide
            # or use model_name if provider requires it
            embedding = self._model.get_embedding(text, embedding_model=self.embedding_model_name)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            raise KISSError(f"Failed to generate embedding: {e}") from e

    def add_documents(self, documents: list[dict[str, Any]], batch_size: int = 100) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of document dictionaries. Each document should have:
                - "id": Unique identifier (str)
                - "text": Document text content (str)
                - "metadata": Optional metadata dictionary (dict)
            batch_size: Number of documents to process in each batch (default: 100).
        """
        if not documents:
            return

        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self._add_batch(batch)

    def _add_batch(self, documents: list[dict[str, Any]]) -> None:
        """Add a batch of documents to the vector store.

        Args:
            documents: List of document dictionaries.
        """
        batch_embeddings = []

        for doc in documents:
            if "id" not in doc or "text" not in doc:
                raise KISSError("Each document must have 'id' and 'text' fields.")

            # Check for duplicate IDs
            if any(d["id"] == doc["id"] for d in self.documents):
                raise KISSError(f"Document with id '{doc['id']}' already exists.")

            # Generate embedding
            embedding = self._generate_embedding(doc["text"])
            batch_embeddings.append(embedding)

            # Store document
            self.documents.append(
                {
                    "id": str(doc["id"]),
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                }
            )

        # Update embeddings matrix
        batch_embeddings_array = np.array(batch_embeddings, dtype=np.float32)
        if self.embeddings is None:
            self.embeddings = batch_embeddings_array
        else:
            self.embeddings = np.vstack([self.embeddings, batch_embeddings_array])

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Query similar documents from the collection.

        Args:
            query_text: Query text to search for.
            top_k: Number of top results to return (default: 5).
            filter_fn: Optional filter function that takes a document dict and returns bool.

        Returns:
            List of dictionaries containing:
                - "id": Document ID
                - "text": Document text
                - "metadata": Document metadata
                - "score": Similarity score (higher is better for cosine, lower for L2)
        """
        if not self.documents or self.embeddings is None:
            return []

        # Generate query embedding
        query_embedding = self._generate_embedding(query_text)

        # Calculate similarities
        if self.metric == "cosine":
            # Cosine similarity: normalize vectors and compute dot product
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            doc_norms = self.embeddings / (
                np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
            )
            similarities = np.dot(doc_norms, query_norm)
            # Higher is better for cosine similarity
            scores = similarities
        elif self.metric == "l2":
            # L2 distance: compute Euclidean distance
            distances = np.linalg.norm(self.embeddings - query_embedding, axis=1)
            # Lower is better for L2 distance, so we negate for consistency
            scores = -distances
        else:
            raise KISSError(f"Unknown metric: {self.metric}. Use 'cosine' or 'l2'.")

        # Get all indices sorted by score (descending)
        sorted_indices = np.argsort(scores)[::-1]

        # Format results, applying filter and collecting up to top_k
        results: list[dict[str, Any]] = []
        for idx in sorted_indices:
            if len(results) >= top_k:
                break

            doc = self.documents[idx]
            # Apply filter if provided - filter BEFORE adding to results
            if filter_fn is not None and not filter_fn(doc):
                continue

            score = float(scores[idx])
            # For cosine, score is already similarity (0-1)
            # For L2, we negated it, so we need to convert back for display
            if self.metric == "l2":
                display_score = -score  # Convert back to distance
            else:
                display_score = score

            results.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": display_score,
                }
            )

        return results

    def delete_documents(self, document_ids: list[str]) -> None:
        """Delete documents from the collection by their IDs.

        Args:
            document_ids: List of document IDs to delete.
        """
        if not document_ids:
            return

        if self.embeddings is None:
            return

        # Find indices to remove
        ids_to_remove = set(str(doc_id) for doc_id in document_ids)
        indices_to_keep = [
            i for i, doc in enumerate(self.documents) if doc["id"] not in ids_to_remove
        ]

        if len(indices_to_keep) == len(self.documents):
            # No documents to remove
            return

        # Update documents and embeddings
        self.documents = [self.documents[i] for i in indices_to_keep]
        self.embeddings = self.embeddings[indices_to_keep]

        # If no documents left, reset embeddings
        if not self.documents:
            self.embeddings = None

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Dictionary containing collection statistics.
        """
        embedding_dimension = None
        if self.embeddings is not None and len(self.embeddings) > 0:
            embedding_dimension = int(self.embeddings.shape[1])
        return {
            "num_documents": len(self.documents),
            "embedding_dimension": embedding_dimension,
            "metric": self.metric,
        }

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.documents = []
        self.embeddings = None

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Get a document by its ID.

        Args:
            document_id: Document ID to retrieve.

        Returns:
            Document dictionary or None if not found.
        """
        for doc in self.documents:
            if doc["id"] == str(document_id):
                return doc
        return None

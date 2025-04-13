import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any
from pathlib import Path
import logging

class ChromaDBRetriever:
    """Retrieves relevant documents from ChromaDB based on a search phrase."""

    def __init__(self, embedding_model_name: str,
                 collection_name: str,
                 vectordb_dir: str,
                 score_threshold: float = 0.5):

        self.vectordb_path = Path(vectordb_dir)
        self.client = chromadb.PersistentClient(path=str(self.vectordb_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.score_threshold = score_threshold  # Minimum similarity score for valid results

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized ChromaDBRetriever: embedding_model_name: {embedding_model_name}, collection_name: {collection_name}, score_threshold: {score_threshold}")

    def embed_text(self, text: str) -> List[float]:
        """Generates an embedding vector for the input text."""
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()

    def extract_context(self, full_text: str, search_str: str) -> str:
        """
        Extracts the paragraph that contains the search term.
        Falls back to the entire text if no match is found.
        """
        paragraphs = full_text.split("\n\n")  # Split by paragraph
        for para in paragraphs:
            if search_str.lower() in para.lower():
                return para.strip()
        return full_text[:300]  # Fallback: Return the first 300 characters if no match

    def query(self, search_phrase: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB collection and returns structured results.
        Filters low-score matches and prioritizes the most relevant document.
        """
        embedding_vector = self.embed_text(search_phrase)
        results = self.collection.query(query_embeddings=[embedding_vector], n_results=top_k)

        # Parse results
        retrieved_docs = []
        query_words = set(search_phrase.lower().split())

        for doc_id, metadata, distance in zip(results.get("ids", [[]])[0], results.get("metadatas", [[]])[0], results.get("distances", [[]])[0]):
            if distance < self.score_threshold:
                continue  # Skip low-confidence matches

            text = metadata.get("text", "")
            extracted_context = self.extract_context(text, search_phrase)

            # Ensure at least one query word is in the retrieved text
            if not any(word in text.lower() for word in query_words):
                continue  # Skip irrelevant documents

            retrieved_docs.append({
                "id": doc_id,
                "score": round(distance, 4),
                "context": extracted_context,
                "source": metadata.get("source", "Unknown"),
            })

        # Sort by score (lower is better in similarity searches)
        retrieved_docs.sort(key=lambda x: x["score"])

        # Keep only the **most relevant** document
        return retrieved_docs[:1] if retrieved_docs else []

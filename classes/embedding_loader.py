import logging
from pathlib import Path
import chromadb
import json
from typing import List

class EmbeddingLoader:
    def __init__(self,
                 cleaned_text_file_list: List[str],
                 cleaned_text_dir: str,
                 embeddings_dir: str,
                 vectordb_dir: str,
                 collection_name: str,
                 batch_size: int = 16):

        self.cleaned_text_file_list = cleaned_text_file_list
        self.cleaned_text_path = Path(cleaned_text_dir)
        self.embeddings_path = Path(embeddings_dir)
        self.vectordb_path = Path(vectordb_dir)
        self.collection_name = collection_name
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.vectordb_path))
        self.collection = self.client.get_or_create_collection(collection_name)

    def _load_cleaned_text(self, file_path: Path) -> str:
        """Loads the cleaned text from a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Error loading text file {file_path}: {e}")
            return ""

    def _load_embeddings(self, file_path: Path) -> List[float]:
        """Loads embeddings from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                embeddings = json.load(f)
                if isinstance(embeddings, list) and all(isinstance(e, float) for e in embeddings):
                    return embeddings
                    raise ValueError("Invalid embedding format.")
                return [embeddings]  # Wrap in list for consistency
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing embeddings file {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading embeddings file {file_path}: {e}")

        return []

    def process_files(self):
        """Processes and stores cleaned text and embeddings into ChromaDB."""
        for cleaned_text_file in self.cleaned_text_file_list:
            cleaned_text_file_path = self.cleaned_text_path / cleaned_text_file
            embedding_file_path = self.embeddings_path / f"{Path(cleaned_text_file).stem}_embeddings.json"

            if not cleaned_text_file_path.exists():
                self.logger.warning(f"Missing cleaned text file: {cleaned_text_file_path}")
                continue
            if not embedding_file_path.exists():
                self.logger.warning(f"Missing embedding file for {cleaned_text_file}, skipping.")
                continue

            text = self._load_cleaned_text(cleaned_text_file_path)
            embeddings = self._load_embeddings(embedding_file_path)

            if not text or not embeddings:
                self.logger.warning(f"Skipping {cleaned_text_file} due to missing text or embeddings.")
                continue

            self.logger.info(f"Storing {cleaned_text_file} in ChromaDB...")

            self.collection.add(
                ids=[cleaned_text_file],
                embeddings=[embeddings],
                metadatas=[{"text": text, "source": cleaned_text_file}]
            )

            self.logger.info(f"Stored {cleaned_text_file} successfully.")

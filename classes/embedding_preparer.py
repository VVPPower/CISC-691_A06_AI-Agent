import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch


class EmbeddingPreparer:
    def __init__(self,
                 file_list,
                 input_dir,
                 output_dir,
                 embedding_model_name):
        self.file_list = file_list
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.embedding_model_name = embedding_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device)

    def process_files(self):
        save_log_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.INFO)

        for file_path in self.file_list:
            file_path = Path(self.input_dir/file_path)
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                continue
            try:
                self.logger.info(f"Processing: {file_path}")
                text = self._read_file(file_path)
                embedding = self._generate_embedding(text)
                self._save_embedding(file_path, embedding)
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")

        logging.getLogger().setLevel(save_log_level)

    def _read_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _generate_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(
            self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()

    def _save_embedding(self, file_path, embedding):
        output_file = self.output_dir / f"{file_path.stem}_embeddings.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(embedding, f)  # Save list of floats
        self.logger.info(f"Saved embeddings to {output_file}")



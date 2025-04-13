import logging
from pathlib import Path
import pdfplumber
from transformers import AutoTokenizer


class DocumentIngestor:
    def __init__(self,
                 file_list,
                 input_dir,
                 output_dir,
                 embedding_model_name):
        """
        Initializes the document ingestor.

        :param file_list: List of file paths to process.
        :param output_dir: Directory to save cleaned text files.
        :param model_name: Hugging Face tokenizer model for preprocessing.
        """
        self.file_list = file_list
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized DocumentIngestor: input_dir: {self.input_dir}"
                         f"output_dir: {self.output_dir}, embedding_model_name: {embedding_model_name}")

    def _extract_text_from_pdf(self, file_path):
        """Extracts text from a PDF file using pdfplumber."""
        try:
            save_log_level = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.INFO)
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            logging.getLogger().setLevel(save_log_level)
            return text
        except Exception as e:
            self.logger.error(f"Error reading PDF {file_path}: {e}")
            return None

    def _extract_text_from_txt(self, file_path):
        """Extracts text from a TXT file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading TXT {file_path}: {e}")
            return None

    def _clean_text(self, text):
        """Cleans and tokenizes text for better embedding preparation."""
        if not text:
            return None

        text = text.replace("\n", " ").strip()  # Remove excessive newlines and trim
        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def process_files(self):
        """Processes the list of files, extracts, cleans, and saves them."""
        for file_path in self.file_list:
            file_path = Path(self.input_dir/file_path)
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                continue

            self.logger.info(f"Processing file: {file_path}")

            if file_path.suffix.lower() == ".pdf":
                text = self._extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() == ".txt":
                text = self._extract_text_from_txt(file_path)
            else:
                self.logger.warning(f"Unsupported file type: {file_path.suffix}")
                continue

            cleaned_text = self._clean_text(text)
            if cleaned_text:
                output_file = self.output_dir / f"{file_path.stem}_cleaned.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                self.logger.info(f"Saved cleaned text to {output_file}")
            else:
                self.logger.warning(f"Skipping {file_path} due to extraction failure.")


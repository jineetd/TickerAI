"""Document processor for ingesting documents into ChromaDB.

Handles reading, chunking, and embedding documents from the knowledge
directory.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for vector storage."""

    def __init__(self):
        self.knowledge_dir = config.KNOWLEDGE_DIR
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP

    def read_text_file(self, file_path: Path) -> str:
        """Read text content from a file.

        Args:
            file_path: Path to the text file.

        Returns:
            The text content of the file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return ""

    def read_pdf_file(self, file_path: Path) -> str:
        """Read text content from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            The extracted text content from the PDF.
        """
        try:
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
            return ""

    def read_json_file(self, file_path: Path) -> str:
        """Read and stringify JSON content.

        Args:
            file_path: Path to the JSON file.

        Returns:
            The JSON content as a formatted string.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            return ""

    def read_document(self, file_path: Path) -> str:
        """Read document based on file type.

        Args:
            file_path: Path to the document file.

        Returns:
            The document content as a string.
        """
        suffix = file_path.suffix.lower()
        
        if suffix in ['.txt', '.md']:
            return self.read_text_file(file_path)
        elif suffix == '.pdf':
            return self.read_pdf_file(file_path)
        elif suffix == '.json':
            return self.read_json_file(file_path)
        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return ""

    def chunk_text(
            self,
            text: str,
            metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks.

        Args:
            text: The text to chunk.
            metadata: Metadata to attach to each chunk.

        Returns:
            List of chunks with text and metadata.
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Create chunk with metadata
            chunk = {
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_id': chunk_id,
                    'start_char': start,
                    'end_char': end
                }
            }
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start += self.chunk_size - self.chunk_overlap
            chunk_id += 1

        return chunks

    def process_documents(self) -> List[Dict[str, Any]]:
        """Process all documents in the knowledge directory.

        Returns:
            List of all processed document chunks with metadata.
        """
        all_chunks = []
        
        if not self.knowledge_dir.exists():
            logger.error(
                f"Knowledge directory does not exist: "
                f"{self.knowledge_dir}"
            )
            return all_chunks

        # Iterate through all files in knowledge directory
        for file_path in self.knowledge_dir.rglob('*'):
            is_supported = (
                file_path.is_file() and
                file_path.suffix in config.SUPPORTED_FILE_TYPES
            )
            if is_supported:
                logger.info(f"Processing: {file_path}")

                # Read document content
                content = self.read_document(file_path)

                if not content:
                    logger.warning(
                        f"No content extracted from {file_path}"
                    )
                    continue

                # Create metadata
                relative_path = str(
                    file_path.relative_to(self.knowledge_dir)
                )
                metadata = {
                    'filename': file_path.name,
                    'filepath': str(file_path),
                    'file_type': file_path.suffix,
                    'relative_path': relative_path
                }

                # Chunk the document
                chunks = self.chunk_text(content, metadata)
                all_chunks.extend(chunks)

                logger.info(
                    f"Created {len(chunks)} chunks from {file_path.name}"
                )

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers (local)."""

    def __init__(self):
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        try:
            # Generate embeddings using sentence-transformers
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            # Convert to list of lists
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.
        """
        try:
            embedding = self.model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

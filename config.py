"""Configuration file for TickerAI application.

Store all constants, API keys, and configuration parameters here.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama, openai, etc.

# Ollama Configuration (for local Llama models)
LLM_HOST = os.getenv("LLM_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")  # llama3.2, llama3.1, llama2, etc.

# LLM Generation Parameters
LLM_TEMPERATURE = 0.7  # 0.0 = deterministic, 1.0 = creative
LLM_MAX_TOKENS = 1000  # Maximum response length

# Embedding Model Configuration (Using sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient local embeddings
# Alternative options:
# "all-mpnet-base-v2" - Better quality, slower
# "paraphrase-multilingual-MiniLM-L12-v2" - Multilingual support

# ChromaDB Configuration
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "stock_documents"

# Knowledge Base Configuration
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
SUPPORTED_FILE_TYPES = [".txt", ".md", ".pdf", ".json"]

# MCP Server Configuration
MCP_SERVER_HOST = "localhost"
MCP_SERVER_PORT = 8765

# Vector Search Configuration
TOP_K_RESULTS = 5  # Number of similar documents to retrieve
SIMILARITY_THRESHOLD = 0.7

# Chunk Configuration for Document Processing
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "tickerai.log"

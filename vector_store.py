"""Vector store implementation using ChromaDB.

Handles document storage, retrieval, and similarity search.
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

import config
from document_processor import DocumentProcessor, EmbeddingGenerator

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorStore:
    """Manage ChromaDB vector store for document retrieval."""

    def __init__(self):
        self.chroma_dir = config.CHROMA_DB_DIR
        self.collection_name = config.CHROMA_COLLECTION_NAME
        self.embedding_generator = EmbeddingGenerator()

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self._get_or_create_collection()

        logger.info(
            f"VectorStore initialized with collection: "
            f"{self.collection_name}"
        )

    def _get_or_create_collection(self):
        """Get existing collection or create new one.

        Returns:
            ChromaDB collection object.
        """
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
            logger.info(
                f"Creating new collection: {self.collection_name}"
            )
            return self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Stock ticker documents and analysis"
                }
            )

    def reset_collection(self):
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Stock ticker documents and analysis"}
        )
        logger.info(f"Created new collection: {self.collection_name}")

    def ingest_documents(self, force_refresh: bool = False):
        """Ingest documents from knowledge directory into ChromaDB.

        Args:
            force_refresh: Whether to force re-ingestion of documents.
        """
        if force_refresh:
            self.reset_collection()

        # Check if collection already has documents
        count = self.collection.count()
        if count > 0 and not force_refresh:
            logger.info(
                f"Collection already contains {count} documents. "
                f"Use force_refresh=True to re-ingest."
            )
            return

        # Process documents
        processor = DocumentProcessor()
        chunks = processor.process_documents()

        if not chunks:
            logger.warning("No documents found to ingest")
            return

        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            documents.append(chunk['text'])
            metadatas.append(chunk['metadata'])
            ids.append(f"doc_{i}")

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(
            documents
        )

        if not embeddings:
            logger.error("Failed to generate embeddings")
            return

        # Add to ChromaDB
        logger.info(
            f"Adding {len(documents)} documents to ChromaDB..."
        )
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(
            f"Successfully ingested {len(documents)} document chunks"
        )

    def query(
            self,
            query_text: str,
            n_results: int = None
    ) -> Dict[str, Any]:
        """Query the vector store for similar documents.

        Args:
            query_text: The query string.
            n_results: Number of results to return.

        Returns:
            Dictionary containing documents, metadatas, and distances.
        """
        if n_results is None:
            n_results = config.TOP_K_RESULTS

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(
            query_text
        )

        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return {
                "documents": [],
                "metadatas": [],
                "distances": []
            }

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        logger.info(
            f"Retrieved {len(results['documents'][0])} results for query"
        )
        return results

    def get_context_for_query(
            self,
            query: str,
            ticker: Optional[str] = None
    ) -> str:
        """Get relevant context documents for a query.

        Args:
            query: The query string.
            ticker: Optional ticker symbol to filter results.

        Returns:
            Formatted context string.
        """
        # Query vector store
        results = self.query(query)

        if not results['documents'] or not results['documents'][0]:
            return "No relevant context found."

        # Format context
        context_parts = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Filter by ticker if provided
            if ticker and 'ticker' in metadata:
                if metadata['ticker'].upper() != ticker.upper():
                    continue

            relevance = 1 - distance
            context_parts.append(
                f"--- Context {i+1} (Relevance: {relevance:.2f}) ---"
            )
            context_parts.append(
                f"Source: {metadata.get('filename', 'Unknown')}"
            )
            context_parts.append(f"Content: {doc}")
            context_parts.append("")

        return "\n".join(context_parts)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "db_path": str(self.chroma_dir)
        }

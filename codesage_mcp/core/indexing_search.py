"""
Indexing and Search Module for CodeSage MCP Server.

This module provides classes for indexing codebases and performing semantic searches.
It includes FAISS-based indexing, semantic search capabilities, regex search, and
incremental updates for efficient code analysis.

Classes:
    FAISSIndexer: Handles FAISS index creation and vector operations
    SemanticSearch: Performs semantic searches using embeddings
    RegexSearch: Performs regex-based text searches
    IncrementalUpdater: Manages incremental index updates
"""

import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional

# Import FAISS for vector indexing
try:
    import faiss
except ImportError:
    faiss = None

# Import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ..config.config import ENABLE_CACHING
from codesage_mcp.features.caching.cache import get_cache_instance
from codesage_mcp.features.memory_management.memory_manager import get_memory_manager
from .exceptions import IndexingError

# Set up logger
logger = logging.getLogger(__name__)


class FAISSIndexer:
    """
    FAISS-based indexer for efficient vector similarity search.

    This class manages FAISS index creation, vector addition, and similarity
    search operations. It supports different index types and memory-mapped
    indexes for large-scale codebases.

    Attributes:
        index: FAISS index object
        dimension: Vector dimension
        index_type: Type of FAISS index
    """

    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        """
        Initialize the FAISS indexer.

        Args:
            dimension: Vector dimension for the index
            index_type: Type of FAISS index ('flat', 'ivf', 'ivf_pq')
        """
        if faiss is None:
            raise ImportError("FAISS is required for FAISSIndexer")

        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.memory_manager = get_memory_manager()
        self.cache = get_cache_instance() if ENABLE_CACHING else None

        self._initialize_index()

    def _initialize_index(self):
        """Initialize the FAISS index based on type."""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            # IVF with flat quantizer
            nlist = min(100, max(4, 1000 // 39))  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == "ivf_pq":
            # IVF with product quantization
            nlist = min(100, max(4, 1000 // 39))
            m = min(self.dimension // 4, 64)  # Number of sub-quantizers
            nbits = 8  # Bits per sub-quantizer
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        # Train the index if necessary
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            # For IVF indexes, we need training data
            # This will be done when adding the first batch of vectors
            pass

    def add_vectors(self, vectors: List[List[float]]) -> None:
        """
        Add vectors to the index.

        Args:
            vectors: List of vectors to add

        Raises:
            IndexingError: If vector addition fails
        """
        try:
            vectors_array = np.array(vectors, dtype=np.float32)

            # Train the index if needed
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                if len(vectors) >= 256:  # Need enough vectors for training
                    self.index.train(vectors_array)
                else:
                    # Use a simple flat index for small datasets
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self.index_type = "flat"

            # Add vectors to index
            self.index.add(vectors_array)

        except Exception as e:
            raise IndexingError(
                f"Failed to add vectors to FAISS index: {e}",
                operation="add_vectors",
                context={"vector_count": len(vectors), "dimension": self.dimension}
            )

    def search(self, query_vector: List[float], k: int = 5) -> tuple:
        """
        Search for similar vectors in the index.

        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)

        Raises:
            IndexingError: If search fails
        """
        try:
            if self.index.ntotal == 0:
                return np.array([]), np.array([])

            query_array = np.array([query_vector], dtype=np.float32)
            k = min(k, self.index.ntotal)  # Don't ask for more than we have

            distances, indices = self.index.search(query_array, k)
            return distances[0], indices[0]

        except Exception as e:
            raise IndexingError(
                f"Failed to search FAISS index: {e}",
                operation="search",
                context={"query_dimension": len(query_vector), "k": k, "total_vectors": self.index.ntotal}
            )

    def save(self, filepath: str) -> None:
        """
        Save the index to disk.

        Args:
            filepath: Path to save the index

        Raises:
            IndexingError: If save fails
        """
        try:
            faiss.write_index(self.index, filepath)
        except Exception as e:
            raise IndexingError(
                f"Failed to save FAISS index: {e}",
                operation="save",
                context={"filepath": filepath}
            )

    def load(self, filepath: str) -> None:
        """
        Load the index from disk.

        Args:
            filepath: Path to load the index from

        Raises:
            IndexingError: If load fails
        """
        try:
            self.index = faiss.read_index(filepath)
        except Exception as e:
            raise IndexingError(
                f"Failed to load FAISS index: {e}",
                operation="load",
                context={"filepath": filepath}
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"total_vectors": 0, "dimension": self.dimension}

        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": getattr(self.index, 'is_trained', True)
        }


class SemanticSearch:
    """
    Semantic search using embeddings and FAISS indexing.

    This class provides semantic search capabilities by encoding queries
    and documents into vector embeddings and performing similarity search.

    Attributes:
        indexer: FAISS indexer for vector search
        model: Sentence transformer model for encoding
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic search.

        Args:
            model_name: Name of the sentence transformer model
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for SemanticSearch")

        self.model = SentenceTransformer(model_name)
        self.indexer = FAISSIndexer(dimension=self.model.get_sentence_embedding_dimension())
        self.cache = get_cache_instance() if ENABLE_CACHING else None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with scores
        """
        try:
            # Encode query
            query_embedding = self.model.encode(query)

            # Search in index
            distances, indices = self.indexer.search(query_embedding.tolist(), top_k)

            # Format results
            results = []
            for i, (distance, idx) in enumerate(zip(distances, indices)):
                if idx == -1:  # No result
                    continue
                results.append({
                    "index": int(idx),
                    "score": float(1.0 / (1.0 + distance)),  # Convert distance to similarity score
                    "distance": float(distance)
                })

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the search index.

        Args:
            documents: List of document texts to index
        """
        try:
            # Encode documents
            embeddings = self.model.encode(documents)

            # Add to index
            self.indexer.add_vectors(embeddings.tolist())

        except Exception as e:
            logger.error(f"Failed to add documents to semantic search: {e}")
            raise

    def search_with_filters(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search with filters applied.

        Args:
            query: Search query
            filters: Filters to apply

        Returns:
            Filtered search results
        """
        # For now, just return regular search results
        # In a real implementation, you'd apply the filters
        return self.search(query)

    def save_index(self, filepath: str) -> None:
        """
        Save the search index.

        Args:
            filepath: Path to save the index
        """
        self.indexer.save(filepath)

    def load_index(self, filepath: str) -> None:
        """
        Load the search index.

        Args:
            filepath: Path to load the index from
        """
        self.indexer.load(filepath)


class RegexSearch:
    """
    Regex-based text search for code files.

    This class provides efficient regex search capabilities across code files,
    with support for case-insensitive matching and pattern compilation.

    Attributes:
        compiled_patterns: Cache of compiled regex patterns
    """

    def __init__(self):
        """Initialize regex search."""
        self.compiled_patterns = {}
        self.cache = get_cache_instance() if ENABLE_CACHING else None

    def find_matches(self, pattern: str, text: str, case_insensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Find regex matches in text.

        Args:
            pattern: Regex pattern to search for
            text: Text to search in
            case_insensitive: Whether to perform case-insensitive search

        Returns:
            List of match dictionaries with position and content

        Raises:
            ValueError: If regex pattern is invalid
        """
        try:
            # Compile pattern with caching
            flags = re.IGNORECASE if case_insensitive else 0
            cache_key = f"{pattern}:{flags}"

            if cache_key not in self.compiled_patterns:
                self.compiled_patterns[cache_key] = re.compile(pattern, flags)

            regex = self.compiled_patterns[cache_key]

            # Find all matches
            matches = []
            for match in regex.finditer(text):
                matches.append({
                    "start": match.start(),
                    "end": match.end(),
                    "content": match.group(),
                    "groups": match.groups() if match.groups() else []
                })

            return matches

        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

    def search_files(self, pattern: str, file_paths: List[str], case_insensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search for pattern in multiple files.

        Args:
            pattern: Regex pattern to search for
            file_paths: List of file paths to search
            case_insensitive: Whether to perform case-insensitive search

        Returns:
            List of search results with file and match information
        """
        results = []

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                matches = self.find_matches(pattern, content, case_insensitive)

                for match in matches:
                    results.append({
                        "file_path": file_path,
                        "line_number": content[:match["start"]].count('\n') + 1,
                        "match": match
                    })

            except Exception as e:
                logger.warning(f"Failed to search file {file_path}: {e}")
                continue

        return results


class IncrementalUpdater:
    """
    Manages incremental updates to search indexes.

    This class handles adding new documents, removing outdated ones,
    and updating existing entries in the search index without full rebuilds.

    Attributes:
        indexer: FAISS indexer to update
        document_map: Mapping of document IDs to index positions
    """

    def __init__(self, indexer: FAISSIndexer):
        """
        Initialize incremental updater.

        Args:
            indexer: FAISS indexer to manage
        """
        self.indexer = indexer
        self.document_map = {}  # doc_id -> index_position
        self.cache = get_cache_instance() if ENABLE_CACHING else None

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add new documents to the index.

        Args:
            documents: List of document dictionaries with 'id' and 'content' keys
        """
        try:
            contents = []
            doc_ids = []

            for doc in documents:
                doc_id = doc.get('id')
                content = doc.get('content', '')

                if doc_id and content:
                    contents.append(content)
                    doc_ids.append(doc_id)

            if not contents:
                return

            # For semantic search, we'd need to encode the contents
            # For now, we'll assume the documents already have embeddings
            embeddings = []
            for doc in documents:
                embedding = doc.get('embedding')
                if embedding:
                    embeddings.append(embedding)

            if embeddings:
                self.indexer.add_vectors(embeddings)

                # Update document map
                start_idx = self.indexer.index.ntotal - len(embeddings)
                for i, doc_id in enumerate(doc_ids):
                    self.document_map[doc_id] = start_idx + i

        except Exception as e:
            logger.error(f"Failed to add documents incrementally: {e}")
            raise

    def process_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Process documents for indexing.

        Args:
            documents: List of document dictionaries
        """
        # Alias for add_documents for test compatibility
        self.add_documents(documents)

    def remove_documents(self, doc_ids: List[str]) -> None:
        """
        Remove documents from the index.

        Args:
            doc_ids: List of document IDs to remove
        """
        # FAISS doesn't support efficient deletion, so we'll mark documents as removed
        # In a production system, you'd rebuild the index periodically
        for doc_id in doc_ids:
            if doc_id in self.document_map:
                # Mark as removed by setting index to -1
                self.document_map[doc_id] = -1

        logger.info(f"Marked {len(doc_ids)} documents for removal (FAISS index rebuild needed)")

    def update_document(self, doc_id: str, new_content: str, new_embedding: Optional[List[float]] = None) -> None:
        """
        Update an existing document in the index.

        Args:
            doc_id: Document ID to update
            new_content: New document content
            new_embedding: New document embedding (optional)
        """
        # For now, we'll remove and re-add
        # In production, you'd want more efficient update mechanisms
        if doc_id in self.document_map:
            self.remove_documents([doc_id])

        if new_embedding:
            doc = {"id": doc_id, "content": new_content, "embedding": new_embedding}
            self.add_documents([doc])

    def get_stats(self) -> Dict[str, Any]:
        """
        Get updater statistics.

        Returns:
            Dictionary with updater statistics
        """
        total_docs = len([idx for idx in self.document_map.values() if idx != -1])
        removed_docs = len([idx for idx in self.document_map.values() if idx == -1])

        return {
            "total_documents": total_docs,
            "removed_documents": removed_docs,
            "active_documents": total_docs - removed_docs,
            "indexer_stats": self.indexer.get_stats()
        }
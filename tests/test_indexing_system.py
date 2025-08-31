"""
Comprehensive Unit Tests for Indexing System Module.

This module contains unit tests for the IndexingManager class and related components,
focusing on codebase indexing, incremental updates, dependency tracking, parallel processing,
and index optimization features.
"""

import pytest
import json
import time
import threading
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import tempfile
import os
import shutil
import faiss
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from codesage_mcp.indexing import IndexingManager
from codesage_mcp.config import (
    ENABLE_CACHING,
    ENABLE_MEMORY_MAPPED_INDEXES,
    INDEX_TYPE,
)
from codesage_mcp.memory_manager import get_memory_manager
from codesage_mcp.cache import get_cache_instance


class TestIndexingManager:
    """Test cases for IndexingManager class."""

    def test_initialization(self):
        """Test IndexingManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager(index_dir_name=temp_dir)

            assert manager.index_dir_name == temp_dir
            assert manager.index_dir == Path(temp_dir)
            assert manager.index_file == Path(temp_dir) / "codebase_index.json"
            assert manager.faiss_index_file == Path(temp_dir) / "codebase_index.faiss"
            assert manager.indexed_codebases == {}
            assert manager.file_paths_map == {}
            assert manager.file_metadata == {}

    def test_get_codebase_key(self):
        """Test codebase key generation."""
        manager = IndexingManager()

        key1 = manager._get_codebase_key("/home/user/project")
        key2 = manager._get_codebase_key("/home/user/project/")  # With trailing slash
        key3 = manager._get_codebase_key("./relative/path")

        assert key1 == key2  # Should be the same
        assert isinstance(key1, str)
        assert "/home/user/project" in key1

    def test_get_gitignore_patterns(self):
        """Test gitignore pattern parsing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager()

            # Create test .gitignore
            gitignore_path = Path(temp_dir) / ".gitignore"
            gitignore_path.write_text("""
# Comment
*.pyc
__pycache__/
venv/

# Empty line above
node_modules/
""")

            patterns = manager._get_gitignore_patterns(Path(temp_dir))

            assert "*.pyc" in patterns
            assert "__pycache__/" in patterns
            assert "venv/" in patterns
            assert "node_modules/" in patterns
            assert "# Comment" not in patterns
            assert "" not in patterns

    def test_is_ignored(self):
        """Test file ignoring logic."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager()

            # Create test .gitignore
            gitignore_path = Path(temp_dir) / ".gitignore"
            gitignore_path.write_text("*.pyc\n__pycache__/\n")

            root_path = Path(temp_dir)
            patterns = ["*.pyc", "__pycache__/"]

            # Test ignored files
            pyc_file = root_path / "test.pyc"
            assert manager._is_ignored(pyc_file, patterns, root_path) is True

            pycache_file = root_path / "__pycache__" / "module.pyc"
            assert manager._is_ignored(pycache_file, patterns, root_path) is True

            # Test non-ignored files
            py_file = root_path / "test.py"
            assert manager._is_ignored(py_file, patterns, root_path) is False

    def test_get_file_mtime(self):
        """Test getting file modification time."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            manager = IndexingManager()

            mtime = manager._get_file_mtime(Path(tmp_file.name))

            assert isinstance(mtime, float)
            assert mtime > 0

    def test_get_file_mtime_nonexistent(self):
        """Test getting mtime for non-existent file."""
        manager = IndexingManager()

        mtime = manager._get_file_mtime(Path("/nonexistent/file"))

        assert mtime == 0.0


class TestIndexingManagerIncremental:
    """Test cases for incremental indexing functionality."""

    def test_detect_changed_files(self):
        """Test detection of changed files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager(index_dir_name=temp_dir)

            # Create test codebase
            codebase_dir = Path(temp_dir) / "codebase"
            codebase_dir.mkdir()

            # Create files
            (codebase_dir / "existing.py").write_text("print('existing')")
            (codebase_dir / "modified.py").write_text("print('modified v1')")
            (codebase_dir / "new.py").write_text("print('new')")

            # Simulate previously indexed state
            codebase_key = str(codebase_dir.resolve())
            manager.indexed_codebases[codebase_key] = {
                "files": ["existing.py", "modified.py", "deleted.py"]
            }
            manager.file_metadata[codebase_key] = {
                "existing.py": {"mtime": 1000, "indexed_at": "2023-01-01"},
                "modified.py": {"mtime": 1000, "indexed_at": "2023-01-01"},  # Old mtime
                "deleted.py": {"mtime": 1000, "indexed_at": "2023-01-01"}
            }

            # Update modified file
            time.sleep(0.01)  # Ensure different mtime
            (codebase_dir / "modified.py").write_text("print('modified v2')")

            added, modified, deleted = manager._detect_changed_files(str(codebase_dir))

            assert "new.py" in added
            assert "modified.py" in modified
            assert "deleted.py" in deleted

    def test_remove_deleted_embeddings(self):
        """Test removing embeddings for deleted files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager(index_dir_name=temp_dir)

            # Create mock FAISS index
            dimension = 128
            manager.faiss_index = faiss.IndexFlatL2(dimension)

            # Add some test vectors
            test_vectors = np.random.rand(5, dimension).astype(np.float32)
            manager.faiss_index.add(test_vectors)

            # Set up file paths map
            manager.file_paths_map = {
                "0": "/path/to/file1.py",
                "1": "/path/to/file2.py",
                "2": "/path/to/deleted.py",
                "3": "/path/to/file3.py",
                "4": "/path/to/deleted2.py"
            }

            # Remove embeddings for deleted files
            deleted_files = {"deleted.py", "deleted2.py"}
            manager._remove_deleted_embeddings("/path/to", deleted_files)

            # Check that deleted files are removed from map
            assert "2" not in manager.file_paths_map  # deleted.py (original index 2)
            assert "4" not in manager.file_paths_map  # deleted2.py (original index 4)
            assert "0" in manager.file_paths_map  # file1.py should remain at index 0
            assert "1" in manager.file_paths_map  # file2.py should remain at index 1
            assert "3" in manager.file_paths_map  # file3.py should remain at index 3
            assert len(manager.file_paths_map) == 3  # Should have 3 files remaining

    def test_update_file_metadata(self):
        """Test updating file metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager(index_dir_name=temp_dir)

            codebase_path = "/test/codebase"
            files = ["file1.py", "file2.py"]

            manager._update_file_metadata(codebase_path, files)

            codebase_key = manager._get_codebase_key(codebase_path)
            assert codebase_key in manager.file_metadata

            for file_path in files:
                assert file_path in manager.file_metadata[codebase_key]
                metadata = manager.file_metadata[codebase_key][file_path]
                assert "mtime" in metadata
                assert "indexed_at" in metadata


class TestIndexingManagerCore:
    """Test cases for core indexing functionality."""

    def test_index_codebase_basic(self):
        """Test basic codebase indexing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager(index_dir_name=temp_dir)

            # Create test codebase
            codebase_dir = Path(temp_dir) / "codebase"
            codebase_dir.mkdir()
            (codebase_dir / "test.py").write_text("print('hello world')")
            (codebase_dir / ".gitignore").write_text("*.pyc\n")

            # Mock sentence transformer
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 128
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
                mock_chunk.return_value = [
                    MagicMock(content="print('hello world')", start_line=1, end_line=1)
                ]
                indexed_files = manager.index_codebase(str(codebase_dir), mock_model)

                assert "test.py" in indexed_files
                assert manager.faiss_index is not None
                assert manager.faiss_index.ntotal > 0

    def test_index_codebase_empty(self):
        """Test indexing empty codebase."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager(index_dir_name=temp_dir)

            # Create empty codebase
            codebase_dir = Path(temp_dir) / "empty_codebase"
            codebase_dir.mkdir()

            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 128
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            indexed_files = manager.index_codebase(str(codebase_dir), mock_model)

            assert len(indexed_files) == 0

    def test_index_codebase_incremental_no_changes(self):
        """Test incremental indexing when no changes detected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager(index_dir_name=temp_dir)

            # Create test codebase
            codebase_dir = Path(temp_dir) / "codebase"
            codebase_dir.mkdir()
            (codebase_dir / "test.py").write_text("print('hello')")

            # Mock initial indexing
            codebase_key = str(codebase_dir.resolve())
            manager.indexed_codebases[codebase_key] = {"files": ["test.py"]}

            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 128
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            indexed_files, was_incremental = manager.index_codebase_incremental(
                str(codebase_dir), mock_model
            )

            assert was_incremental is True
            assert "test.py" in indexed_files

    def test_index_codebase_incremental_force_full(self):
        """Test forcing full re-indexing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager(index_dir_name=temp_dir)

            codebase_dir = Path(temp_dir) / "codebase"
            codebase_dir.mkdir()

            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 128
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            indexed_files, was_incremental = manager.index_codebase_incremental(
                str(codebase_dir), mock_model, force_full=True
            )

            assert was_incremental is False

    def test_process_incremental_changes_batch(self):
        """Test batch processing of incremental changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager(index_dir_name=temp_dir)

            # Create test codebase
            codebase_dir = Path(temp_dir) / "codebase"
            codebase_dir.mkdir()
            (codebase_dir / "modified.py").write_text("print('modified')")

            # Mock FAISS index
            manager.faiss_index = faiss.IndexFlatL2(128)

            # Set up initial codebase metadata
            codebase_key = str(codebase_dir.resolve())
            manager.indexed_codebases[codebase_key] = {"files": ["modified.py"]}

            # Mock model
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            added_files = []
            modified_files = ["modified.py"]
            deleted_files = []

            with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
                mock_chunk.return_value = [
                    MagicMock(content="print('modified')", start_line=1, end_line=1)
                ]
                indexed_files, was_incremental = manager._process_incremental_changes_batch(
                    str(codebase_dir), mock_model, added_files, modified_files, deleted_files
                )

                assert was_incremental is True
                assert "modified.py" in indexed_files


class TestIndexingManagerDependencyTracking:
    """Test cases for dependency tracking functionality."""

    def test_analyze_file_dependencies_python(self):
        """Test analyzing Python file dependencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager()

            # Create test Python file with imports
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("""
import os
from pathlib import Path
from . import utils
""")

            dependencies = manager._analyze_file_dependencies("test.py", Path(temp_dir))

            # Should find some dependencies (exact set may vary)
            assert isinstance(dependencies, set)

    def test_resolve_module_to_path(self):
        """Test resolving module names to file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager()

            # Create test module structure
            (Path(temp_dir) / "utils.py").write_text("def helper(): pass")
            mypackage_dir = Path(temp_dir) / "mypackage"
            mypackage_dir.mkdir(parents=True)
            (mypackage_dir / "__init__.py").write_text("")

            # Test resolving different module types
            path1 = manager._resolve_module_to_path("utils", Path(temp_dir))
            path2 = manager._resolve_module_to_path("mypackage", Path(temp_dir))

            assert path1 == "utils.py"
            assert path2 == "mypackage/__init__.py"

    def test_build_dependency_graph(self):
        """Test building dependency graph for codebase."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager()

            # Create test codebase with dependencies
            codebase_dir = Path(temp_dir) / "codebase"
            codebase_dir.mkdir()

            (codebase_dir / "main.py").write_text("import utils")
            (codebase_dir / "utils.py").write_text("import helper")
            (codebase_dir / "helper.py").write_text("print('helper')")

            # Mock indexed files
            codebase_key = str(codebase_dir.resolve())
            manager.indexed_codebases[codebase_key] = {
                "files": ["main.py", "utils.py", "helper.py"]
            }

            manager._build_dependency_graph(str(codebase_dir))

            assert codebase_key in manager._dependency_graph
            assert codebase_key in manager._reverse_dependencies

    def test_get_dependent_files(self):
        """Test getting files that depend on a given file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager()

            # Set up mock dependency graph
            codebase_key = "/test/codebase"
            manager._dependency_graph[codebase_key] = {
                "utils.py": {"main.py", "test.py"}
            }
            manager._reverse_dependencies[codebase_key] = {
                "main.py": {"utils.py"},
                "test.py": {"utils.py"}
            }

            dependent_files = manager._get_dependent_files("utils.py", "/test/codebase")

            assert "main.py" in dependent_files
            assert "test.py" in dependent_files

    def test_update_dependency_graph(self):
        """Test updating dependency graph for changed files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager()

            codebase_dir = Path(temp_dir) / "codebase"
            codebase_dir.mkdir()
            (codebase_dir / "main.py").write_text("import utils")
            (codebase_dir / "utils.py").write_text("def helper(): pass")

            changed_files = ["main.py"]
            manager._update_dependency_graph(str(codebase_dir), changed_files)

            codebase_key = manager._get_codebase_key(str(codebase_dir))
            assert codebase_key in manager._dependency_graph


class TestIndexingManagerParallelProcessing:
    """Test cases for parallel processing functionality."""

    def test_should_use_parallel_processing_small(self):
        """Test parallel processing decision for small workloads."""
        manager = IndexingManager()

        use_parallel = manager._should_use_parallel_processing(5)

        assert use_parallel is False

    def test_should_use_parallel_processing_large(self):
        """Test parallel processing decision for large workloads."""
        manager = IndexingManager()

        use_parallel = manager._should_use_parallel_processing(50)

        assert use_parallel is True

    def test_get_thread_pool(self):
        """Test thread pool creation and retrieval."""
        manager = IndexingManager()

        executor = manager._get_thread_pool()

        assert isinstance(executor, ThreadPoolExecutor)
        assert executor is manager._executor

        # Cleanup
        manager.cleanup()

    def test_process_file_batch_parallel(self):
        """Test parallel file batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager()

            # Create test files
            codebase_dir = Path(temp_dir) / "codebase"
            codebase_dir.mkdir()
            (codebase_dir / "file1.py").write_text("print('file1')")
            (codebase_dir / "file2.py").write_text("print('file2')")

            # Mock model
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            file_paths = ["file1.py", "file2.py"]
            gitignore_patterns = []

            with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
                mock_chunk.return_value = [
                    MagicMock(content="print('test')", start_line=1, end_line=1)
                ]
                embeddings, file_paths_list, indexed_files = manager._process_file_batch_parallel(
                    file_paths, Path(codebase_dir), gitignore_patterns, mock_model
                )

                assert len(embeddings) == 2
                assert len(file_paths_list) == 2
                assert len(indexed_files) == 2

            # Cleanup
            manager.cleanup()


class TestIndexingManagerOptimization:
    """Test cases for index optimization functionality."""

    def test_get_indexing_stats(self):
        """Test getting indexing statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager(index_dir_name=temp_dir)

            # Create mock FAISS index
            manager.faiss_index = faiss.IndexFlatL2(128)
            test_vectors = np.random.rand(10, 128).astype(np.float32)
            manager.faiss_index.add(test_vectors)

            stats = manager.get_indexing_stats()

            assert "memory_stats" in stats
            assert "index_stats" in stats
            assert stats["index_stats"]["total_vectors"] == 10
            assert stats["index_stats"]["dimension"] == 128

    def test_analyze_index_health(self):
        """Test index health analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager(index_dir_name=temp_dir)

            # Test with no index
            health = manager._analyze_index_health()
            assert health["healthy"] is False
            assert health["reason"] == "no index"

        # Test with mock index
        mock_index = MagicMock()
        mock_index.ntotal = 100
        mock_index.d = 128
        mock_index.is_trained = True
        manager.faiss_index = mock_index

        health = manager._analyze_index_health()
        assert health["healthy"] is True
        assert health["total_vectors"] == 100
        assert health["dimension"] == 128

    def test_rebuild_index_optimized(self):
        """Test optimized index rebuilding."""
        manager = IndexingManager()

        # Create test embeddings
        embeddings = np.random.rand(100, 128).astype(np.float32)

        manager._rebuild_index_optimized(embeddings)

        assert manager.faiss_index is not None
        assert manager.faiss_index.ntotal == 100
        assert manager.faiss_index.d == 128

    def test_optimize_index_comprehensive(self):
        """Test comprehensive index optimization."""
        manager = IndexingManager()

        # Create test index
        manager.faiss_index = faiss.IndexFlatL2(128)
        test_vectors = np.random.rand(50, 128).astype(np.float32)
        manager.faiss_index.add(test_vectors)

        result = manager.optimize_index_comprehensive()

        assert "success" in result
        assert "message" in result

    def test_compress_index(self):
        """Test index compression."""
        manager = IndexingManager()

        # Create test index
        manager.faiss_index = faiss.IndexFlatL2(128)
        test_vectors = np.random.rand(100, 128).astype(np.float32)
        manager.faiss_index.add(test_vectors)

        result = manager.compress_index(compression_type="scalar_quant")

        assert result["success"] is True
        assert "compression_type" in result
        assert "original_memory_mb" in result
        assert "compressed_memory_mb" in result


class TestIndexingManagerPersistence:
    """Test cases for index persistence functionality."""

    def test_save_index(self):
        """Test saving index to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager(index_dir_name=temp_dir)

            # Set up test data
            manager.indexed_codebases = {"test_codebase": {"files": ["test.py"]}}
            manager.file_paths_map = {"0": "/path/to/test.py"}
            manager.file_metadata = {"test_codebase": {"test.py": {"mtime": 123456}}}

            # Create mock FAISS index
            manager.faiss_index = faiss.IndexFlatL2(128)

            manager._save_index()

            # Check files were created
            assert manager.index_file.exists()
            assert manager.metadata_file.exists()

            # Check content
            with open(manager.index_file, 'r') as f:
                saved_data = json.load(f)
                assert "indexed_codebases" in saved_data
                assert "file_paths_map" in saved_data

    def test_initialize_index_load_existing(self):
        """Test loading existing index during initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager(index_dir_name=temp_dir)

            # Create test index file
            test_data = {
                "indexed_codebases": {"test_codebase": {"files": ["test.py"]}},
                "file_paths_map": {"0": "/path/to/test.py"}
            }
            with open(manager.index_file, 'w') as f:
                json.dump(test_data, f)

            # Create test metadata file
            test_metadata = {"test_codebase": {"test.py": {"mtime": 123456}}}
            with open(manager.metadata_file, 'w') as f:
                json.dump(test_metadata, f)

            # Re-initialize
            manager._initialize_index()

            assert "test_codebase" in manager.indexed_codebases
            assert manager.file_paths_map == {"0": "/path/to/test.py"}
            assert "test_codebase" in manager.file_metadata


class TestIndexingManagerErrorHandling:
    """Test cases for error handling in IndexingManager."""

    def test_index_codebase_invalid_path(self):
        """Test indexing with invalid path."""
        manager = IndexingManager()

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

        with pytest.raises(ValueError, match="Path is not a directory"):
            manager.index_codebase("/nonexistent/path", mock_model)

    def test_load_faiss_index_nonexistent(self):
        """Test loading non-existent FAISS index."""
        manager = IndexingManager()

        with pytest.raises(FileNotFoundError):
            manager.load_faiss_index("/nonexistent/path/index.faiss")

    def test_process_file_batch_exception_handling(self):
        """Test exception handling in file batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IndexingManager()

            codebase_dir = Path(temp_dir) / "codebase"
            codebase_dir.mkdir()

            mock_model = MagicMock()
            mock_model.encode.side_effect = Exception("Encoding failed")

            file_paths = ["nonexistent.py"]
            gitignore_patterns = []

            embeddings, file_paths_list, indexed_files = manager._process_file_batch(
                file_paths, Path(codebase_dir), gitignore_patterns, mock_model
            )

            # Should handle exception gracefully
            assert len(embeddings) == 0
            assert len(file_paths_list) == 0
            assert len(indexed_files) == 0


if __name__ == "__main__":
    pytest.main([__file__])
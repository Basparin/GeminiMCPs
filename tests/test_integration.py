"""
Integration Tests for End-to-End Workflows.

This module contains integration tests that verify the interaction between
different components of the CodeSage MCP system, including indexing,
searching, caching, and memory management working together.
"""

import pytest
import time
import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import tempfile
import os
import shutil
import numpy as np
from sentence_transformers import SentenceTransformer

from codesage_mcp.indexing import IndexingManager
from codesage_mcp.searching import SearchingManager
from codesage_mcp.memory_manager import MemoryManager
from codesage_mcp.cache import IntelligentCache
from codesage_mcp.chunking import DocumentChunker


class TestIndexingSearchingIntegration:
    """Integration tests for indexing and searching workflows."""

    def test_full_indexing_and_search_workflow(self):
        """Test complete workflow from indexing to searching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test codebase
            codebase_dir = Path(temp_dir) / "test_project"
            codebase_dir.mkdir()

            # Create test files
            (codebase_dir / "main.py").write_text("""
import os
from utils import helper

def main():
    print("Hello, World!")
    helper()

if __name__ == "__main__":
    main()
""")

            (codebase_dir / "utils.py").write_text("""
def helper():
    print("Helper function")
    return "helper result"

class Utility:
    def process(self):
        return "processed"
""")

            (codebase_dir / "README.md").write_text("# Test Project\n\nThis is a test project.")

            # Initialize components
            indexing_manager = IndexingManager(index_dir_name=str(codebase_dir / ".codesage"))
            searching_manager = SearchingManager(indexing_manager)

            # Mock sentence transformer for consistent results
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 128

            # Mock embeddings for consistent search results
            main_embedding = np.random.rand(128).astype(np.float32)
            utils_embedding = np.random.rand(128).astype(np.float32)
            readme_embedding = np.random.rand(128).astype(np.float32)

            def mock_encode_side_effect(text):
                if "main.py" in text or "Hello, World!" in text:
                    return main_embedding
                elif "utils.py" in text or "helper" in text:
                    return utils_embedding
                else:
                    return readme_embedding

            mock_model.encode.side_effect = mock_encode_side_effect

            # Step 1: Index the codebase
            with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
                mock_chunk.return_value = [
                    MagicMock(content="test content", start_line=1, end_line=1)
                ]
                indexed_files = indexing_manager.index_codebase(str(codebase_dir), mock_model)

                assert len(indexed_files) >= 2  # Should index Python files
                assert "main.py" in indexed_files
                assert "utils.py" in indexed_files

            # Step 2: Perform semantic search
            search_results = searching_manager.semantic_search_codebase(
                "function helper", mock_model, top_k=5
            )

            assert len(search_results) > 0
            assert all("file_path" in result and "score" in result for result in search_results)

            # Step 3: Perform text search
            text_results = searching_manager.search_codebase(
                str(codebase_dir), "def helper"
            )

            assert len(text_results) > 0
            assert all("file_path" in result and "line_number" in result for result in text_results)

            # Step 4: Test duplicate code detection
            duplicate_results = searching_manager.find_duplicate_code(
                str(codebase_dir), mock_model, min_similarity=0.1
            )

            # Should handle gracefully even with small test codebase
            assert isinstance(duplicate_results, list)

    def test_incremental_indexing_workflow(self):
        """Test incremental indexing workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test codebase
            codebase_dir = Path(temp_dir) / "incremental_test"
            codebase_dir.mkdir()

            # Create initial files
            (codebase_dir / "stable.py").write_text("print('stable code')")
            (codebase_dir / "changing.py").write_text("print('version 1')")

            # Initialize components
            indexing_manager = IndexingManager(index_dir_name=str(codebase_dir / ".codesage"))

            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 128
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            # Initial indexing
            with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
                mock_chunk.return_value = [
                    MagicMock(content="test content", start_line=1, end_line=1)
                ]
                initial_files = indexing_manager.index_codebase(str(codebase_dir), mock_model)

                assert len(initial_files) == 2

            # Modify a file
            time.sleep(0.01)  # Ensure different modification time
            (codebase_dir / "changing.py").write_text("print('version 2')")

            # Incremental indexing
            incremental_files, was_incremental = indexing_manager.index_codebase_incremental(
                str(codebase_dir), mock_model
            )

            assert was_incremental is True
            assert "changing.py" in incremental_files
            assert "stable.py" in incremental_files  # Should still be included

    def test_cache_integration_with_indexing(self):
        """Test that caching works properly with indexing operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test codebase
            codebase_dir = Path(temp_dir) / "cache_test"
            codebase_dir.mkdir()
            (codebase_dir / "test.py").write_text("print('test code')")

            # Initialize components
            indexing_manager = IndexingManager(index_dir_name=str(codebase_dir / ".codesage"))

            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 128
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            # Index with caching enabled
            with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
                mock_chunk.return_value = [
                    MagicMock(content="print('test code')", start_line=1, end_line=1)
                ]
                indexed_files = indexing_manager.index_codebase(str(codebase_dir), mock_model)

                # Check that cache was used if available
                if indexing_manager.cache:
                    # Try to get cached embedding
                    cached_embedding, cache_hit = indexing_manager.cache.get_embedding(
                        "test.py", "print('test code')"
                    )
                    # May or may not be cached depending on implementation
                    assert isinstance(cache_hit, bool)


class TestMemoryManagementIntegration:
    """Integration tests for memory management with other components."""

    def test_memory_manager_with_large_indexing(self):
        """Test memory manager behavior during large indexing operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test codebase with multiple files
            codebase_dir = Path(temp_dir) / "large_test"
            codebase_dir.mkdir()

            # Create multiple test files
            for i in range(10):
                (codebase_dir / f"module_{i}.py").write_text(f"""
def function_{i}():
    return {i}

class Class_{i}:
    def method(self):
        return "method_{i}"
""")

            # Initialize components
            memory_manager = MemoryManager()
            indexing_manager = IndexingManager(
                index_dir_name=str(codebase_dir / ".codesage")
            )
            indexing_manager.memory_manager = memory_manager

            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 128
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            # Monitor memory before indexing
            initial_memory = memory_manager.get_memory_usage_mb()

            # Index the codebase
            with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
                mock_chunk.return_value = [
                    MagicMock(content=f"test content {i}", start_line=1, end_line=1)
                    for i in range(3)  # Simulate chunking
                ]
                indexed_files = indexing_manager.index_codebase(str(codebase_dir), mock_model)

                assert len(indexed_files) == 10

            # Check memory after indexing
            final_memory = memory_manager.get_memory_usage_mb()

            # Memory should not have grown excessively
            memory_increase = final_memory - initial_memory
            assert memory_increase < 100  # Less than 100MB increase

            # Cleanup
            memory_manager.cleanup()

    def test_memory_manager_cache_integration(self):
        """Test memory manager working with cache system."""
        memory_manager = MemoryManager()

        # Test model caching
        mock_model = MagicMock()

        with patch('codesage_mcp.memory_manager.SentenceTransformer') as mock_st:
            mock_st.return_value = mock_model

            # Load model multiple times
            loaded_model1 = memory_manager.load_model("test_model")
            loaded_model2 = memory_manager.load_model("test_model")

            # Should be the same cached instance
            assert loaded_model1 is loaded_model2

            # Should only create model once
            assert mock_st.call_count == 1

        # Test memory cleanup
        memory_manager._cleanup_memory()

        # Cleanup
        memory_manager.cleanup()


class TestCacheIntegration:
    """Integration tests for cache system with other components."""

    def test_cache_warming_workflow(self):
        """Test cache warming with real codebase."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test codebase
            codebase_dir = Path(temp_dir) / "warm_test"
            codebase_dir.mkdir()

            (codebase_dir / "important.py").write_text("""
def critical_function():
    return "important"

class ImportantClass:
    def run(self):
        return "running"
""")

            (codebase_dir / "secondary.py").write_text("print('secondary')")

            # Initialize cache
            cache = IntelligentCache(cache_dir=str(codebase_dir / ".cache"))

            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            # Warm cache
            warming_stats = cache.warm_cache(str(codebase_dir), mock_model)

            assert "files_warmed" in warming_stats
            assert "embeddings_cached" in warming_stats

    def test_adaptive_cache_sizing(self):
        """Test adaptive cache sizing based on workload."""
        cache = IntelligentCache()

        # Simulate high workload
        cache.workload_stats["accesses_last_minute"] = 200

        # Trigger adaptation
        result = cache.adapt_cache_sizes()

        assert "adapted" in result

        # Simulate low memory
        with patch.object(cache, '_get_memory_usage', return_value=0.9):
            result = cache.adapt_cache_sizes()
            assert result["adapted"] is True

    def test_cache_prefetching(self):
        """Test smart prefetching based on usage patterns."""
        cache = IntelligentCache()

        # Simulate usage patterns
        cache.usage_patterns["file_coaccess"]["file1.py"]["file2.py"] = 5
        cache.usage_patterns["file_access_counts"]["file1.py"] = 10

        # Predict next files
        predictions = cache.predict_next_files("file1.py")

        assert isinstance(predictions, list)
        assert len(predictions) > 0


class TestChunkingIntegration:
    """Integration tests for document chunking with other components."""

    def test_chunking_with_indexing(self):
        """Test that chunking works properly with indexing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file with multiple functions
            test_file = Path(temp_dir) / "chunk_test.py"
            test_file.write_text("""
def function_one():
    print("Function one")
    x = 1 + 1
    return x

class MyClass:
    def method_one(self):
        print("Method one")
        return "result"

    def method_two(self):
        print("Method two")
        for i in range(10):
            print(i)
        return "done"

def function_two():
    print("Function two")
    data = [1, 2, 3, 4, 5]
    result = sum(data)
    return result
""")

            # Test chunking
            from codesage_mcp.chunking import chunk_file
            chunks = chunk_file(str(test_file))

            assert len(chunks) > 0

            # Check chunk properties
            for chunk in chunks:
                assert hasattr(chunk, 'content')
                assert hasattr(chunk, 'start_line')
                assert hasattr(chunk, 'end_line')
                assert hasattr(chunk, 'token_count')
                assert hasattr(chunk, 'chunk_type')
                assert chunk.start_line <= chunk.end_line
                assert chunk.token_count > 0

    def test_chunker_statistics(self):
        """Test chunker statistics generation."""
        chunker = DocumentChunker()

        # Create test content
        content = """
def func1():
    return 1

def func2():
    return 2

def func3():
    return 3
"""

        chunks = chunker.split_into_chunks(content)
        stats = chunker.get_chunk_statistics(chunks)

        assert "total_chunks" in stats
        assert "total_tokens" in stats
        assert "average_chunk_size" in stats
        assert "chunk_types" in stats
        assert stats["total_chunks"] == len(chunks)


class TestEndToEndPerformance:
    """End-to-end performance tests."""

    def test_indexing_performance_scaling(self):
        """Test that indexing performance scales reasonably with codebase size."""
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test codebase of varying sizes
            codebase_sizes = [5, 10, 20]

            for size in codebase_sizes:
                codebase_dir = Path(temp_dir) / f"perf_test_{size}"
                codebase_dir.mkdir()

                # Create test files
                for i in range(size):
                    (codebase_dir / f"module_{i}.py").write_text(f"""
def function_{i}():
    return {i}

# Some comments here
class Class_{i}:
    def method(self):
        return "method_{i}"
""")

                # Time the indexing process
                indexing_manager = IndexingManager(
                    index_dir_name=str(codebase_dir / ".codesage")
                )

                mock_model = MagicMock()
                mock_model.get_sentence_embedding_dimension.return_value = 128
                mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

                start_time = time.time()

                with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
                    mock_chunk.return_value = [
                        MagicMock(content=f"content {i}", start_line=1, end_line=1)
                    ]
                    indexed_files = indexing_manager.index_codebase(str(codebase_dir), mock_model)

                    indexing_time = time.time() - start_time

                    # Performance should be reasonable (less than 30 seconds for 20 files)
                    assert indexing_time < 30
                    assert len(indexed_files) == size

    def test_search_performance(self):
        """Test search performance with indexed codebase."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test codebase
            codebase_dir = Path(temp_dir) / "search_perf"
            codebase_dir.mkdir()

            # Create multiple files with searchable content
            for i in range(10):
                (codebase_dir / f"search_{i}.py").write_text(f"""
def search_function_{i}():
    # This function contains searchable content
    query = "search query {i}"
    result = f"result {i}"
    return result

class SearchClass_{i}:
    def find(self, term):
        return f"found {term} in class {i}"
""")

            # Index the codebase
            indexing_manager = IndexingManager(
                index_dir_name=str(codebase_dir / ".codesage")
            )
            searching_manager = SearchingManager(indexing_manager)

            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 128
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
                mock_chunk.return_value = [
                    MagicMock(content=f"search content {i}", start_line=1, end_line=1)
                ]
                indexing_manager.index_codebase(str(codebase_dir), mock_model)

            # Time search operations
            import time

            # Semantic search performance
            start_time = time.time()
            semantic_results = searching_manager.semantic_search_codebase(
                "search query", mock_model, top_k=5
            )
            semantic_time = time.time() - start_time

            # Text search performance
            start_time = time.time()
            text_results = searching_manager.search_codebase(
                str(codebase_dir), "def search_function"
            )
            text_time = time.time() - start_time

            # Performance should be reasonable
            assert semantic_time < 5  # Less than 5 seconds
            assert text_time < 2      # Less than 2 seconds

            assert len(semantic_results) > 0
            assert len(text_results) > 0


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""

    def test_indexing_failure_recovery(self):
        """Test system recovery when indexing fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            codebase_dir = Path(temp_dir) / "error_test"
            codebase_dir.mkdir()

            (codebase_dir / "good.py").write_text("print('good')")
            (codebase_dir / "bad.py").write_text("this has syntax errors +++")

            indexing_manager = IndexingManager(
                index_dir_name=str(codebase_dir / ".codesage")
            )

            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 128
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            # Should handle errors gracefully and continue with good files
            with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
                mock_chunk.return_value = [
                    MagicMock(content="good content", start_line=1, end_line=1)
                ]
                indexed_files = indexing_manager.index_codebase(str(codebase_dir), mock_model)

                # Should have indexed the good file
                assert len(indexed_files) >= 1

    def test_search_with_empty_index(self):
        """Test search behavior when index is empty or unavailable."""
        indexing_manager = IndexingManager()
        searching_manager = SearchingManager(indexing_manager)

        mock_model = MagicMock()

        # Search with no index
        semantic_results = searching_manager.semantic_search_codebase(
            "test query", mock_model
        )

        assert semantic_results == []

        # Text search with no indexed codebase
        text_results = searching_manager.search_codebase(
            "/nonexistent", "test pattern"
        )

        assert len(text_results) == 0


if __name__ == "__main__":
    pytest.main([__file__])
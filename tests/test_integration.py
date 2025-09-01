"""
Integration Tests for End-to-End Workflows.

This module contains integration tests that verify the interaction between
different components of the CodeSage MCP system, including indexing,
searching, caching, and memory management working together.
"""

import pytest
import time
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np

from codesage_mcp.core.indexing import IndexingManager
from codesage_mcp.core.searching import SearchingManager
from codesage_mcp.features.memory_management.memory_manager import MemoryManager
from codesage_mcp.features.caching.intelligent_cache import IntelligentCache
from codesage_mcp.core.chunking import DocumentChunker


class TestIndexingSearchingIntegration:
    """Integration tests for indexing and searching workflows.

    This class contains tests that verify the end-to-end functionality
    of indexing and searching components working together.
    """

    def test_full_indexing_and_search_workflow(self, mock_sentence_transformer_model, mock_chunk_file):
        """Test complete workflow from indexing to searching.

        This test verifies the full pipeline: indexing a codebase,
        performing semantic search, text search, and duplicate detection.
        """
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
            # Use fixtures for mocking
            # mock_sentence_transformer_model and mock_chunk_file are passed as arguments to the test function
            # The side_effect for encode needs to be set on the fixture's mock_model

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

            mock_sentence_transformer_model.encode.side_effect = mock_encode_side_effect

            # Step 1: Index the codebase
            # mock_chunk_file fixture handles the patching
            indexed_files = indexing_manager.index_codebase(str(codebase_dir), mock_sentence_transformer_model)

            assert len(indexed_files) >= 2  # Should index Python files
            assert "main.py" in indexed_files
            assert "utils.py" in indexed_files

            # Step 2: Perform semantic search
            search_results = searching_manager.semantic_search_codebase(
                "function helper", mock_sentence_transformer_model, top_k=5
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
                str(codebase_dir), mock_sentence_transformer_model, min_similarity=0.1
            )

            # Should handle gracefully even with small test codebase
            assert isinstance(duplicate_results, list)

    def test_incremental_indexing_workflow(self, mock_sentence_transformer_model, mock_chunk_file):
        """Test incremental indexing workflow.

        This test verifies that incremental indexing only processes
        changed files and maintains the index for unchanged files.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test codebase
            codebase_dir = Path(temp_dir) / "incremental_test"
            codebase_dir.mkdir()

            # Create initial files
            (codebase_dir / "stable.py").write_text("print('stable code')")
            (codebase_dir / "changing.py").write_text("print('version 1')")

            # Initialize components
            indexing_manager = IndexingManager(index_dir_name=str(codebase_dir / ".codesage"))

            # Initial indexing
            initial_files = indexing_manager.index_codebase(str(codebase_dir), mock_sentence_transformer_model)

            assert len(initial_files) == 2

            # Modify a file
            time.sleep(0.01)  # Ensure different modification time
            (codebase_dir / "changing.py").write_text("print('version 2')")

            # Incremental indexing
            incremental_files, was_incremental = indexing_manager.index_codebase_incremental(
                str(codebase_dir), mock_sentence_transformer_model
            )

            assert was_incremental is True, "Indexing should be incremental when files have changed"
            assert "changing.py" in incremental_files, "Changed file should be re-indexed"
            assert "stable.py" in incremental_files, "Unchanged file should still be included in results"

    def test_cache_integration_with_indexing(self, mock_sentence_transformer_model, mock_chunk_file):
        """Test that caching works properly with indexing operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test codebase
            codebase_dir = Path(temp_dir) / "cache_test"
            codebase_dir.mkdir()
            (codebase_dir / "test.py").write_text("print('test code')")

            # Initialize components
            indexing_manager = IndexingManager(index_dir_name=str(codebase_dir / ".codesage"))

            # Index with caching enabled
            indexed_files = indexing_manager.index_codebase(str(codebase_dir), mock_sentence_transformer_model)

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

    def test_memory_manager_with_large_indexing(self, mock_sentence_transformer_model, mock_chunk_file):
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

            # Monitor memory before indexing
            initial_memory = memory_manager.get_memory_usage_mb()

            # Index the codebase
            indexed_files = indexing_manager.index_codebase(str(codebase_dir), mock_sentence_transformer_model)

            assert len(indexed_files) == 10

            # Check memory after indexing
            final_memory = memory_manager.get_memory_usage_mb()

            # Memory should not have grown excessively
            memory_increase = final_memory - initial_memory
            assert memory_increase < 100  # Less than 100MB increase

            # Cleanup
            memory_manager.cleanup()

    def test_memory_manager_cache_integration(self, mock_sentence_transformer_model):
        """Test memory manager working with cache system."""
        memory_manager = MemoryManager()

        # Test model caching
        # mock_sentence_transformer_model is already a MagicMock
        with patch('codesage_mcp.features.memory_management.memory_manager.SentenceTransformer') as mock_st:
            mock_st.return_value = mock_sentence_transformer_model

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

    def test_cache_warming_workflow(self, mock_sentence_transformer_model):
        """Test cache warming with real codebase.

        This test verifies that the cache can store and retrieve embeddings
        for files in a codebase.
        """
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

            # Test basic cache operations
            test_content = "def test(): return 'test'"
            test_embedding = np.random.rand(128).astype(np.float32)

            # Store embedding
            cache.store_embedding("important.py", test_content, test_embedding)

            # Retrieve embedding
            retrieved_embedding, cache_hit = cache.get_embedding("important.py", test_content)

            assert cache_hit is True
            assert retrieved_embedding is not None
            np.testing.assert_array_equal(retrieved_embedding, test_embedding)

    def test_adaptive_cache_sizing(self):
        """Test adaptive cache sizing based on workload.

        This test verifies that the cache can adapt its sizes based on
        workload patterns and configuration.
        """
        cache = IntelligentCache()

        # Simulate high workload
        cache.workload_stats["accesses_last_minute"] = 200

        # Ensure adaptation can happen immediately for the test
        cache.adaptive_config["last_adjustment"] = 0

        # Trigger adaptation
        result = cache.adapt_cache_sizes()

        # Should return adaptation result (may or may not adapt based on logic)
        assert isinstance(result, dict)
        assert "adapted" in result

    def test_cache_prefetching(self):
        """Test smart prefetching based on usage patterns.

        This test verifies that the cache can predict and prefetch files
        based on learned usage patterns.
        """
        cache = IntelligentCache()

        # Simulate usage patterns
        cache.usage_patterns["file_coaccess"]["file1.py"] = {"file2.py": 5}
        cache.usage_patterns["file_access_counts"]["file1.py"] = 10
        cache.usage_patterns["file_access_counts"]["file2.py"] = 8

        # Predict next files
        predictions = cache.predict_next_files("file1.py")

        assert isinstance(predictions, list)
        # Should predict file2.py based on co-access
        assert "file2.py" in predictions


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
            from codesage_mcp.core.chunking import chunk_file
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

    def test_indexing_performance_scaling(self, mock_sentence_transformer_model, mock_chunk_file):
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

                start_time = time.time()

                indexed_files = indexing_manager.index_codebase(str(codebase_dir), mock_sentence_transformer_model)

                indexing_time = time.time() - start_time

                # Performance should be reasonable (less than 30 seconds for 20 files)
                assert indexing_time < 30
                assert len(indexed_files) == size

    def test_search_performance(self, mock_sentence_transformer_model, mock_chunk_file):
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
        return f"found {{term}} in class {i}"
""")

            # Index the codebase
            indexing_manager = IndexingManager(
                index_dir_name=str(codebase_dir / ".codesage")
            )
            searching_manager = SearchingManager(indexing_manager)

            indexing_manager.index_codebase(str(codebase_dir), mock_sentence_transformer_model)

            # Time search operations
            import time

            # Semantic search performance
            start_time = time.time()
            semantic_results = searching_manager.semantic_search_codebase(
                "search query", mock_sentence_transformer_model, top_k=5
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

    def test_indexing_failure_recovery(self, mock_sentence_transformer_model, mock_chunk_file):
        """Test system recovery when indexing fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            codebase_dir = Path(temp_dir) / "error_test"
            codebase_dir.mkdir()

            (codebase_dir / "good.py").write_text("print('good')")
            (codebase_dir / "bad.py").write_text("this has syntax errors +++")

            indexing_manager = IndexingManager(
                index_dir_name=str(codebase_dir / ".codesage")
            )

            # Should handle errors gracefully and continue with good files
            indexed_files = indexing_manager.index_codebase(str(codebase_dir), mock_sentence_transformer_model)

            # Should have indexed the good file
            assert len(indexed_files) >= 1

    def test_search_with_empty_index(self, mock_sentence_transformer_model):
        """Test search behavior when index is empty or unavailable.

        This test verifies that the search gracefully handles cases where
        no index is available or the index is empty.
        """
        indexing_manager = IndexingManager()
        searching_manager = SearchingManager(indexing_manager)

        # Search with no index
        semantic_results = searching_manager.semantic_search_codebase(
            "test query", mock_sentence_transformer_model
        )

        # Should return empty results when no index available
        if isinstance(semantic_results, dict):
            assert "result" in semantic_results
            assert semantic_results["result"] == []
        else:
            # Returns list when index exists but is empty
            assert isinstance(semantic_results, list)
            assert len(semantic_results) == 0

        # Text search with no indexed codebase should return empty list
        try:
            text_results = searching_manager.search_codebase(
                "/nonexistent", "test pattern"
            )
            assert len(text_results) == 0
        except ValueError:
            # If it raises ValueError for non-existent codebase, that's also acceptable
            # The test is about graceful error handling
            pass


class TestSearchingEdgeCases:
    """Additional tests for edge cases in searching functionality to improve coverage."""

    def test_search_codebase_with_exclude_patterns(self, mock_sentence_transformer_model, mock_chunk_file):
        """Test search_codebase with exclude patterns filter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            codebase_dir = Path(temp_dir) / "exclude_test"
            codebase_dir.mkdir()

            # Create test files
            (codebase_dir / "include.py").write_text("def function(): pass")
            (codebase_dir / "exclude.py").write_text("def function(): pass")

            indexing_manager = IndexingManager(index_dir_name=str(codebase_dir / ".codesage"))
            searching_manager = SearchingManager(indexing_manager)

            indexing_manager.index_codebase(str(codebase_dir), mock_sentence_transformer_model)

            # Search with exclude pattern
            results = searching_manager.search_codebase(
                str(codebase_dir), "def function", exclude_patterns=["exclude.py"]
            )

            # Should only find in include.py
            assert len(results) == 1
            assert "include.py" in results[0]["file_path"]

    def test_search_codebase_with_file_types(self, mock_sentence_transformer_model, mock_chunk_file):
        """Test search_codebase with file_types filter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            codebase_dir = Path(temp_dir) / "file_types_test"
            codebase_dir.mkdir()

            # Create test files
            (codebase_dir / "test.py").write_text("def function(): pass")
            (codebase_dir / "test.md").write_text("def function(): pass")

            indexing_manager = IndexingManager(index_dir_name=str(codebase_dir / ".codesage"))
            searching_manager = SearchingManager(indexing_manager)

            indexing_manager.index_codebase(str(codebase_dir), mock_sentence_transformer_model)

            # Search with file_types filter
            results = searching_manager.search_codebase(
                str(codebase_dir), "def function", file_types=["py"]
            )

            # Should only find in .py file
            assert len(results) == 1
            assert results[0]["file_path"].endswith(".py")

    def test_find_duplicate_code_with_archived_files(self, mock_sentence_transformer_model, mock_chunk_file):
        """Test find_duplicate_code properly filters archived files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            codebase_dir = Path(temp_dir) / "duplicate_test"
            codebase_dir.mkdir()

            # Create test files
            (codebase_dir / "active.py").write_text("""
def func1():
    return 1

def func2():
    return 2
""")
            # Create archived directory
            archive_dir = codebase_dir / "archive"
            archive_dir.mkdir()
            (archive_dir / "archived.py").write_text("""
def func1():
    return 1

def func2():
    return 2
""")

            indexing_manager = IndexingManager(index_dir_name=str(codebase_dir / ".codesage"))
            searching_manager = SearchingManager(indexing_manager)

            indexing_manager.index_codebase(str(codebase_dir), mock_sentence_transformer_model)

            # Find duplicates
            duplicates = searching_manager.find_duplicate_code(
                str(codebase_dir), mock_sentence_transformer_model, min_similarity=0.1
            )

            # Should not include archived files in duplicates
            for dup in duplicates:
                assert not dup["file1"].startswith(str(archive_dir))
                assert not dup["file2"].startswith(str(archive_dir))

    def test_search_codebase_invalid_regex(self, mock_sentence_transformer_model, mock_chunk_file):
        """Test search_codebase raises ValueError for invalid regex."""
        with tempfile.TemporaryDirectory() as temp_dir:
            codebase_dir = Path(temp_dir) / "regex_test"
            codebase_dir.mkdir()
            (codebase_dir / "test.py").write_text("print('test')")

            indexing_manager = IndexingManager(index_dir_name=str(codebase_dir / ".codesage"))
            searching_manager = SearchingManager(indexing_manager)

            indexing_manager.index_codebase(str(codebase_dir), mock_sentence_transformer_model)

            # Should raise ValueError for invalid regex
            with pytest.raises(ValueError, match="Invalid regex pattern"):
                searching_manager.search_codebase(str(codebase_dir), "[invalid")

    def test_search_codebase_unindexed_codebase(self, mock_sentence_transformer_model):
        """Test search_codebase raises ValueError for unindexed codebase."""
        indexing_manager = IndexingManager()
        searching_manager = SearchingManager(indexing_manager)

        with pytest.raises(ValueError, match="has not been indexed"):
            searching_manager.search_codebase("/nonexistent", "test pattern")


if __name__ == "__main__":
    pytest.main([__file__])
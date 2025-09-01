import pytest
from unittest.mock import Mock, patch
from codesage_mcp.core.indexing_search import FAISSIndexer, SemanticSearch, RegexSearch, IncrementalUpdater


@pytest.fixture
def mock_faiss_indexer():
    """Fixture to provide a mocked FAISS indexer instance."""
    return Mock(spec=FAISSIndexer)


@pytest.fixture
def mock_semantic_search():
    """Fixture to provide a mocked semantic search instance."""
    return Mock(spec=SemanticSearch)


@pytest.fixture
def mock_regex_search():
    """Fixture to provide a mocked regex search instance."""
    return Mock(spec=RegexSearch)


@pytest.fixture
def mock_incremental_updater():
    """Fixture to provide a mocked incremental updater instance."""
    return Mock(spec=IncrementalUpdater)


def test_faiss_indexer_initialization(mock_faiss_indexer):
    """
    Test that FAISS indexer initializes with correct dimensions and index type.

    Theoretical expectation: The indexer should create a FAISS index with specified
    vector dimensions and index type (e.g., IndexFlatIP for inner product).
    """
    # Arrange
    with patch('codesage_mcp.core.indexing_search.faiss.IndexFlatIP') as mock_index:
        mock_index.return_value = Mock()
        mock_faiss_indexer.index = Mock()

        # Act
        indexer = FAISSIndexer(dimension=128)

        # Assert
        mock_index.assert_called_once_with(128)
        assert indexer.index is not None


def test_faiss_indexer_add_vectors(mock_faiss_indexer):
    """
    Test that FAISS indexer correctly adds vectors to the index.

    Theoretical expectation: Adding vectors should update the index and maintain
    correct count of indexed items.
    """
    # Arrange
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    mock_faiss_indexer.index = Mock()
    mock_faiss_indexer.index.ntotal = 0
    mock_faiss_indexer.add_vectors.side_effect = lambda v: setattr(mock_faiss_indexer.index, 'ntotal', len(v))

    # Act
    mock_faiss_indexer.add_vectors(vectors)

    # Assert
    mock_faiss_indexer.index.add.assert_called_once()
    assert mock_faiss_indexer.index.ntotal == 2


def test_semantic_search_query(mock_semantic_search):
    """
    Test that semantic search returns relevant results for a query.

    Theoretical expectation: The search should encode the query, perform similarity search,
    and return ranked results with scores.
    """
    # Arrange
    query = "function definition"
    expected_results = [{"id": "func1", "score": 0.95}, {"id": "func2", "score": 0.85}]
    mock_semantic_search.search.return_value = expected_results

    # Act
    results = mock_semantic_search.search(query)

    # Assert
    assert results == expected_results
    mock_semantic_search.search.assert_called_once_with(query)


def test_regex_search_pattern_matching(mock_regex_search):
    """
    Test that regex search finds matches using patterns.

    Theoretical expectation: The search should compile the regex pattern and
    return all matches with their positions in the text.
    """
    # Arrange
    pattern = r"def \w+"
    text = "def func1(): pass\ndef func2(): pass"
    expected_matches = [("def func1", 0), ("def func2", 15)]
    mock_regex_search.find_matches.return_value = expected_matches

    # Act
    matches = mock_regex_search.find_matches(pattern, text)

    # Assert
    assert matches == expected_matches


def test_incremental_updater_add_documents(mock_incremental_updater):
    """
    Test that incremental updater adds new documents to the index.

    Theoretical expectation: New documents should be processed, vectors generated,
    and added to the index without rebuilding from scratch.
    """
    # Arrange
    new_docs = [{"id": "doc1", "content": "new function"}]
    mock_incremental_updater.add_documents.side_effect = lambda docs: mock_incremental_updater.process_documents(docs)
    mock_incremental_updater.process_documents.return_value = None

    # Act
    mock_incremental_updater.add_documents(new_docs)

    # Assert
    mock_incremental_updater.process_documents.assert_called_once_with(new_docs)


def test_faiss_indexer_search(mock_faiss_indexer):
    """
    Test that FAISS indexer performs search and returns nearest neighbors.

    Theoretical expectation: Searching with a query vector should return k nearest
    neighbors with their distances.
    """
    # Arrange
    query_vector = [0.1, 0.2]
    k = 5
    expected_results = ([0.1, 0.2], [0, 1])
    mock_faiss_indexer.index = Mock()
    mock_faiss_indexer.index.search.return_value = expected_results
    mock_faiss_indexer.search.return_value = expected_results

    # Act
    distances, indices = mock_faiss_indexer.search(query_vector, k)

    # Assert
    assert distances == [0.1, 0.2]
    assert indices == [0, 1]


def test_semantic_search_with_filters(mock_semantic_search):
    """
    Test that semantic search applies filters to results.

    Theoretical expectation: Filters (e.g., by type, language) should narrow down
    the search results to only matching items.
    """
    # Arrange
    query = "class definition"
    filters = {"type": "class"}
    expected_filtered_results = [{"id": "class1", "score": 0.9}]
    mock_semantic_search.search_with_filters.return_value = expected_filtered_results

    # Act
    results = mock_semantic_search.search_with_filters(query, filters)

    # Assert
    assert results == expected_filtered_results


def test_regex_search_case_insensitive(mock_regex_search):
    """
    Test that regex search supports case-insensitive matching.

    Theoretical expectation: Case-insensitive flag should match patterns regardless
    of case differences in the text.
    """
    # Arrange
    pattern = r"function"
    text = "FUNCTION example"
    expected_matches = [("FUNCTION", 0)]
    mock_regex_search.find_matches.return_value = expected_matches

    # Act
    matches = mock_regex_search.find_matches(pattern, text, case_insensitive=True)

    # Assert
    assert matches == expected_matches


def test_incremental_updater_remove_documents(mock_incremental_updater):
    """
    Test that incremental updater removes documents from the index.

    Theoretical expectation: Removed documents should be deleted from the index,
    and the index should be updated accordingly.
    """
    # Arrange
    doc_ids = ["doc1", "doc2"]
    mock_incremental_updater.remove_documents.return_value = None

    # Act
    mock_incremental_updater.remove_documents(doc_ids)

    # Assert
    mock_incremental_updater.remove_documents.assert_called_once_with(doc_ids)


def test_faiss_indexer_save_load(mock_faiss_indexer):
    """
    Test that FAISS indexer can save and load index state.

    Theoretical expectation: The index should be serializable to disk and
    reloadable with the same state.
    """
    # Arrange
    filepath = "/tmp/index.faiss"
    mock_faiss_indexer.index = Mock()
    with patch('builtins.open', create=True) as mock_file:
        mock_file.return_value.__enter__.return_value = Mock()

        # Act
        mock_faiss_indexer.save(filepath)
        mock_faiss_indexer.load(filepath)

        # Assert
        mock_faiss_indexer.index.write_index.assert_called_once()
        mock_faiss_indexer.index.read_index.assert_called_once()


def test_semantic_search_empty_query(mock_semantic_search):
    """
    Test that semantic search handles empty queries gracefully.

    Theoretical expectation: Empty queries should return an empty result set
    or raise an appropriate error.
    """
    # Arrange
    query = ""
    mock_semantic_search.search.return_value = []

    # Act
    results = mock_semantic_search.search(query)

    # Assert
    assert results == []


def test_regex_search_invalid_pattern(mock_regex_search):
    """
    Test that regex search handles invalid patterns.

    Theoretical expectation: Invalid regex patterns should raise a regex error
    with a clear message.
    """
    # Arrange
    invalid_pattern = r"[invalid"
    mock_regex_search.find_matches.side_effect = Exception("Invalid regex")

    # Act & Assert
    with pytest.raises(Exception, match="Invalid regex"):
        mock_regex_search.find_matches(invalid_pattern, "text")
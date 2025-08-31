import pytest
import numpy as np
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_sentence_transformer_model():
    """
    Pytest fixture that provides a mock SentenceTransformer model.
    """
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 128
    mock_model.encode.return_value = np.random.rand(128).astype(np.float32)
    return mock_model

@pytest.fixture
def mock_chunk_file():
    """
    Pytest fixture that provides a mock for codesage_mcp.indexing.chunk_file.
    """
    with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
        mock_chunk.return_value = [
            MagicMock(content="mocked content", start_line=1, end_line=1)
        ]
        yield mock_chunk

@pytest.fixture(autouse=True)
def reset_global_cache():
    """
    Fixture to reset the global cache instance before and after each test.
    This ensures test isolation for the IntelligentCache singleton.
    """
    from codesage_mcp.cache import reset_cache_instance
    reset_cache_instance() # Reset before test
    yield
    reset_cache_instance() # Reset after test
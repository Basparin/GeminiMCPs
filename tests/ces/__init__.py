"""
CES Test Suite

Test suite for the Cognitive Enhancement System components.
"""

import pytest
from pathlib import Path

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"

@pytest.fixture
def test_data_dir():
    """Fixture providing test data directory"""
    TEST_DATA_DIR.mkdir(exist_ok=True)
    return TEST_DATA_DIR

@pytest.fixture
def sample_task_description():
    """Sample task description for testing"""
    return "Implement a user authentication system with password hashing"

@pytest.fixture
def mock_memory_db(test_data_dir):
    """Mock memory database for testing"""
    return test_data_dir / "test_memory.db"
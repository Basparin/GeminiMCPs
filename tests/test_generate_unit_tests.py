import pytest
import tempfile
import os
from codesage_mcp.tools import generate_unit_tests_tool

def test_generate_unit_tests_tool():
    """Test the generate_unit_tests_tool function with a simple Python file."""
    # Create a simple test Python file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
def calculate_sum(a, b):
    \"\"\"Calculate the sum of two numbers.\"\"\"
    return a + b
""")
        temp_file_path = f.name
    
    try:
        # Test generating tests for the entire file
        result = generate_unit_tests_tool(temp_file_path)
        
        # Check the structure of the result
        assert "message" in result
        assert "file_path" in result
        assert "generated_tests" in result
        assert result["file_path"] == temp_file_path
        
        # Check that we got test templates
        assert isinstance(result["generated_tests"], list)
        assert len(result["generated_tests"]) >= 1
        
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


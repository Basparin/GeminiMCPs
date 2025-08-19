# CodeSage MCP Server Internal Documentation

## Architecture
- `main.py`: FastAPI application entry point, exposes MCP tools via HTTP.
- `tools.py`: Defines the functions that will be exposed as MCP tools.
- `config.py`: Handles environment variables for API keys and other configurations.
- `codebase_manager.py`: Manages codebase ingestion, indexing, and file operations.

## Tools Implemented

### Core Tools
- `read_code_file(file_path: str)`: Reads and returns the content of a specified code file.
- `index_codebase(path: str)`: Indexes a given codebase path for analysis. The index is persistent and respects `.gitignore`.
- `search_codebase(codebase_path: str, pattern: str, file_types: list[str], exclude_patterns: list[str])`: Searches for a pattern within indexed code files.
- `get_file_structure(codebase_path: str, file_path: str)`: Provides a high-level overview of a file's structure.
- `summarize_code_section(file_path: str, start_line: int, end_line: int, llm_model: str, function_name: str, class_name: str)`: Summarizes a specific section of code using the Groq, OpenRouter, or Google AI APIs.
- `semantic_search_codebase(codebase_path: str, query: str, top_k: int)`: Performs a semantic search within the indexed codebase to find code snippets semantically similar to the given query.
- `find_duplicate_code(codebase_path: str, min_similarity: float, min_lines: int)`: Identifies duplicate or highly similar code sections within the indexed codebase using semantic similarity analysis.

### Analysis Tools
- `list_undocumented_functions(file_path: str)`: Identifies and lists Python functions in a specified file that are missing docstrings.
- `count_lines_of_code()`: Counts lines of code (LOC) in the indexed codebase, providing a summary by file type.
- `get_dependencies_overview()`: Analyzes Python files in the indexed codebase and extracts import statements, providing a high-level overview of internal and external dependencies.
- `profile_code_performance(file_path: str, function_name: str)`: Profiles the performance of a specific function or the entire file using cProfile to measure execution time and resource usage.
- `suggest_code_improvements(file_path: str, start_line: int, end_line: int)`: Analyzes a code section and suggests improvements by consulting external LLMs. It identifies potential code quality issues and provides suggestions for improvements.
- `generate_unit_tests(file_path: str, function_name: str)`: Generates unit tests for functions in a Python file. The generated tests can be manually reviewed and added to the test suite.

### Configuration Tools
- `configure_api_key(llm_provider: str, api_key: str)`: Configures API keys for LLMs (e.g., Groq, OpenRouter, Google AI).
- `get_configuration()`: Returns the current configuration, with API keys masked for security.
- `analyze_codebase_improvements(codebase_path: str)`: Analyzes the codebase for potential improvements and suggestions.

## Tool Parameters and Usage

### find_duplicate_code
Identifies duplicate or highly similar code sections within the indexed codebase.

Parameters:
- `codebase_path` (str, required): Path to the indexed codebase.
- `min_similarity` (float, optional, default 0.8): Minimum similarity score to consider snippets as duplicates (0.0 to 1.0).
- `min_lines` (int, optional, default 10): Minimum number of lines a code section must have to be considered for duplication.

Example usage:
```json
{
  "name": "find_duplicate_code",
  "arguments": {
    "codebase_path": "/path/to/your/codebase",
    "min_similarity": 0.85,
    "min_lines": 15
  }
}
```

### suggest_code_improvements

Analyzes a code section and suggests improvements by consulting external LLMs. It identifies potential code quality issues and provides suggestions for improvements.

Parameters:
- `file_path` (str, required): Path to the file to analyze.
- `start_line` (int, optional): Starting line number of the section to analyze. If not provided, analyzes from the beginning of the file.
- `end_line` (int, optional): Ending line number of the section to analyze. If not provided, analyzes to the end of the file.

Example usage:
```json
{
  "name": "suggest_code_improvements",
  "arguments": {
    "file_path": "/path/to/your/code/file.py",
    "start_line": 10,
    "end_line": 25
  }
}
```

### generate_unit_tests

Generates unit tests for functions in a Python file. The generated tests can be manually reviewed and added to the test suite.

Parameters:
- `file_path` (str, required): Path to the Python file to analyze.
- `function_name` (str, optional): Name of a specific function to generate tests for. If not provided, generates tests for all functions in the file.

Example usage:
```json
{
  "name": "generate_unit_tests",
  "arguments": {
    "file_path": "/path/to/your/code/file.py",
    "function_name": "calculate_sum"
  }
}
```
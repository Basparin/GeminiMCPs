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

## Tool Parameters and Usage

### read_code_file
Reads and returns the content of a specified code file.

Parameters:
- `file_path` (str, required): The absolute path to the code file to read.

Example usage:
```json
{
  "name": "read_code_file",
  "arguments": {
    "file_path": "/path/to/your/code/file.py"
  }
}
```

### index_codebase
Indexes a given codebase path for analysis. The index is persistent and respects `.gitignore`.

Parameters:
- `path` (str, required): The absolute path to the codebase directory to index.

Example usage:
```json
{
  "name": "index_codebase",
  "arguments": {
    "path": "/path/to/your/codebase"
  }
}
```

### search_codebase
Searches for a pattern within indexed code files, with optional exclusion patterns.

Parameters:
- `codebase_path` (str, required): Path to the indexed codebase.
- `pattern` (str, required): Regex pattern to search for.
- `file_types` (list[str], optional): List of file extensions to include in the search (e.g., `[".py", ".js"]`). If None, all file types are included.
- `exclude_patterns` (list[str], optional): List of glob patterns to exclude from the search (e.g., `["temp/*", "*.log"]`). Files matching these patterns will be skipped.

Example usage:
```json
{
  "name": "search_codebase",
  "arguments": {
    "codebase_path": "/path/to/your/codebase",
    "pattern": "my_function",
    "file_types": [".py"],
    "exclude_patterns": ["tests/*"]
  }
}
```

### get_file_structure
Provides a high-level overview of a file's structure.

Parameters:
- `codebase_path` (str, required): Path to the indexed codebase.
- `file_path` (str, required): Relative path to the file within the codebase.

Example usage:
```json
{
  "name": "get_file_structure",
  "arguments": {
    "codebase_path": "/path/to/your/codebase",
    "file_path": "src/main.py"
  }
}
```

### summarize_code_section
Summarizes a specific section of code using the Groq, OpenRouter, or Google AI APIs.

Parameters:
- `file_path` (str, required): Path to the file containing the code to summarize.
- `start_line` (int, optional): Starting line number of the section to summarize. If not provided, summarizes the entire file.
- `end_line` (int, optional): Ending line number of the section to summarize. If not provided, summarizes to the end of the file.
- `llm_model` (str, required): The LLM model to use for summarization (e.g., `llama3-8b-8192`, `openrouter/google/gemini-pro`, `google/gemini-pro`).
- `function_name` (str, optional): Name of a specific function to summarize. If provided, `start_line` and `end_line` will be ignored.
- `class_name` (str, optional): Name of a specific class to summarize. If provided, `start_line` and `end_line` will be ignored.

Example usage:
```json
{
  "name": "summarize_code_section",
  "arguments": {
    "file_path": "/path/to/your/code/file.py",
    "function_name": "my_function",
    "llm_model": "llama3-8b-8192"
  }
}
```

### semantic_search_codebase
Performs a semantic search within the indexed codebase to find code snippets semantically similar to the given query.

Parameters:
- `codebase_path` (str, required): Path to the indexed codebase.
- `query` (str, required): Semantic query to search for.
- `top_k` (int, optional, default 5): Number of results to return.

Example usage:
```json
{
  "name": "semantic_search_codebase",
  "arguments": {
    "codebase_path": "/path/to/your/codebase",
    "query": "how to connect to a database",
    "top_k": 10
  }
}
```

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

### list_undocumented_functions
Identifies and lists Python functions in a specified file that are missing docstrings.

Parameters:
- `file_path` (str, required): Path to the Python file to analyze.

Example usage:
```json
{
  "name": "list_undocumented_functions",
  "arguments": {
    "file_path": "/path/to/your/code/file.py"
  }
}
```

### count_lines_of_code
Counts lines of code (LOC) in the indexed codebase, providing a summary by file type.

Parameters:
- `codebase_path` (str, required): Path to the indexed codebase.

Example usage:
```json
{
  "name": "count_lines_of_code",
  "arguments": {
    "codebase_path": "/path/to/your/codebase"
  }
}
```

### get_dependencies_overview
Analyzes Python files in the indexed codebase and extracts import statements, providing a high-level overview of internal and external dependencies.

Parameters:
- `codebase_path` (str, required): Path to the indexed codebase.

Example usage:
```json
{
  "name": "get_dependencies_overview",
  "arguments": {
    "codebase_path": "/path/to/your/codebase"
  }
}
```

### profile_code_performance
Profiles the performance of a specific function or the entire file using cProfile to measure execution time and resource usage.

Parameters:
- `file_path` (str, required): Path to the Python file to profile.
- `function_name` (str, optional): Name of the specific function to profile. If None, profiles the entire file.

Example usage:
```json
{
  "name": "profile_code_performance",
  "arguments": {
    "file_path": "/path/to/your/code/file.py",
    "function_name": "my_function"
  }
}
```

### analyze_codebase_improvements
Analyzes the codebase for potential improvements and suggestions.

Parameters:
- `codebase_path` (str, required): Path to the indexed codebase.

Example usage:
```json
{
  "name": "analyze_codebase_improvements",
  "arguments": {
    "codebase_path": "/path/to/your/codebase"
  }
}
```

### configure_api_key
Configures API keys for LLMs (e.g., Groq, OpenRouter, Google AI).

Parameters:
- `llm_provider` (str, required): The LLM provider (e.g., `groq`, `openrouter`, `google`).
- `api_key` (str, required): The API key for the specified LLM provider.

Example usage:
```json
{
  "name": "configure_api_key",
  "arguments": {
    "llm_provider": "groq",
    "api_key": "your_groq_api_key"
  }
}
```

### get_configuration
Returns the current configuration, with API keys masked for security.

Parameters: None

Example usage:
```json
{
  "name": "get_configuration",
  "arguments": {}
}
```

# CodeSage MCP Tools Reference

This document provides a comprehensive reference for all tools available in the CodeSage MCP server.

## Core Tools

### read_code_file
Reads and returns the content of a specified code file.

**Parameters:**
- `file_path` (str, required): Path to the file to read.

### index_codebase
Indexes a given codebase path for analysis. The index is persistent and respects `.gitignore`.

**Parameters:**
- `path` (str, required): Path to the codebase to index.

### search_codebase
Searches for a pattern within indexed code files.

**Parameters:**
- `codebase_path` (str, required): Path to the indexed codebase.
- `pattern` (str, required): Regex pattern to search for.
- `file_types` (list[str], optional): List of file extensions to include in the search.
- `exclude_patterns` (list[str], optional): List of patterns to exclude from the search.

### get_file_structure
Provides a high-level overview of a file's structure.

**Parameters:**
- `codebase_path` (str, required): Path to the indexed codebase.
- `file_path` (str, required): Path to the file within the codebase.

### summarize_code_section
Summarizes a specific section of code using the Groq, OpenRouter, or Google AI APIs.

**Parameters:**
- `file_path` (str, required): Path to the file to summarize.
- `start_line` (int, optional): Starting line number of the section to summarize.
- `end_line` (int, optional): Ending line number of the section to summarize.
- `llm_model` (str, optional): LLM model to use for summarization.
- `function_name` (str, optional): Name of the function to summarize.
- `class_name` (str, optional): Name of the class to summarize.

### semantic_search_codebase
Performs a semantic search within the indexed codebase to find code snippets semantically similar to the given query.

**Parameters:**
- `codebase_path` (str, required): Path to the indexed codebase.
- `query` (str, required): Semantic query to search for.
- `top_k` (int, optional, default 5): Number of results to return.

### find_duplicate_code
Identifies duplicate or highly similar code sections within the indexed codebase using semantic similarity analysis.

**Parameters:**
- `codebase_path` (str, required): Path to the indexed codebase.
- `min_similarity` (float, optional, default 0.8): Minimum similarity score to consider snippets as duplicates (0.0 to 1.0).
- `min_lines` (int, optional, default 10): Minimum number of lines a code section must have to be considered for duplication.

## Analysis Tools

### list_undocumented_functions
Identifies and lists Python functions in a specified file that are missing docstrings.

**Parameters:**
- `file_path` (str, required): Path to the Python file to analyze.

### count_lines_of_code
Counts lines of code (LOC) in the indexed codebase, providing a summary by file type.

**Parameters:**
None

### get_dependencies_overview
Analyzes Python files in the indexed codebase and extracts import statements, providing a high-level overview of internal and external dependencies.

**Parameters:**
None

### get_configuration
Returns the current configuration, with API keys masked for security.

**Parameters:**
None

### analyze_codebase_improvements
Analyzes the codebase for potential improvements and suggestions.

**Parameters:**
- `codebase_path` (str, required): Path to the indexed codebase.

## Configuration Tools

### configure_api_key
Configures API keys for LLMs (e.g., Groq, OpenRouter, Google AI).

**Parameters:**
- `llm_provider` (str, required): LLM provider (groq, openrouter, google).
- `api_key` (str, required): API key for the provider.
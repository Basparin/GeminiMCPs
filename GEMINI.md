# Project: CodeSage MCP Server

## Executive Summary
The CodeSage Model Context Protocol (MCP) Server is a high-performance, production-ready platform designed to revolutionize code analysis and search capabilities. It functions as an intermediary, enabling the Gemini CLI to interact with larger codebases and integrate with various Large Language Models (LLMs) for specialized tasks. It boasts exceptional performance metrics in indexing speed, search response, memory usage, and cache hit rate.

## Definitions
*   **MCP:** Model Context Protocol
*   **LLM:** Large Language Model
*   **FAISS:** Facebook AI Similarity Search
*   **ASGI:** Asynchronous Server Gateway Interface
*   **CLI:** Command Line Interface

## Project Overview
The server is a FastAPI application that exposes a JSON-RPC endpoint (`/mcp`). This endpoint handles requests for tool discovery (`initialize`, `tools/list`) and tool execution (`tools/call`). It integrates with a `CodebaseManager` for file system operations, codebase indexing, and LLM integration, enabling advanced features such as intelligent codebase indexing, semantic search, duplicate code detection, smart code summarization, memory optimization, multi-strategy caching, incremental indexing, parallel processing, index compression, adaptive cache sizing, smart prefetching, usage pattern learning, comprehensive monitoring, and enterprise security.

**Key Technologies:**
*   **Python:** The primary programming language.
*   **FastAPI:** Used for building the web API that handles JSON-RPC requests.
*   **Uvicorn:** An ASGI server that runs the FastAPI application.
*   **FAISS:** For vector similarity search.
*   **Sentence Transformers:** For semantic understanding and embeddings.
*   **psutil:** For memory management and monitoring.
*   **python-dotenv:** For managing environment variables, especially for API keys.
*   **groq:** The Python client for the Groq LLM API.
*   **openai:** The official OpenAI Python client, used to interact with the OpenRouter API.
*   **google-generativeai:** The Python client for Google AI models.
*   **pytest:** For running the test suite.
*   **Docker:** For containerization and easy deployment.

## Building and Running

### Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd GeminiMCPs
    ```
2.  **Install dependencies into the virtual environment:**
    ```bash
    python3 -m venv venv # On Windows: python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate; On macOS: source venv/bin/activate
    pip install -r requirements.txt
    ```
    For performance optimization, it's recommended to run:
    ```bash
    pip install --upgrade pip
    pip install --no-cache-dir -r requirements.txt
    ```

### Running the Server
You can run the CodeSage MCP Server either directly using `uvicorn` or via Docker Compose.

#### Running Directly (using uvicorn)
To start the CodeSage MCP Server in development mode:
```bash
uvicorn codesage_mcp.main:app --host 127.0.0.1 --port 8000 --reload
```
Alternatively, you can use the `start_server.sh` script:
```bash
./start_server.sh # On Windows, you might need to use `bash start_server.sh` if Git Bash is installed.
```

#### Running with Docker Compose (Recommended for Production)
Ensure you have Docker and Docker Compose installed. From the project root directory, run:
```bash
docker compose up -d
```
This will build the Docker image (if not already built) and start the server in a container. The server will be accessible at `http://localhost:8000`.

### Environment Variables
Before running the server, you need to create a `.env` file in the project root. You can copy the `.env.example` file to get started and then edit it to add your API keys and performance settings:

```bash
cp .env.example .env # On Windows, you can use `copy .env.example .env`
```

Example `.env` content:
```bash
# LLM API Keys (choose your preferred provider)
GROQ_API_KEY="gsk_..."
OPENROUTER_API_KEY="sk-or-..."
GOOGLE_API_KEY="AIza..."

# Performance Tuning
CODESAGE_MEMORY_LIMIT=512MB
CODESAGE_CACHE_SIZE=1GB
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_PARALLEL_WORKERS=4

# Production Settings
CODESAGE_LOG_LEVEL=INFO
CODESAGE_MONITORING_ENABLED=true
CODESAGE_METRICS_PORT=9090
```

### Configuring Gemini CLI
To enable the Gemini CLI to utilize the CodeSage MCP Server, you need to add its configuration to your Gemini CLI `settings.json` file. This file is typically located at `~/.config/gemini-cli/settings.json` on Linux/macOS or `%APPDATA%\gemini-cli\settings.json` on Windows.

Add the following entry to the `mcpServers` array in your `settings.json`:
```json
{
  "mcpServers": [
    {
      "name": "codesage",
      "httpUrl": "http://127.0.0.1:8000/mcp",
      "trust": true
    }
  ]
}
```
After adding this configuration, restart your Gemini CLI session for the changes to take effect. You should then be able to discover and use the tools exposed by the CodeSage MCP Server.

## Development Conventions

**Tooling:**
The project leverages the key technologies listed in the "Project Overview" section, along with `ruff` for linting and formatting.

**Pre-commit Hooks:**
The project uses pre-commit hooks to enforce code quality and consistency. The hooks are defined in the `.pre-commit-config.yaml` file and include checks for large files, trailing whitespace, private keys, `ruff` for linting and formatting, a custom hook to automatically generate the `docs/tools_reference.md` file, and a custom hook to run the `pytest` test suite.

To use the pre-commit hooks, you need to install `pre-commit` and then run `pre-commit install` in the project root.

**Docstring Convention:**
The project follows the Google docstring convention. More details can be found in `docs/docstring_standard.md`.

**Code Structure:**
*   **`codesage_mcp/main.py`**: Contains the main FastAPI application logic, handling JSON-RPC requests and routing them to the appropriate tool functions.
*   **`codesage_mcp/tools.py`**: Defines the individual tool functions that the MCP server exposes to the Gemini CLI.
*   **`codesage_mcp/codebase_manager.py`**: Manages file system operations, codebase indexing, and LLM integration.
*   **`codesage_mcp/config.py`**: Loads API keys for various LLMs from environment variables.
*   **`tests/`**: Contains the test suite for the project, with tests for the `CodebaseManager` and the FastAPI application.

**Current Tools Implemented:**
*   `read_code_file`: Reads and returns the content of a specified code file.
*   `index_codebase`: Indexes a given codebase path for analysis. The index is persistent and respects `.gitignore`.
*   `search_codebase`: Searches for a pattern within indexed code files.
*   `get_file_structure`: Provides a high-level overview of a file's structure.
*   `summarize_code_section`: Summarizes a specific section of code using the Groq, OpenRouter, or Google AI APIs.
*   `semantic_search_codebase`: Performs a semantic search within the indexed codebase to find code snippets semantically similar to the given query.
*   `find_duplicate_code`: Identifies duplicate or highly similar code sections within the indexed codebase using semantic similarity analysis.
*   `list_undocumented_functions`: Identifies and lists Python functions in a specified file that are missing docstrings.
*   `count_lines_of_code`: Counts lines of code (LOC) in the indexed codebase, providing a summary by file type.
*   `get_dependencies_overview`: Analyzes Python files in the indexed codebase and extracts import statements, providing a high-level overview of internal and external dependencies.
*   `configure_api_key`: Configures API keys for LLMs (e.g., Groq, OpenRouter, Google AI).
*   `analyze_codebase_improvements`: Analyzes the codebase for potential improvements and suggestions.
*   `auto_document_tool`: Automatically generates documentation for tools that lack detailed documentation.
*   `generate_boilerplate`: Generates standardized boilerplate code for new modules, tools, or tests.
*   `generate_llm_api_wrapper`: Generates Python wrapper code for interacting with various LLM APIs.
*   `generate_unit_tests`: Generates unit tests for functions in a Python file.
*   `parse_llm_response`: Parses the content of an LLM response, extracting and validating JSON data.
*   `profile_code_performance`: Profiles the performance of a specific function or the entire file.
*   `resolve_todo_fixme`: Analyzes a TODO/FIXME comment and suggests potential resolutions using LLMs.
*   `suggest_code_improvements`: Analyzes a code section and suggests improvements by consulting external LLMs.
*   `analyze_function_dependencies`: Analyzes function-level dependencies for a specific function or all functions in a file.
*   `analyze_external_library_usage`: Analyzes external library usage across files or a specific file.
*   `predict_performance_bottlenecks`: Predicts potential performance bottlenecks in code based on structural analysis.
*   `run_comprehensive_advanced_analysis`: Runs comprehensive advanced analysis combining dependency mapping and performance prediction.
*   `get_advanced_analysis_stats`: Gets statistics about the advanced analysis capabilities and current state.
*   `get_configuration`: Returns the current configuration, with API keys masked for security.
*   `get_cache_statistics`: Returns comprehensive statistics about the intelligent caching system.
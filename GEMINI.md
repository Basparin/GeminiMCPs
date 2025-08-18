# Project: CodeSage MCP Server

## Project Overview
The CodeSage Model Context Protocol (MCP) Server is designed to extend the capabilities of the Gemini CLI, particularly in the areas of code analysis and context management. It functions as an intermediary, enabling the Gemini CLI to interact with larger codebases and integrate with various Large Language Models (LLMs) for specialized tasks.

**Key Technologies:**
*   **Python:** The primary programming language.
*   **FastAPI:** Used for building the web API that handles JSON-RPC requests.
*   **Uvicorn:** An ASGI server that runs the FastAPI application.
*   **python-dotenv:** For managing environment variables, especially for API keys.
*   **groq:** The Python client for the Groq LLM API.
*   **openai:** The official OpenAI Python client, used to interact with the OpenRouter API.
*   **google-generativeai:** The Python client for Google AI models.
*   **pytest:** For running the test suite.
*   **Docker:** For containerization and easy deployment.

**Architecture:**
The server is a FastAPI application that exposes a JSON-RPC endpoint (`/mcp`). This endpoint handles requests for tool discovery (`initialize`, `tools/list`) and tool execution (`tools/call`). It integrates with a `CodebaseManager` for file system operations, codebase indexing, and LLM integration.

## Building and Running

### Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd GeminiMCPs
    ```
2.  **Install dependencies into the virtual environment:**
    ```bash
    venv/bin/pip install -r requirements.txt
    ```

### Running the Server
You can run the CodeSage MCP server either directly using `uvicorn` or via Docker Compose.

#### Running Directly (using uvicorn)
To start the CodeSage MCP server:
1.  Navigate to the project root directory (`/home/basparin/Escritorio/GeminiMCPs`).
2.  Execute the following command:
    ```bash
    uvicorn codesage_mcp.main:app --host 127.0.0.1 --port 8000
    ```
    The server will then be accessible at `http://127.0.0.1:8000`.

#### Running with Docker Compose
Ensure you have Docker and Docker Compose installed. From the project root directory, run:

```bash
docker compose up --build
```

This will build the Docker image (if not already built) and start the server in a container. The server will be accessible at `http://localhost:8000`.

### Configuring Gemini CLI
To enable the Gemini CLI to utilize the CodeSage MCP server, you need to add its configuration to your Gemini CLI `settings.json` file. This file is typically located at `~/.config/gemini-cli/settings.json` on Linux/macOS or `%APPDATA%\gemini-cli\settings.json` on Windows.

Add the following entry to the `mcpServers` array in your `settings.json`:
```json
{
  "mcpServers": [
    {
      "name": "codesage",
      "httpUrl": "http://127.0.0.1:8000",
      "trust": true
    }
  ]
}
```
After adding this configuration, restart your Gemini CLI session for the changes to take effect. You should then be able to discover and use the tools exposed by the CodeSage MCP server.

## Development Conventions

**Tooling:**
The project leverages `fastapi` for API development, `uvicorn` for serving the application, `python-dotenv` for secure management of environment variables, `groq`, `openai`, and `google-generativeai` for LLM integration, `pytest` for testing, and `ruff` for linting and formatting.

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

**Planned Features:**
The project aims to implement more advanced features, including:
*   More advanced search capabilities (e.g., semantic search).

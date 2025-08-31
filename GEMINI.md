# Project: CodeSage MCP Server
*Last Updated: 2025-08-31*

## Executive Summary
[Brief summary of the document's purpose and key takeaways.]

## Definitions
[Glossary of key terms used in this document.]

## Project Overview
The CodeSage Model Context Protocol (MCP) Server is a high-performance, production-ready platform designed to revolutionize code analysis and search capabilities. It functions as an intermediary, enabling the Gemini CLI to interact with larger codebases and integrate with various Large Language Models (LLMs) for specialized tasks.

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

**Architecture:**
The server is a FastAPI application that exposes a JSON-RPC endpoint (`/mcp`). This endpoint handles requests for tool discovery (`initialize`, `tools/list`) and tool execution (`tools/call`). It integrates with advanced features such as intelligent codebase indexing, semantic search, duplicate code detection, smart code summarization, memory optimization, multi-strategy caching, incremental indexing, parallel processing, index compression, adaptive cache sizing, smart prefetching, usage pattern learning, comprehensive monitoring, and enterprise security.

## Building and Running

### Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
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

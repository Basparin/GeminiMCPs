# Project: CodeSage MCP Server

## Overview
This project is a Model Context Protocol (MCP) server, named "CodeSage", designed to enhance the capabilities of the Gemini CLI. It allows the Gemini CLI to interact with larger codebases and leverage various Large Language Models (LLMs) for specialized tasks like code analysis and summarization.

## Features

The CodeSage MCP Server exposes a powerful set of tools to the Gemini CLI:

*   **Codebase Indexing:** Recursively scans a directory, respects `.gitignore` rules, and creates a persistent index of all relevant files. This index is used by other tools for efficient file access.
*   **File Reading:** Reads and returns the content of any file in the indexed codebase.
*   **Code Search:** Performs regex-based searches across all indexed files to quickly find code snippets, function definitions, or any other pattern.
*   **File Structure Overview:** Provides a tree-like view of the directory structure.
*   **LLM-Powered Code Summarization:** Summarizes sections of code using the Groq, OpenRouter, or Google AI APIs. This feature requires an API key for the desired service.
*   **Duplicate Code Detection:** Identifies duplicate or highly similar code sections within the indexed codebase using semantic similarity analysis.

For a complete reference of all available tools and their parameters, see the [Tools Reference](docs/tools_reference.md).

We've also added a new `get_configuration` tool that allows you to check the current configuration, with API keys masked for security.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd GeminiMCPs
    ```

2.  **Install dependencies into the virtual environment:**
    *Note: This project uses a virtual environment to manage dependencies. Make sure you have one set up.*
    ```bash
    # First, ensure you have a virtual environment (e.g., named 'venv')
    # python3 -m venv venv

    # Install dependencies into the venv
    venv/bin/pip install -r requirements.txt
    ```

## Configuration

To use the code summarization feature, you must set the appropriate environment variables. You can do this by creating a `.env` file in the project root directory:

```
# For Groq
GROQ_API_KEY="gsk_..."

# For OpenRouter
OPENROUTER_API_KEY="sk-or-...

# For Google AI
GOOGLE_API_KEY="AIza...
```

The server uses `python-dotenv` to automatically load these variables.

## Running the MCP Server

YouYou can run the CodeSage MCP server either directly using `uvicorn` or via Docker Compose.

### Running Directly (using uvicorn)

Navigate to the project root directory and execute:

```bash
uvicorn codesage_mcp.main:app --host 127.0.0.1 --port 8000
```

This will start the FastAPI server, making it accessible at `http://127.0.0.1:8000`.

### Running with Docker Compose

Ensure you have Docker and Docker Compose installed. From the project root directory, run:

```bash
docker compose up --build
```

This will build the Docker image (if not already built) and start the server in a container. The server will be accessible at `http://localhost:8000`.

## Tool Documentation

For detailed documentation on all available tools, including parameters and usage examples, see the [Tools Reference](docs/tools_reference.md).

## Configuring Gemini CLI

To enable the Gemini CLI to use the CodeSage MCP server, you need to add its configuration to your Gemini CLI `settings.json` file. This file is typically located in `~/.config/gemini-cli/settings.json` on Linux/macOS or `%APPDATA%\gemini-cli\settings.json` on Windows.

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

After adding this configuration, restart your Gemini CLI session for the changes to take effect. You should then be able to discover and use the tools exposed by the CodeSage MCP server.# Test commit for pre-commit hook verification

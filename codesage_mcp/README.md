# CodeSage MCP Server Internal Documentation

## Architecture
- `main.py`: FastAPI application entry point, exposes MCP tools via HTTP.
- `tools/`: Directory containing individual MCP tool implementations, organized by functionality.
- `config/config.py`: Handles application configuration settings, including environment variables for API keys.
- `features/codebase_manager/codebase_manager.py`: Manages codebase ingestion, indexing, and file operations.

For a comprehensive list of available tools and their usage, please refer to the [CodeSage MCP Tools Reference](docs/tools_reference.md).

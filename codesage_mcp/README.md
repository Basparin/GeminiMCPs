# CodeSage MCP Server Internal Documentation

## Architecture
- `main.py`: FastAPI application entry point, exposes MCP tools via HTTP.
- `tools.py`: Defines the functions that will be exposed as MCP tools.
- `config.py`: Handles environment variables for API keys and other configurations.
- `codebase_manager.py`: Manages codebase ingestion, indexing, and file operations.

## Tools Implemented
- `read_code_file(file_path: str)`: Reads and returns the content of a specified code file.

## Future Development
- Implement remaining tools.
- Integrate with LLM APIs.
- Develop codebase indexing and context management logic.
- Design and implement "Thinking Mode" functionality.
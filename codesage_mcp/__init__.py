"""CodeSage MCP Server Package.

This package contains all the modules and components of the CodeSage MCP Server,
designed to enhance the capabilities of the Gemini CLI by allowing it to interact
with larger codebases and leverage various Large Language Models (LLMs) for
specialized tasks like code analysis and summarization.

Submodules:
    codebase_manager: Main coordinator for managing codebases and delegating tasks.
    indexing: Handles codebase indexing logic.
    searching: Handles code search and similarity analysis.
    llm_analysis: Handles LLM-based code analysis and generation.
    tools: Defines the available tools for the MCP server.
    main: FastAPI application and JSON-RPC request handler.
    config: Configuration management for API keys and settings.
"""

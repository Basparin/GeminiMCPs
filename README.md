# Project: CodeSage MCP

## Overview
This project aims to develop a Model Context Protocol (MCP) server, named "CodeSage", to enhance the capabilities of the Gemini CLI, particularly in code analysis and context management. It will allow the Gemini CLI to interact with larger codebases and leverage various Large Language Models (LLMs) for specialized tasks.

## Features (Planned)
- Codebase Ingestion & Indexing
- Context Window Extension
- Multi-LLM Integration (Groq, Openrouter, Google)
- Code Analysis Tools (read_code_file, search_codebase, summarize_code_section, get_file_structure)
- "Thinking Mode" / Iterative Analysis

## Setup
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the MCP server (details to follow).
4. Configure Gemini CLI to use this MCP server (details to follow).

## Development Status
Initial setup complete. Core structure and basic `read_code_file` tool are being implemented.
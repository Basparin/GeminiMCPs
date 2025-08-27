"""Tools Package for CodeSage MCP Server.

This package organizes all the available tools for the CodeSage MCP Server
into functional categories to improve maintainability and scalability.

The tools are divided into:
    - codebase_analysis: Tools for basic codebase operations
    - llm_analysis: Tools that leverage LLMs for code analysis and summarization
    - code_generation: Tools for generating code, tests, and documentation
    - configuration: Tools for managing configuration and API keys
"""

# Import all tools from their respective modules
from .codebase_analysis import *
from .llm_analysis import *
from .code_generation import *
from .configuration import *

# Explicitly import analyze_codebase_improvements_tool to ensure it's available
from .codebase_analysis import analyze_codebase_improvements_tool

# Explicitly import cache statistics tool
from .configuration import get_cache_statistics_tool
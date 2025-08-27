"""LLM Analysis Tools Module for CodeSage MCP Server.

This module provides tools that leverage Large Language Models (LLMs) for code analysis,
summarization, and other intelligent code processing tasks.

Tools included:
    - summarize_code_section_tool: Summarizes a specific section of code using a chosen LLM.
    - semantic_search_codebase_tool: Performs a semantic search within the indexed codebase.
    - profile_code_performance_tool: Profiles the performance of a specific function or the entire file.
    - suggest_code_improvements_tool: Analyzes a code section and suggests improvements by consulting external LLMs.
"""

import ast
from codesage_mcp.codebase_manager import codebase_manager
from codesage_mcp.utils import tool_error_handler


@tool_error_handler
def summarize_code_section_tool(
    file_path: str,
    start_line: int = None,
    end_line: int = None,
    llm_model: str = None,
    function_name: str = None,
    class_name: str = None,
) -> dict:
    """Summarizes a specific section of code using a chosen LLM."""
    if function_name or class_name:
        # Find start and end lines for function/class using AST
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)

        found_node = None
        for node in ast.walk(tree):
            if (
                function_name
                and isinstance(node, ast.FunctionDef)
                and node.name == function_name
            ):
                found_node = node
                break
            if (
                class_name
                and isinstance(node, ast.ClassDef)
                and node.name == class_name
            ):
                found_node = node
                break

        if not found_node:
            raise ValueError(
                f"Function/Class {function_name or class_name} not found "
                f"in {file_path}."
            )

        start_line = found_node.lineno
        end_line = found_node.end_lineno
    elif start_line is None and end_line is None:  # Summarize entire file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        start_line = 1
        end_line = len(content.splitlines())

    if llm_model is None:
        llm_model = "llama3-8b-8192"  # Default model

    # Use the codebase manager's LLM analysis manager
    summary = codebase_manager.llm_analysis_manager.summarize_code_section(
        file_path, start_line, end_line, llm_model
    )
    return {"message": "Code section summarized successfully.", "summary": summary}


@tool_error_handler
def semantic_search_codebase_tool(
    codebase_path: str, query: str, top_k: int = 5
) -> dict:
    """Performs a semantic search within the indexed codebase to find code snippets
    semantically similar to the given query.
    """
    search_results = codebase_manager.searching_manager.semantic_search_codebase(
        query, codebase_manager.sentence_transformer_model, top_k
    )
    if search_results:
        return {
            "message": (
                f"Found {len(search_results)} semantically similar code snippets "
                f"for query '{query}'."
            ),
            "results": search_results,
        }
    else:
        return {
            "message": (
                f"No semantically similar code snippets found for query '{query}'."
            ),
            "results": [],
        }


@tool_error_handler
def profile_code_performance_tool(
    file_path: str, function_name: str = None, llm_model: str = None
) -> dict:
    """Profiles the performance of a specific function or the entire file."""
    # Use the codebase manager's LLM analysis manager
    profiling_results = codebase_manager.llm_analysis_manager.profile_code_performance(
        file_path, function_name
    )
    return profiling_results


@tool_error_handler
def suggest_code_improvements_tool(
    file_path: str,
    start_line: int = None,
    end_line: int = None,
    llm_model: str = None,
) -> dict:
    """Analyzes a code section and suggests improvements by consulting external LLMs."""
    # Use the codebase manager's LLM analysis manager
    analysis_results = codebase_manager.llm_analysis_manager.suggest_code_improvements(
        file_path, start_line, end_line
    )
    return analysis_results

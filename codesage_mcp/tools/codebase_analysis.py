"""Codebase Analysis Tools Module for CodeSage MCP Server.

This module provides tools for basic codebase operations like reading files, searching,
indexing, and analyzing codebase structure.

Tools included:
    - read_code_file_tool: Reads and returns the content of a specified code file.
    - search_codebase_tool: Searches for a pattern within indexed code files.
    - get_file_structure_tool: Provides a high-level overview of a file's structure.
    - index_codebase_tool: Indexes a given codebase path for analysis.
    - list_undocumented_functions_tool: Identifies and lists Python functions in a specified file that are missing docstrings.
    - count_lines_of_code_tool: Counts lines of code (LOC) in the indexed codebase.
    - get_dependencies_overview_tool: Analyzes Python files in the indexed codebase and extracts import statements.
    - find_duplicate_code_tool: Finds duplicate code sections within the indexed codebase.
    - analyze_codebase_improvements_tool: Analyzes the codebase for potential improvements and suggestions.
"""

import ast
import os
from collections import defaultdict
from codesage_mcp.codebase_manager import codebase_manager
from codesage_mcp.utils import _count_todo_fixme_comments, create_error_response, tool_error_handler, safe_read_file


def read_code_file_tool(file_path: str) -> dict:
    """Reads and returns the content of a specified code file."""
    content = codebase_manager.read_code_file(file_path)
    return {"content": [{"type": "text", "text": content}]}


@tool_error_handler
def search_codebase_tool(
    codebase_path: str,
    pattern: str,
    file_types: list[str] = None,
    exclude_patterns: list[str] = None,
) -> dict:
    """Searches for a pattern within indexed code files, with optional exclusion patterns.

    Args:
        codebase_path (str): Path to the indexed codebase.
        pattern (str): Regex pattern to search for.
        file_types (list[str], optional): List of file extensions to include in the search.
            If None, all file types are included.
        exclude_patterns (list[str], optional): List of patterns to exclude from the search.
            Files matching these patterns will be skipped.

    Returns:
        dict: Search results with matches and metadata, or an error message.
    """
    search_results = codebase_manager.searching_manager.search_codebase(
        codebase_path, pattern, file_types, exclude_patterns
    )
    return {
        "message": f"Found {len(search_results)} matches for pattern '{pattern}'.",
        "results": search_results,
    }


def get_file_structure_tool(codebase_path: str, file_path: str) -> dict:
    """Provides a high-level overview of a file's structure within a given codebase."""
    file_structure = codebase_manager.get_file_structure(codebase_path, file_path)
    return {
        "message": f"File structure for {file_path} in {codebase_path}:",
        "structure": file_structure,
    }


def index_codebase_tool(path: str) -> dict:
    """Indexes a given codebase path for analysis."""
    indexed_files = codebase_manager.index_codebase(path)
    return {
        "message": f"Codebase at {path} indexed successfully.",
        "indexed_files_count": len(indexed_files),
        "indexed_files": indexed_files,
    }


def list_undocumented_functions_tool(file_path: str) -> dict:
    """Identifies and lists Python functions in a specified file that are missing
    docstrings.

    Args:
        file_path (str): Path to the Python file to analyze.

    Returns:
        dict: List of undocumented functions with their line numbers, or an error message.
    """
    try:
        content = safe_read_file(file_path)
        tree = ast.parse(content)

        undocumented_functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if the function has a docstring
                if not (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    undocumented_functions.append(
                        {"name": node.name, "line_number": node.lineno}
                    )

        if undocumented_functions:
            return {
                "message": (
                    f"Found {len(undocumented_functions)} undocumented functions "
                    f"in {file_path}."
                ),
                "undocumented_functions": undocumented_functions,
            }
        else:
            return {"message": f"No undocumented functions found in {file_path}."}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    except SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Syntax error in {file_path}: {e}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


def count_lines_of_code_tool() -> dict:
    """Counts lines of code (LOC) in the indexed codebase, providing a summary by
    file type.

    Returns:
        dict: Summary of LOC by file type and total LOC, or an error message.
    """
    loc_by_file_type = defaultdict(int)
    total_loc = 0

    # Assuming the codebase_manager has already indexed the current project root
    # and we can access the indexed files directly.
    # The key for the current codebase is its absolute path.
    current_codebase_path = os.path.abspath("/home/basparin/Escritorio/GeminiMCPs")

    if current_codebase_path not in codebase_manager.indexed_codebases:
        return {
            "error": {
                "code": "NOT_INDEXED",
                "message": (
                    f"Codebase at {current_codebase_path} has not been indexed. "
                    "Please index it first."
                ),
            }
        }

    indexed_files = codebase_manager.indexed_codebases[current_codebase_path]["files"]

    for relative_file_path in indexed_files:
        file_path = os.path.join(current_codebase_path, relative_file_path)
        try:
            lines = safe_read_file(file_path, as_lines=True)
            loc = len(lines)
            total_loc += loc

            file_extension = os.path.splitext(file_path)[1]
            if not file_extension:  # Handle files without extensions
                file_extension = "no_extension"
            loc_by_file_type[file_extension] += loc
        except Exception:  # Ignore files that cannot be read (e.g., binary files)
            pass

    return {
        "message": "Lines of code count completed.",
        "total_loc": total_loc,
        "loc_by_file_type": dict(loc_by_file_type),
    }


@tool_error_handler
def get_dependencies_overview_tool(codebase_path: str) -> dict:
    """Analyzes Python files in the indexed codebase and extracts import statements,
    providing a high-level overview of internal and external dependencies.

    Args:
        codebase_path (str): Path to the indexed codebase.

    Returns:
        dict: Dependency overview with statistics and lists of internal, stdlib, and
            third-party dependencies, or an error message.
    """
    return codebase_manager.get_dependencies_overview(codebase_path)


@tool_error_handler
def find_duplicate_code_tool(
    codebase_path: str,
    min_similarity: float = 0.8,
    min_lines: int = 10,
) -> dict:
    """Finds duplicate code sections within the indexed codebase."""
    duplicates = codebase_manager.searching_manager.find_duplicate_code(
        codebase_path,
        codebase_manager.sentence_transformer_model,
        min_similarity,
        min_lines,
    )
    return {
        "message": f"Found {len(duplicates)} duplicate code sections.",
        "duplicates": duplicates,
    }


@tool_error_handler
def analyze_codebase_improvements_tool(codebase_path: str) -> dict:
    """Analyzes the codebase for potential improvements and suggestions.

    This tool provides a comprehensive analysis of the codebase to identify
    areas for improvement, including TODO/FIXME comments, undocumented functions,
    large files, and other code quality metrics.

    Args:
        codebase_path (str): Path to the indexed codebase.

    Returns:
        dict: Analysis results with suggestions for improvement.
    """
    # Check if codebase is indexed
    abs_codebase_path = str(os.path.abspath(codebase_path))
    if abs_codebase_path not in codebase_manager.indexed_codebases:
        raise ValueError(
            f"Codebase at {codebase_path} has not been indexed. "
            "Please index it first using the 'index_codebase' tool."
        )

    # Get indexed files
    indexed_files = codebase_manager.indexed_codebases[abs_codebase_path]["files"]

    # Initialize analysis results
    analysis = {
        "total_files": len(indexed_files),
        "python_files": 0,
        "todo_comments": 0,
        "fixme_comments": 0,
        "undocumented_functions": 0,
        "potential_duplicates": 0,
        "large_files": [],  # Files with > 500 lines
        "high_complexity_functions": [],  # Functions with cyclomatic complexity > 10
        "suggestions": [],
    }

    # Analyze each file
    for relative_file_path in indexed_files:
        # Skip archived files
        if "archive/" in relative_file_path:
            continue

        file_path = os.path.join(codebase_path, relative_file_path)

        # Only analyze Python files for detailed metrics
        if file_path.endswith(".py"):
            analysis["python_files"] += 1

            try:
                # Count lines
                lines = safe_read_file(file_path, as_lines=True)

                large_file_info = _count_large_files(file_path, lines)
                if large_file_info:
                    analysis["large_files"].append(large_file_info)

                # Search for TODO and FIXME comments (more precise)
                # Only count actual TODO/FIXME comments, not mentions in strings or comments
                found_todo_fixme_comments = _count_todo_fixme_comments(lines)
                analysis["todo_comments"] += len(
                    [
                        c
                        for c in found_todo_fixme_comments
                        if "TODO" in c["comment"].upper()
                    ]
                )
                analysis["fixme_comments"] += len(
                    [
                        c
                        for c in found_todo_fixme_comments
                        if "FIXME" in c["comment"].upper()
                    ]
                )

                # For Python files, check for undocumented functions using AST
                undocumented_functions_count = _count_undocumented_functions(
                    file_path
                )
                analysis["undocumented_functions"] += undocumented_functions_count

                # Analyze cyclomatic complexity using radon
                high_complexity_functions = _analyze_cyclomatic_complexity(
                    file_path
                )
                analysis["high_complexity_functions"].extend(
                    high_complexity_functions
                )

            except Exception:
                # Skip files that can't be read
                continue

    # Generate suggestions based on analysis
    analysis["suggestions"] = _generate_suggestions_based_on_analysis(analysis)

    return {
        "message": "Codebase analysis completed successfully.",
        "analysis": analysis,
    }


def _count_large_files(file_path: str, lines: list[str]) -> dict:
    """Count large files (> 500 lines).

    Args:
        file_path (str): Path to the file.
        lines (list[str]): Lines of the file.

    Returns:
        dict: Dictionary with file path and line count if the file is large, None otherwise.
    """
    line_count = len(lines)
    if line_count > 500:
        return {"file": str(file_path), "lines": line_count}
    return None


def _analyze_cyclomatic_complexity(file_path: str) -> list[dict]:
    """Analyze cyclomatic complexity of functions in a Python file using radon.

    Args:
        file_path (str): Path to the Python file to analyze.

    Returns:
        list[dict]: List of functions with high cyclomatic complexity (>10).
    """
    try:
        from radon.complexity import cc_visit

        content = safe_read_file(file_path)
        complexity_results = cc_visit(content)

        # Count functions with high complexity (e.g., > 10)
        high_complexity_functions = [
            result for result in complexity_results if result.complexity > 10
        ]

        # Format the results
        formatted_results = []
        for result in high_complexity_functions:
            formatted_results.append(
                {
                    "file": str(file_path),
                    "function": result.name,
                    "complexity": result.complexity,
                }
            )

        return formatted_results

    except Exception:
        # If radon analysis fails, skip it for this file
        return []


def _count_undocumented_functions(file_path: str) -> int:
    """Count undocumented functions in a Python file using AST.

    Args:
        file_path (str): Path to the Python file to analyze.

    Returns:
        int: Number of undocumented functions.
    """
    try:
        import ast

        content = safe_read_file(file_path)
        tree = ast.parse(content)

        # Count functions with and without docstrings
        function_count = 0
        documented_function_count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
                # Check if the function has a docstring
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    documented_function_count += 1

        # Return undocumented functions count
        return function_count - documented_function_count

    except SyntaxError:
        # Skip files with syntax errors
        return 0
    except Exception:
        # Skip files that can't be read or parsed
        return 0


def _generate_suggestions_based_on_analysis(analysis: dict) -> list[str]:
    """Generate suggestions based on the analysis results.

    Args:
        analysis (dict): Analysis results dictionary.

    Returns:
        list[str]: List of suggestions.
    """
    suggestions = []

    if analysis["todo_comments"] > 0:
        suggestions.append(
            f"Address {analysis['todo_comments']} TODO comments in the codebase"
        )

    if analysis["fixme_comments"] > 0:
        suggestions.append(
            f"Fix {analysis['fixme_comments']} FIXME issues in the codebase"
        )

    if analysis["undocumented_functions"] > 0:
        suggestions.append(
            f"Document {analysis['undocumented_functions']} functions missing docstrings"
        )

    if len(analysis["large_files"]) > 0:
        suggestions.append(
            f"Refactor {len(analysis['large_files'])} large files (>500 lines)"
        )

    # Add suggestions based on cyclomatic complexity analysis
    if len(analysis["high_complexity_functions"]) > 0:
        suggestions.append(
            f"Refactor {len(analysis['high_complexity_functions'])} functions with high cyclomatic complexity (>10)"
        )

    # Add general suggestions
    suggestions.append(
        "Consider using the 'find_duplicate_code' tool to identify duplicated code sections"
    )
    suggestions.append(
        "Use the 'list_undocumented_functions' tool for detailed analysis of missing documentation"
    )

    return suggestions




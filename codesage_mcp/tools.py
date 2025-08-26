"""Tools Module for CodeSage MCP Server.

This module defines the available tools for the CodeSage MCP Server. Each tool is a function
that takes specific arguments and returns a dictionary with the results or an error message.

The tools are designed to be called via JSON-RPC requests from the Gemini CLI. They provide
a wide range of functionalities, including:

- Codebase Indexing
- File Reading
- Code Search
- File Structure Overview
- LLM-Powered Code Summarization
- Duplicate Code Detection
- Dependency Analysis
- Code Quality Analysis
- Performance Profiling
- Code Improvement Suggestions
- Unit Test Generation
- Auto Documentation
- Configuration Management

Each tool function follows a consistent pattern:
- It takes specific arguments as input.
- It performs the required action.
- It returns a dictionary with a 'message' key and other relevant data, or an 'error' key if something goes wrong.

Example:
    ```python
    result = read_code_file_tool(file_path="path/to/file.py")
    if "error" in result:
        print(f"Error: {result['error']['message']}")
    else:
        print(result["content"])
    ```
"""

import ast
import os
from collections import defaultdict
from fastapi import HTTPException  # New import
from codesage_mcp.codebase_manager import codebase_manager
from codesage_mcp.utils import _count_todo_fixme_comments  # Import the function


def read_code_file_tool(file_path: str) -> dict:
    """Reads and returns the content of a specified code file."""
    content = codebase_manager.read_code_file(file_path)
    return {"content": [{"type": "text", "text": content}]}


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
    try:
        search_results = codebase_manager.searching_manager.search_codebase(
            codebase_path, pattern, file_types, exclude_patterns
        )
        return {
            "message": f"Found {len(search_results)} matches for pattern '{pattern}'.",
            "results": search_results,
        }
    except ValueError as e:
        return {"error": {"code": "INVALID_INPUT", "message": str(e)}}
    except Exception as e:
        return {
            "error": {
                "code": "SEARCH_ERROR",
                "message": f"An unexpected error occurred during search: {e}",
            }
        }


def semantic_search_codebase_tool(
    codebase_path: str, query: str, top_k: int = 5
) -> dict:
    """Performs a semantic search within the indexed codebase to find code snippets
    semantically similar to the given query.

    Args:
        codebase_path (str): Path to the indexed codebase.
        query (str): Semantic query to search for.
        top_k (int, optional): Number of results to return. Defaults to 5.

    Returns:
        dict: Semantically similar code snippets with scores and metadata, or an error message.
    """
    try:
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
    except Exception as e:
        return {
            "error": {
                "code": "SEMANTIC_SEARCH_ERROR",
                "message": f"An unexpected error occurred during semantic search: {e}",
            }
        }


def find_duplicate_code_tool(
    codebase_path: str,
    min_similarity: float = 0.8,
    min_lines: int = 10,
) -> dict:
    """Finds duplicate code sections within the indexed codebase."""
    try:
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
    except ValueError as e:
        return {"error": {"code": "INVALID_INPUT", "message": str(e)}}
    except Exception as e:
        return {
            "error": {
                "code": "DUPLICATE_CODE_ERROR",
                "message": (
                    f"An unexpected error occurred during duplicate code detection: {e}"
                ),
            }
        }


def summarize_code_section_tool(
    file_path: str,
    start_line: int = None,
    end_line: int = None,
    llm_model: str = None,
    function_name: str = None,
    class_name: str = None,
    llm_analysis_manager=None,
) -> dict:
    """Summarizes a specific section of code using a chosen LLM."""
    try:
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
                return {
                    "error": {
                        "code": "NOT_FOUND",
                        "message": (
                            f"Function/Class {function_name or class_name} not found "
                            f"in {file_path}."
                        ),
                    }
                }

            start_line = found_node.lineno
            end_line = found_node.end_lineno
        elif start_line is None and end_line is None:  # Summarize entire file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            start_line = 1
            end_line = len(content.splitlines())

        if llm_analysis_manager is None:
            # Create a new instance of LLMAnalysisManager if none is provided
            from codesage_mcp.llm_analysis import LLMAnalysisManager
            from codesage_mcp.codebase_manager import codebase_manager

            llm_analysis_manager = LLMAnalysisManager(
                groq_client=codebase_manager.groq_client,
                openrouter_client=codebase_manager.openrouter_client,
                google_ai_client=codebase_manager.google_ai_client,
            )

        # Use the provided or newly created llm_analysis_manager
        summary = llm_analysis_manager.summarize_code_section(
            file_path, start_line, end_line, llm_model
        )
        return {"message": "Code section summarized successfully.", "summary": summary}
    except FileNotFoundError as e:
        return {"error": {"code": "FILE_NOT_FOUND", "message": str(e)}}
    except ValueError as e:
        return {"error": {"code": "INVALID_INPUT", "message": str(e)}}
    except Exception as e:
        return {
            "error": {
                "code": "SUMMARIZATION_ERROR",
                "message": f"An unexpected error occurred during summarization: {e}",
            }
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
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
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
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
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


def configure_api_key_tool(llm_provider: str, api_key: str) -> dict:
    """Configures API keys for LLMs (e.g., Groq, OpenRouter, Google AI)."""
    config_file_path = "/home/basparin/Escritorio/GeminiMCPs/codesage_mcp/config.py"

    # Map provider to the environment variable name
    env_var_map = {
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    env_var_name = env_var_map.get(llm_provider.lower())
    if not env_var_name:
        return {
            "error": {
                "code": "INVALID_PROVIDER",
                "message": (
                    f"Unsupported LLM provider: {llm_provider}. "
                    f"Supported providers are: {', '.join(env_var_map.keys())}."
                ),
            }
        }

    # Read the config file content
    with open(config_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_lines = []
    key_updated = False
    for line in lines:
        if line.strip().startswith(f"{env_var_name} ="):
            updated_lines.append(f'{env_var_name} = "{api_key}"\n')
            key_updated = True
        else:
            updated_lines.append(line)

    if not key_updated:
        # If the key was not found, append it to the end of the file
        updated_lines.append(f'{env_var_name} = "{api_key}"\n')

    # Write the updated content back to the file
    with open(config_file_path, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)

    return {
        "message": (
            f"API key for {llm_provider} updated successfully. "
            "A server restart may be required for changes to take full effect."
        )
    }


def get_dependencies_overview_tool(codebase_path: str) -> dict:
    """Analyzes Python files in the indexed codebase and extracts import statements,
    providing a high-level overview of internal and external dependencies.

    Args:
        codebase_path (str): Path to the indexed codebase.

    Returns:
        dict: Dependency overview with statistics and lists of internal, stdlib, and
            third-party dependencies, or an error message.
    """
    try:
        return codebase_manager.get_dependencies_overview(codebase_path)
    except ValueError as e:
        return {"error": {"code": "NOT_INDEXED", "message": str(e)}}
    except Exception as e:
        return {
            "error": {
                "code": "DEPENDENCY_ANALYSIS_ERROR",
                "message": f"An unexpected error occurred during dependency analysis: {e}",
            }
        }


def get_configuration_tool() -> dict:
    """Returns the current configuration, with API keys masked for security."""
    try:
        from codesage_mcp.config import GROQ_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY

        def mask_api_key(key: str) -> str:
            """Mask an API key, showing only the first and last few characters."""
            if not key:
                return "Not set"
            if len(key) <= 8:
                return "*" * len(key)
            return f"{key[:4]}...{key[-4:]}"

        return {
            "message": "Current configuration retrieved successfully.",
            "configuration": {
                "groq_api_key": mask_api_key(GROQ_API_KEY),
                "openrouter_api_key": mask_api_key(OPENROUTER_API_KEY),
                "google_api_key": mask_api_key(GOOGLE_API_KEY),
            },
        }
    except Exception as e:
        return {
            "error": {
                "code": "CONFIGURATION_ERROR",
                "message": (
                    f"An unexpected error occurred while retrieving configuration: {e}"
                ),
            }
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

        with open(file_path, "r", encoding="utf-8") as f:
            complexity_results = cc_visit(f.read())

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
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

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


def analyze_codebase_improvements_tool(codebase_path: str) -> dict:
    """
    Analyzes the codebase for potential improvements and suggestions.

    This tool provides a comprehensive analysis of the codebase to identify
    areas for improvement, including TODO/FIXME comments, undocumented functions,
    large files, and other code quality metrics.

    Args:
        codebase_path (str): Path to the indexed codebase.

    Returns:
        dict: Analysis results with suggestions for improvement.
    """
    try:
        # Import codebase manager
        from codesage_mcp.codebase_manager import codebase_manager
        import os

        # Check if codebase is indexed
        abs_codebase_path = str(os.path.abspath(codebase_path))
        if abs_codebase_path not in codebase_manager.indexed_codebases:
            return {
                "error": {
                    "code": "NOT_INDEXED",
                    "message": (
                        f"Codebase at {codebase_path} has not been indexed. "
                        "Please index it first using the 'index_codebase' tool."
                    ),
                }
            }

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
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

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

    except Exception as e:
        return {
            "error": {
                "code": "ANALYSIS_ERROR",
                "message": f"An unexpected error occurred during codebase analysis: {e}",
            }
        }


def profile_code_performance_tool(
    file_path: str, function_name: str = None, llm_analysis_manager=None
) -> dict:
    """
    Profiles the performance of a specific function or the entire file.

    This tool uses cProfile to measure the execution time and resource usage
    of Python code. It can profile either a specific function or the entire file.

    Args:
        file_path (str): Path to the Python file to profile.
        function_name (str, optional): Name of the specific function to profile.
            If None, profiles the entire file.
        llm_analysis_manager (LLMAnalysisManager, optional): An instance of LLMAnalysisManager to use.
            If None, creates a new instance of LLMAnalysisManager.

    Returns:
        dict: Profiling results including execution time, function calls, and
            performance bottlenecks.
    """
    try:
        if llm_analysis_manager is None:
            # Create a new instance of LLMAnalysisManager if none is provided
            from codesage_mcp.llm_analysis import LLMAnalysisManager
            from codesage_mcp.codebase_manager import codebase_manager

            llm_analysis_manager = LLMAnalysisManager(
                groq_client=codebase_manager.groq_client,
                openrouter_client=codebase_manager.openrouter_client,
                google_ai_client=codebase_manager.google_ai_client,
            )

        # Use the provided or newly created llm_analysis_manager
        profiling_results = llm_analysis_manager.profile_code_performance(
            file_path, function_name
        )
        return profiling_results
    except FileNotFoundError as e:
        return {"error": {"code": "FILE_NOT_FOUND", "message": str(e)}}
    except ValueError as e:
        return {"error": {"code": "FUNCTION_NOT_FOUND", "message": str(e)}}
    except Exception as e:
        return {
            "error": {
                "code": "PROFILING_ERROR",
                "message": f"An unexpected error occurred during performance profiling: {str(e)}",
            }
        }


def suggest_code_improvements_tool(
    file_path: str,
    start_line: int = None,
    end_line: int = None,
    llm_analysis_manager=None,
) -> dict:
    """
    Analyzes a code section and suggests improvements by consulting external LLMs.

    This tool extracts a code snippet from the specified file and sends it to
    external LLMs for analysis. It identifies potential code quality issues and
    provides suggestions for improvements.

    Args:
        file_path (str): Path to the file to analyze.
        start_line (int, optional): Starting line number of the section to analyze.
            If None, analyzes from the beginning of the file.
        end_line (int, optional): Ending line number of the section to analyze.
            If None, analyzes to the end of the file.
        llm_analysis_manager (LLMAnalysisManager, optional): An instance of LLMAnalysisManager to use.
            If None, uses the default codebase_manager.

    Returns:
        dict: Analysis results with suggestions for improvements, or an error message.
    """
    try:
        if llm_analysis_manager is None:
            # Use the default codebase_manager if no llm_analysis_manager is provided
            from codesage_mcp.codebase_manager import codebase_manager

            analysis_results = (
                codebase_manager.llm_analysis_manager.suggest_code_improvements(
                    file_path, start_line, end_line
                )
            )
        else:
            # Use the provided llm_analysis_manager
            analysis_results = llm_analysis_manager.suggest_code_improvements(
                file_path, start_line, end_line
            )
        return analysis_results
    except FileNotFoundError as e:
        return {"error": {"code": "FILE_NOT_FOUND", "message": str(e)}}
    except ValueError as e:
        return {"error": {"code": "INVALID_INPUT", "message": str(e)}}
    except Exception as e:
        return {
            "error": {
                "code": "ANALYSIS_ERROR",
                "message": f"An unexpected error occurred during code analysis: {str(e)}",
            }
        }


def generate_unit_tests_tool(file_path: str, function_name: str = None) -> dict:
    """
    Generates unit tests for functions in a Python file.

    This tool analyzes function signatures and return types to generate
    appropriate test cases with edge cases. The generated tests can be
    manually reviewed and added to the test suite.

    Args:
        file_path (str): Path to the Python file to analyze.
        function_name (str, optional): Name of a specific function to generate tests for.
            If None, generates tests for all functions in the file.

    Returns:
        dict: Generated test code and metadata, or an error message.
    """
    try:
        test_results = codebase_manager.llm_analysis_manager.generate_unit_tests(
            file_path, function_name
        )
        return test_results
    except FileNotFoundError as e:
        return {"error": {"code": "FILE_NOT_FOUND", "message": str(e)}}
    except ValueError as e:
        if "Function" in str(e) and "not found" in str(e):
            return {"error": {"code": "FUNCTION_NOT_FOUND", "message": str(e)}}
        elif "No functions found" in str(e):
            return {"error": {"code": "NO_FUNCTIONS_FOUND", "message": str(e)}}
        else:
            return {"error": {"code": "INVALID_INPUT", "message": str(e)}}
    except Exception as e:
        return {
            "error": {
                "code": "TEST_GENERATION_ERROR",
                "message": f"An unexpected error occurred during test generation: {str(e)}",
            }
        }


def auto_document_tool(tool_name: str = None) -> dict:
    """
    Automatically generates documentation for tools that lack detailed documentation.

    This tool analyzes tool functions in the codebase, extracts their signatures
    and docstrings, and uses LLMs to generate human-readable documentation in
    the existing format. It can document a specific tool or all tools that lack
    detailed documentation.

    Args:
        tool_name (str, optional): Name of a specific tool to document.
            If None, documents all tools that lack detailed documentation.

    Returns:
        dict: Generated documentation and metadata, or an error message.
    """
    try:
        documentation_results = (
            codebase_manager.llm_analysis_manager.auto_document_tool(tool_name)
        )
        return documentation_results
    except FileNotFoundError as e:
        return {"error": {"code": "FILE_NOT_FOUND", "message": str(e)}}
    except ValueError as e:
        return {"error": {"code": "INVALID_INPUT", "message": str(e)}}
    except Exception as e:
        return {
            "error": {
                "code": "DOCUMENTATION_ERROR",
                "message": f"An unexpected error occurred during documentation generation: {str(e)}",
            }
        }


def resolve_todo_fixme_tool(file_path: str, line_number: int = None) -> dict:
    """
    Analyzes a TODO/FIXME comment and suggests potential resolutions using LLMs.

    Args:
        file_path (str): The absolute path to the file containing the TODO/FIXME comment.
        line_number (int, optional): The specific line number of the TODO/FIXME comment.
            If None, the tool will attempt to find and resolve the first TODO/FIXME comment in the file.

    Returns:
        dict: Suggested resolutions from LLMs, or an error message.
    """
    try:
        resolution_results = codebase_manager.llm_analysis_manager.resolve_todo_fixme(
            file_path, line_number
        )
        return resolution_results
    except FileNotFoundError as e:
        return {"error": {"code": "FILE_NOT_FOUND", "message": str(e)}}
    except ValueError as e:
        return {"error": {"code": "INVALID_INPUT", "message": str(e)}}
    except Exception as e:
        return {
            "error": {
                "code": "TODO_FIXME_RESOLUTION_ERROR",
                "message": f"An unexpected error occurred during TODO/FIXME resolution: {str(e)}",
            }
        }


def parse_llm_response_tool(llm_response_content: str) -> dict:
    """
    Parses the content of an LLM response, extracting and validating JSON data.

    This tool is designed to robustly handle various LLM output formats, including
    responses wrapped in markdown code blocks, and attempts to parse them as JSON.

    Args:
        llm_response_content (str): The raw content string received from an LLM.

    Returns:
        dict: A dictionary containing the parsed JSON data under the 'parsed_data' key,
              or an 'error' key with a message if parsing fails.
    """
    try:
        parsed_data = codebase_manager.llm_analysis_manager.parse_llm_response(
            llm_response_content
        )
        return {
            "message": "LLM response parsed successfully.",
            "parsed_data": parsed_data,
        }
    except ValueError as e:
        return {"error": {"code": "JSON_PARSE_ERROR", "message": str(e)}}
    except Exception as e:
        return {
            "error": {
                "code": "LLM_RESPONSE_PARSING_ERROR",
                "message": f"An unexpected error occurred during LLM response parsing: {str(e)}",
            }
        }


def generate_llm_api_wrapper_tool(
    llm_provider: str,
    model_name: str,
    api_key_env_var: str = None,
    output_file_path: str = None,
) -> dict:
    """
    Generates Python wrapper code for interacting with various LLM APIs.

    This tool creates a Python class that abstracts away the specifics of different
    LLM providers (Groq, OpenRouter, Google AI), providing a unified interface for
    making LLM calls. It handles API key loading from environment variables and
    includes basic error handling.

    Args:
        llm_provider (str): The LLM provider (e.g., 'groq', 'openrouter', 'google').
        model_name (str): The specific model name to use (e.g., 'llama3-8b-8192', 'gemini-pro').
        api_key_env_var (str, optional): The name of the environment variable that stores the API key.
            If None, a default will be used based on the provider.
        output_file_path (str, optional): The absolute path to save the generated wrapper code.
            If None, the generated code will be returned as a string.

    Returns:
        dict: A dictionary containing the generated wrapper code (as a string) or a success message
              if saved to a file, or an error message if generation fails.
    """
    try:
        generated_code = codebase_manager.llm_analysis_manager.generate_llm_api_wrapper(
            llm_provider, model_name, api_key_env_var
        )

        if output_file_path:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(generated_code)
            return {"message": f"LLM API wrapper saved to {output_file_path}"}
        else:
            return {"generated_code": generated_code}
    except ValueError as e:
        return {"error": {"code": "INVALID_INPUT", "message": str(e)}}
    except Exception as e:
        return {
            "error": {
                "code": "WRAPPER_GENERATION_ERROR",
                "message": f"An unexpected error occurred during wrapper generation: {str(e)}",
            }
        }

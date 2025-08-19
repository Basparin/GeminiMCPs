import ast
import os
from collections import defaultdict
from fastapi import HTTPException  # New import
from codesage_mcp.codebase_manager import codebase_manager


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
    """Searches for a pattern within indexed code files, with optional
    exclusion patterns."""
    try:
        search_results = codebase_manager.search_codebase(
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
    semantically similar to the given query."""
    try:
        search_results = codebase_manager.semantic_search_codebase(query, top_k)
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
        duplicates = codebase_manager.find_duplicate_code(
            codebase_path, min_similarity, min_lines
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

        summary = codebase_manager.summarize_code_section(
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
    docstrings."""
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
    file type."""
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


def get_dependencies_overview_tool() -> dict:
    """Analyzes Python files in the indexed codebase and extracts import statements,
    providing a high-level overview of internal and external dependencies."""
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

    internal_dependencies = defaultdict(set)
    external_dependencies = defaultdict(set)
    all_external_modules = set()

    for relative_file_path in indexed_files:
        if not relative_file_path.endswith(".py"):
            continue

        file_path = os.path.join(current_codebase_path, relative_file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split(".")[0]
                        if (
                            os.path.exists(
                                os.path.join(current_codebase_path, module_name)
                            )
                            or os.path.exists(
                                os.path.join(current_codebase_path, module_name + ".py")
                            )
                            or os.path.exists(
                                os.path.join(
                                    current_codebase_path, module_name, "__init__.py"
                                )
                            )
                        ):
                            internal_dependencies[relative_file_path].add(module_name)
                        else:
                            external_dependencies[relative_file_path].add(module_name)
                            all_external_modules.add(module_name)
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module.split(".")[0] if node.module else ""
                    if module_name:
                        if (
                            os.path.exists(
                                os.path.join(current_codebase_path, module_name)
                            )
                            or os.path.exists(
                                os.path.join(current_codebase_path, module_name + ".py")
                            )
                            or os.path.exists(
                                os.path.join(
                                    current_codebase_path, module_name, "__init__.py"
                                )
                            )
                        ):
                            internal_dependencies[relative_file_path].add(module_name)
                        else:
                            external_dependencies[relative_file_path].add(module_name)
                            all_external_modules.add(module_name)

        except Exception:  # Ignore files that cannot be parsed or read
            pass

    return {
        "message": "Dependency overview generated.",
        "total_python_files_analyzed": len(internal_dependencies)
        + len(external_dependencies),  # This is not accurate, need to fix
        "total_internal_dependencies": sum(
            len(deps) for deps in internal_dependencies.values()
        ),
        "total_external_dependencies": sum(
            len(deps) for deps in external_dependencies.values()
        ),
        "unique_external_modules": sorted(list(all_external_modules)),
        "internal_dependencies_by_file": {
            k: sorted(list(v)) for k, v in internal_dependencies.items()
        },
        "external_dependencies_by_file": {
            k: sorted(list(v)) for k, v in external_dependencies.items()
        },
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
                "google_api_key": mask_api_key(GOOGLE_API_KEY)
            }
        }
    except Exception as e:
        return {
            "error": {
                "code": "CONFIGURATION_ERROR",
                "message": (
                    "An unexpected error occurred while retrieving configuration: "
                    f"{e}"
                ),
            }
        }


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
        import fnmatch
        
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
            "suggestions": []
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
                    
                    line_count = len(lines)
                    if line_count > 500:
                        analysis["large_files"].append({
                            "file": str(file_path),
                            "lines": line_count
                        })
                    
                    # Search for TODO and FIXME comments (more precise)
                    # Only count actual TODO/FIXME comments, not mentions in strings or comments
                    todo_count = 0
                    fixme_count = 0
                    in_multiline_comment = False
                    multiline_comment_delimiter = None
                    
                    for line_num, line in enumerate(lines, 1):
                        stripped_line = line.strip()
                        
                        # Handle multiline comments
                        if in_multiline_comment:
                            if multiline_comment_delimiter in line:
                                in_multiline_comment = False
                            continue
                        
                        # Check for multiline comment start
                        if '"""' in line:
                            in_multiline_comment = True
                            multiline_comment_delimiter = '"""'
                            # Check if it's also closed on the same line
                            if line.count('"""') >= 2:
                                in_multiline_comment = False
                            continue
                        elif "'''" in line:
                            in_multiline_comment = True
                            multiline_comment_delimiter = "'''"
                            # Check if it's also closed on the same line
                            if line.count("'''") >= 2:
                                in_multiline_comment = False
                            continue
                        
                        # Check for single-line comments that start with # TODO or # FIXME
                        if stripped_line.startswith("# TODO"):
                            todo_count += 1
                        elif stripped_line.startswith("# FIXME"):
                            fixme_count += 1
                        
                        # Also check for TODO/FIXME in regular comments (not at the start)
                        # but only if it's clearly a comment marker
                        comment_parts = line.split("#", 1)
                        if len(comment_parts) > 1:
                            comment_content = comment_parts[1].strip()
                            if comment_content.startswith("TODO"):
                                # Make sure it's not part of a string or variable name
                                if len(comment_content) > 4 and (len(comment_content) == 4 or not comment_content[4].isalnum()):
                                    todo_count += 1
                            elif comment_content.startswith("FIXME"):
                                # Make sure it's not part of a string or variable name
                                if len(comment_content) > 5 and (len(comment_content) == 5 or not comment_content[5].isalnum()):
                                    fixme_count += 1
                    
                    analysis["todo_comments"] += todo_count
                    analysis["fixme_comments"] += fixme_count
                    
                    # For Python files, check for undocumented functions using AST
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
                                if (node.body and 
                                    isinstance(node.body[0], ast.Expr) and 
                                    isinstance(node.body[0].value, ast.Constant) and 
                                    isinstance(node.body[0].value.value, str)):
                                    documented_function_count += 1
                        
                        # Add undocumented functions to the count
                        analysis["undocumented_functions"] += (function_count - documented_function_count)
                    except SyntaxError:
                        # Skip files with syntax errors
                        pass
                    
                except Exception:
                    # Skip files that can't be read
                    continue
        
        # Generate suggestions based on analysis
        if analysis["todo_comments"] > 0:
            analysis["suggestions"].append(
                f"Address {analysis['todo_comments']} TODO comments in the codebase"
            )
        
        if analysis["fixme_comments"] > 0:
            analysis["suggestions"].append(
                f"Fix {analysis['fixme_comments']} FIXME issues in the codebase"
            )
        
        if analysis["undocumented_functions"] > 0:
            analysis["suggestions"].append(
                f"Document {analysis['undocumented_functions']} functions missing docstrings"
            )
        
        if len(analysis["large_files"]) > 0:
            analysis["suggestions"].append(
                f"Refactor {len(analysis['large_files'])} large files (>500 lines)"
            )
        
        # Add general suggestions
        analysis["suggestions"].append(
            "Consider using the 'find_duplicate_code' tool to identify duplicated code sections"
        )
        analysis["suggestions"].append(
            "Use the 'list_undocumented_functions' tool for detailed analysis of missing documentation"
        )
        
        return {
            "message": "Codebase analysis completed successfully.",
            "analysis": analysis
        }
        
    except Exception as e:
        return {
            "error": {
                "code": "ANALYSIS_ERROR",
                "message": f"An unexpected error occurred during codebase analysis: {e}",
            }
        }

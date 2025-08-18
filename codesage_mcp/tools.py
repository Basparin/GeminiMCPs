from codesage_mcp.codebase_manager import codebase_manager


def read_code_file_tool(file_path: str) -> dict:
    """Reads and returns the content of a specified code file."""
    content = codebase_manager.read_code_file(file_path)
    return {"content": [{"type": "text", "text": content}]}


def search_codebase_tool(
    codebase_path: str, pattern: str, file_types: list[str] = None
) -> dict:
    """Searches for a pattern within indexed code files."""
    try:
        search_results = codebase_manager.search_codebase(
            codebase_path, pattern, file_types
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


def summarize_code_section_tool(
    file_path: str, start_line: int, end_line: int, llm_model: str
) -> dict:
    """Summarizes a specific section of code using a chosen LLM."""
    try:
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


def get_file_structure_tool(path: str) -> dict:
    """Provides a high-level overview of a file's structure."""
    file_structure = codebase_manager.get_file_structure(path)
    return {"message": f"File structure for {path}:", "structure": file_structure}


def index_codebase_tool(path: str) -> dict:
    """Indexes a given codebase path for analysis."""
    indexed_files = codebase_manager.index_codebase(path)
    return {
        "message": f"Codebase at {path} indexed successfully.",
        "indexed_files_count": len(indexed_files),
        "indexed_files": indexed_files,
    }

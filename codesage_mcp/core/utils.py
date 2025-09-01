import os
import re
from functools import wraps


def _update_multiline_comment_state(
    line: str, in_multiline_comment: bool, delimiter: str
) -> tuple[bool, str]:
    """Updates the multiline comment state based on the current line."""
    if in_multiline_comment:
        if delimiter in line:
            in_multiline_comment = False
        return in_multiline_comment, delimiter

    if '"""' in line:
        in_multiline_comment = True
        delimiter = '"""'
        if line.count('"""') >= 2:
            in_multiline_comment = False
    elif "'''" in line:
        in_multiline_comment = True
        delimiter = "'''"
        if line.count("'''") >= 2:
            in_multiline_comment = False
    return in_multiline_comment, delimiter


def _count_todo_fixme_comments(lines: list[str]) -> list[dict]:
    """Identifies TODO and FIXME comments in a list of lines.

    Args:
        lines: List of lines in the file.

    Returns:
        A list of dictionaries, each containing 'line_number' and 'comment' for TODO/FIXME comments.
    """
    found_comments = []
    in_multiline_comment = False
    multiline_comment_delimiter = None

    for line_num, line in enumerate(lines, 1):
        in_multiline_comment, multiline_comment_delimiter = (
            _update_multiline_comment_state(
                line, in_multiline_comment, multiline_comment_delimiter
            )
        )
        if in_multiline_comment:
            continue

        # Check for single-line comments that start with # TODO or # FIXME
        todo_match = re.search(r"#\s*TODO", line, re.IGNORECASE)
        fixme_match = re.search(r"#\s*FIXME", line, re.IGNORECASE)

        if todo_match:
            found_comments.append({"line_number": line_num, "comment": line.strip()})
        if fixme_match:
            found_comments.append({"line_number": line_num, "comment": line.strip()})
    return found_comments


def create_error_response(error_code: str, message: str) -> dict:
    """Create a standardized error response dictionary with numeric error codes.

    Args:
        error_code: The error code identifier (string)
        message: The error message

    Returns:
        dict: Standardized error response with numeric code
    """
    # Map string error codes to numeric codes (JSON-RPC 2.0 specification)
    error_code_mapping = {
        "INVALID_REQUEST": -32600,
        "METHOD_NOT_FOUND": -32601,
        "INVALID_PARAMS": -32602,
        "INTERNAL_ERROR": -32603,
        "SERVER_ERROR": -32000,
        "TOOL_NOT_FOUND": -32001,
        "TOOL_EXECUTION_ERROR": -32002,
        "FILE_NOT_FOUND": -32003,
        "INVALID_INPUT": -32004,
        "PERMISSION_DENIED": -32005,
        "CACHE_ERROR": -32006,
        "PROFILING_ERROR": -32007,
        "ANALYSIS_ERROR": -32008,
        "TEST_GENERATION_ERROR": -32009,
        "DOCUMENTATION_ERROR": -32010,
        "TODO_FIXME_RESOLUTION_ERROR": -32011,
        "TOOL_ERROR": -32012,
    }

    # Convert string code to numeric code, defaulting to INTERNAL_ERROR if unknown
    numeric_code = error_code_mapping.get(error_code, -32603)

    return {"code": numeric_code, "message": message}


def create_numeric_error_response(error_code: int, message: str) -> dict:
    """Create a numeric error response dictionary for Gemini CLI compatibility.

    Args:
        error_code: The numeric error code
        message: The error message

    Returns:
        dict: Numeric error response
    """
    return {"code": error_code, "message": message}


def tool_error_handler(func):
    """Decorator for tool functions to provide standardized error handling.

    Catches common exceptions and returns standardized error responses with numeric codes.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            return {"error": create_error_response("FILE_NOT_FOUND", str(e))}
        except ValueError as e:
            return {"error": create_error_response("INVALID_INPUT", str(e))}
        except PermissionError as e:
            return {"error": create_error_response("PERMISSION_DENIED", str(e))}
        except Exception as e:
            return {"error": create_error_response("TOOL_ERROR", f"An error occurred: {str(e)}")}

    return wrapper


def safe_read_file(file_path: str, as_lines: bool = False) -> str | list[str]:
    """Safely read a file with consistent error handling.

    Args:
        file_path: Path to the file to read
        as_lines: If True, return list of lines; if False, return string content

    Returns:
        File content as string or list of lines

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines() if as_lines else f.read()

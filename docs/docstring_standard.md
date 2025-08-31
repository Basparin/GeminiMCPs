# Docstring Standard for CodeSage MCP Server

## Overview

This document proposes a standardized docstring format for the CodeSage MCP server codebase. Based on analysis of the current codebase using the `list_undocumented_functions` and `summarize_code_section` MCP tools, the codebase predominantly uses Google-style docstrings. This proposal formalizes that style as the standard to ensure consistency, readability, and maintainability across all Python modules.

## Format Structure

All docstrings must follow the Google-style format with the following structure:

```python
"""
Brief one-line description of the function/class/module.

Detailed description if needed, explaining purpose, behavior, and any
important implementation details.

Args:
    arg1 (type): Description of argument 1.
    arg2 (type, optional): Description of argument 2. Defaults to value.

Returns:
    return_type: Description of return value.

Raises:
    ExceptionType: Description of when this exception is raised.

Examples:
    >>> function_name(arg1, arg2)
    expected_output

Note:
    Any additional notes or important information.
"""
```

## Section Descriptions

### Brief Description
- One-line summary of the function/class/module's purpose
- Should be concise but descriptive
- Ends with a period

### Detailed Description
- Optional longer explanation
- Explains behavior, purpose, and implementation details
- Can span multiple paragraphs if needed

### Args
- Documents all parameters
- Format: `param_name (type[, optional]): description`
- Include default values for optional parameters
- Use `Optional[type]` for optional parameters in type hints

### Returns
- Documents return value(s)
- Format: `return_type: description`
- Use `Tuple[type1, type2]` for multiple returns
- Omit if function returns None

### Raises
- Documents exceptions that may be raised
- Format: `ExceptionType: description`
- Only include exceptions that are intentionally raised

### Examples
- Optional code examples showing usage
- Use doctest format for testable examples
- Include both simple and complex use cases

### Note
- Additional important information
- Deprecations, performance considerations, etc.

## Examples

### Function Example

```python
def list_undocumented_functions_tool(file_path: str) -> dict:
    """Identifies and lists Python functions in a specified file that are missing
    docstrings.

    This tool analyzes Python source files using AST parsing to detect functions
    that lack proper documentation. It helps maintain code quality by identifying
    areas that need documentation improvements.

    Args:
        file_path (str): Path to the Python file to analyze. Must be a valid
            Python source file with .py extension.

    Returns:
        dict: Analysis results containing:
            - message: Summary of findings
            - undocumented_functions: List of dicts with 'name' and 'line_number'

    Raises:
        HTTPException: If file is not found or contains syntax errors.

    Examples:
        >>> result = list_undocumented_functions_tool("example.py")
        >>> print(result["message"])
        Found 3 undocumented functions in example.py
    """
```

### Class Example

```python
class AdaptiveCacheManager:
    """Main adaptive cache management class.

    This class provides intelligent cache management with automatic adaptation
    based on usage patterns and performance metrics. It monitors cache
    effectiveness and adjusts sizes dynamically to optimize memory usage.

    Attributes:
        adaptation_interval_minutes (int): How often to perform adaptation cycles.
        max_adaptation_rate (float): Maximum rate of cache size changes.

    Examples:
        >>> manager = AdaptiveCacheManager(adaptation_interval_minutes=5)
        >>> manager.start_adaptive_management()
    """

    def __init__(self, adaptation_interval_minutes: int = 5, max_adaptation_rate: float = 0.2):
        """Initialize the adaptive cache manager.

        Args:
            adaptation_interval_minutes (int, optional): Minutes between
                adaptation cycles. Defaults to 5.
            max_adaptation_rate (float, optional): Maximum fraction of cache
                size that can change per cycle. Defaults to 0.2.
        """
```

### Module Example

```python
"""Codebase Analysis Tools Module for CodeSage MCP Server.

This module provides tools for basic codebase operations like reading files,
searching, indexing, and analyzing codebase structure. It serves as the
foundation for more advanced analysis tools in the CodeSage ecosystem.

Tools included:
    - read_code_file_tool: Reads and returns the content of a specified code file.
    - search_codebase_tool: Searches for patterns within indexed code files.
    - list_undocumented_functions_tool: Identifies undocumented functions.
"""
```

## Implementation Guidelines

### When to Write Docstrings

1. **All Public Functions/Methods**: Every public function, method, and class must have a docstring.
2. **Private Functions**: Private functions (_prefix) should have docstrings if they are complex or non-obvious.
3. **Modules**: All modules must have module-level docstrings.
4. **Classes**: All classes must have class docstrings.
5. **Constants**: Complex constants or enums may benefit from docstrings.

### Docstring Quality

1. **Completeness**: Include all parameters, return values, and exceptions.
2. **Accuracy**: Keep docstrings synchronized with code changes.
3. **Clarity**: Use clear, concise language.
4. **Consistency**: Follow the established format exactly.

### Tools for Enforcement

- Use `list_undocumented_functions_tool` to identify missing docstrings
- Configure linters (e.g., pydocstyle) to enforce docstring standards
- Integrate docstring checks into CI/CD pipeline
- Use `summarize_code_section_tool` to review complex functions for adequate documentation

### Migration Strategy

1. **Phase 1**: Audit current codebase using MCP tools
2. **Phase 2**: Add docstrings to undocumented functions
3. **Phase 3**: Standardize existing docstrings to Google format
4. **Phase 4**: Implement automated checks and enforcement

## Benefits

1. **Consistency**: Uniform documentation style across the codebase
2. **Maintainability**: Easier for developers to understand and modify code
3. **Tool Integration**: Better support from IDEs, documentation generators, and analysis tools
4. **Code Quality**: Improved overall code quality and readability
5. **Onboarding**: Faster ramp-up time for new developers

## References

- Google Style Guide: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
- PEP 257: https://www.python.org/dev/peps/pep-0257/
- NumPy/SciPy Docstring Standard: https://numpydoc.readthedocs.io/en/latest/format.html
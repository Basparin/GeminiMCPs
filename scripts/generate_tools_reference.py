#!/usr/bin/env python3
"""
Script to automatically generate docs/tools_reference.md from the tool definitions
in codesage_mcp/main.py.

This script parses the tool definitions and their corresponding functions to
generate a markdown document with a reference for all available tools.
"""

import sys
import inspect
import textwrap
from pathlib import Path
from codesage_mcp.main import get_all_tools_definitions_as_object, TOOL_FUNCTIONS

# Add the project root to the path so we can import from codesage_mcp
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_docstring_for_parameters(docstring):
    """Parses a docstring to extract parameter descriptions.

    Looks for an 'Args:' section and extracts parameter names and descriptions.
    """
    if not docstring:
        return {}

    param_descriptions = {}
    lines = docstring.split("\n")
    in_args_section = False
    current_param = None
    current_description = []

    for line in lines:
        stripped_line = line.strip()

        # Detect start of Args section
        if stripped_line == "Args:":
            in_args_section = True
            continue

        # If we're in the Args section
        if in_args_section:
            # End of Args section
            if stripped_line in ("Returns:", "Raises:", "") and not current_param:
                break

            # If line starts with a parameter definition like 'param_name (type): description'
            if (
                ":" in stripped_line
                and "(" in stripped_line
                and stripped_line.index("(") < stripped_line.index(":")
            ):
                # If we were accumulating a description for a previous parameter, save it
                if current_param:
                    param_descriptions[current_param] = " ".join(
                        current_description
                    ).strip()
                    current_description = []

                # Parse the new parameter
                # This is a bit fragile, but should work for common formats
                parts = stripped_line.split(":", 1)
                if len(parts) == 2:
                    param_part, desc_part = parts
                    param_part = param_part.strip()
                    desc_part = desc_part.strip()

                    # Extract parameter name (before the first space or parenthesis)
                    param_name = param_part.split()[0].split("(")[0]
                    param_descriptions[param_name] = desc_part
                    current_param = param_name
                else:
                    # If we can't parse it, treat it as part of a description
                    if current_param:
                        current_description.append(stripped_line)
            # If line starts with '- ' it's a list item, likely part of a description
            elif stripped_line.startswith("- "):
                if current_param:
                    current_description.append(stripped_line[2:])
            # If line is indented, it's likely a continuation of a description
            elif line.startswith(" ") and current_param:
                current_description.append(stripped_line)
            # If line is not indented and not a new parameter, it might be the end of Args
            # or a new section, or a malformed description line
            elif not line.startswith(" ") and stripped_line and current_param:
                # Save the previous parameter's description
                param_descriptions[current_param] = " ".join(
                    current_description
                ).strip()
                current_description = []
                current_param = None
                # Check if this line starts a new section
                if stripped_line in ("Returns:", "Raises:"):
                    break
                # Or if it's a new parameter definition
                elif ":" in stripped_line and "(" in stripped_line:
                    parts = stripped_line.split(":", 1)
                    if len(parts) == 2:
                        param_part, desc_part = parts
                        param_part = param_part.strip()
                        desc_part = desc_part.strip()
                        param_name = param_part.split()[0].split("(")[0]
                        param_descriptions[param_name] = desc_part
                        current_param = param_name

    # Don't forget the last parameter
    if current_param:
        param_descriptions[current_param] = " ".join(current_description).strip()

    return param_descriptions


def parse_docstring_for_returns(docstring):
    """Parses a docstring to extract the 'Returns:' section."""
    if not docstring:
        return ""

    lines = docstring.split("\n")
    in_returns_section = False
    return_lines = []

    for line in lines:
        stripped_line = line.strip()

        # Detect start of Returns section
        if stripped_line == "Returns:":
            in_returns_section = True
            continue

        # If we're in the Returns section
        if in_returns_section:
            # End of Returns section
            if stripped_line in ("Args:", "Raises:", ""):
                break
            return_lines.append(line)

    return textwrap.dedent("\n".join(return_lines)).strip()


def generate_tools_reference():
    """Generates the tools reference markdown content."""

    # Get all tool definitions
    tool_definitions = get_all_tools_definitions_as_object()

    # Categorize tools based on their names (this is a simple heuristic)
    # A more robust approach would be to have categories defined in main.py
    categories = {
        "Core Tools": [
            "read_code_file",
            "index_codebase",
            "search_codebase",
            "semantic_search_codebase",
            "find_duplicate_code",
            "get_file_structure",
            "summarize_code_section",
        ],
        "Analysis Tools": [
            "list_undocumented_functions",
            "count_lines_of_code",
            "get_dependencies_overview",
            "profile_code_performance",
            "analyze_codebase_improvements",
            "suggest_code_improvements",
            "generate_unit_tests",
            "auto_document_tool",
            "resolve_todo_fixme",
            "parse_llm_response",
            "generate_llm_api_wrapper",
        ],
        "Configuration Tools": ["get_configuration", "configure_api_key"],
    }

    # Verify that all tools are categorized
    all_categorized_tools = []
    for tools in categories.values():
        all_categorized_tools.extend(tools)

    all_defined_tools = list(tool_definitions.keys())
    uncategorized_tools = set(all_defined_tools) - set(all_categorized_tools)
    if uncategorized_tools:
        print(f"Warning: Found uncategorized tools: {uncategorized_tools}")
        # Add them to a default category
        categories["Other Tools"] = list(uncategorized_tools)

    # Check for tools that are categorized but not defined
    undefined_tools = set(all_categorized_tools) - set(all_defined_tools)
    if undefined_tools:
        print(
            f"Warning: Found categorized tools that are not defined: {undefined_tools}"
        )
        # Remove them from categories
        for category, tools in categories.items():
            categories[category] = [t for t in tools if t in all_defined_tools]

    # Start building the markdown content
    content = []
    content.append("# CodeSage MCP Tools Reference")
    content.append("")
    content.append(
        "This document provides a comprehensive reference for all tools available in the CodeSage MCP server."
    )
    content.append("")

    tool_examples = {
        "find_duplicate_code": """```json
{
  "name": "find_duplicate_code",
  "arguments": {
    "codebase_path": "/path/to/your/codebase",
    "min_similarity": 0.85,
    "min_lines": 15
  }
}
```""",
        "suggest_code_improvements": """```json
{
  "name": "suggest_code_improvements",
  "arguments": {
    "file_path": "/path/to/your/code/file.py",
    "start_line": 10,
    "end_line": 25
  }
}
```""",
        "generate_unit_tests": """```json
{
  "name": "generate_unit_tests",
  "arguments": {
    "file_path": "/path/to/your/code/file.py",
    "function_name": "calculate_sum"
  }
}
```""",
        "auto_document_tool": """```json
{
  "name": "auto_document_tool",
  "arguments": {
    "tool_name": "my_tool"
  }
}
```""",
        "get_file_structure": """```json
{
  "name": "get_file_structure",
  "arguments": {
    "codebase_path": "/path/to/your/codebase",
    "file_path": "src/main.py"
  }
}
```""",
        "index_codebase": """```json
{
  "name": "index_codebase",
  "arguments": {
    "path": "/path/to/your/codebase"
  }
}
```""",
        "read_code_file": """```json
{
  "name": "read_code_file",
  "arguments": {
    "file_path": "src/example.py"
  }
}
```""",
        "search_codebase": """```json
{
  "name": "search_codebase",
  "arguments": {
    "codebase_path": "/path/to/your/codebase",
    "pattern": "def\\s+\\w+",
    "file_types": ["py"],
    "exclude_patterns": ["tests/"]
  }
}
```""",
        "semantic_search_codebase": """```json
{
  "name": "semantic_search_codebase",
  "arguments": {
    "codebase_path": "/path/to/your/codebase",
    "query": "find functions related to user authentication",
    "top_k": 3
  }
}
```""",
        "summarize_code_section": """```json
{
  "name": "summarize_code_section",
  "arguments": {
    "file_path": "src/utils.py",
    "start_line": 10,
    "end_line": 20,
    "llm_model": "llama3-8b-8192" # Example LLM model
  }
}
```""",
        "analyze_codebase_improvements": """```json
{
  "name": "analyze_codebase_improvements",
  "arguments": {
    "codebase_path": "/path/to/your/codebase"
  }
}
```""",
        "count_lines_of_code": """```json
{
  "name": "count_lines_of_code",
  "arguments": {}
}
```""",
        "get_dependencies_overview": """```json
{
  "name": "get_dependencies_overview",
  "arguments": {}
}
```""",
        "list_undocumented_functions": """```json
{
  "name": "list_undocumented_functions",
  "arguments": {
    "file_path": "src/my_module.py"
  }
}
```""",
        "profile_code_performance": """```json
{
  "name": "profile_code_performance",
  "arguments": {
    "file_path": "src/performance_critical.py",
    "function_name": "heavy_computation"
  }
}
```""",
        "configure_api_key": """```json
{
  "name": "configure_api_key",
  "arguments": {
    "llm_provider": "groq",
    "api_key": "your_groq_api_key"
  }
}
```""",
        "get_configuration": """```json
{
  "name": "get_configuration",
  "arguments": {}
}
```""",
    }

    # Process each category
    for category_name, tool_names in categories.items():
        if not tool_names:
            continue

        content.append(f"## {category_name}")
        content.append("")

        # Sort tools by name for consistent ordering within the category
        sorted_tool_names = sorted(tool_names)

        for tool_name in sorted_tool_names:
            if tool_name not in tool_definitions:
                print(f"Warning: Tool '{tool_name}' is categorized but not defined.")
                continue

            definition = tool_definitions[tool_name]
            input_schema = definition.get("inputSchema", {})
            content.append(f"### {tool_name}")

            # Description
            description = definition.get("description", "No description available.")
            content.append(description)
            content.append("")

            # Get the actual function to extract more detailed information
            tool_function = TOOL_FUNCTIONS.get(tool_name)
            param_descriptions = {}
            returns_description = ""
            param_descriptions = {}
            returns_description = ""
            sig = None
            if tool_function:
                func_docstring = inspect.getdoc(tool_function)
                if func_docstring:
                    param_descriptions = parse_docstring_for_parameters(func_docstring)
                    returns_description = parse_docstring_for_returns(func_docstring)

                # Get parameter types from function signature
                sig = inspect.signature(tool_function)

            properties = input_schema.get("properties", {})
            if sig:
                for param_name, param_obj in sig.parameters.items():
                    if (
                        param_name in properties
                    ):  # Only process parameters that are in the inputSchema
                        param_type_str = (
                            str(param_obj.annotation)
                            if param_obj.annotation != inspect.Parameter.empty
                            else properties[param_name].get("type", "any")
                        )
                        # Clean up type string (e.g., <class 'str'> -> str)
                        if param_type_str.startswith(
                            "<class '"
                        ) and param_type_str.endswith(">"):
                            param_type_str = param_type_str[8:-2]
                        elif param_type_str.startswith("typing."):
                            param_type_str = param_type_str[7:]
                        properties[param_name]["type"] = param_type_str

            # Parameters
            if input_schema and "properties" in input_schema:
                content.append("**Parameters:**")
                required_params = input_schema.get("required", [])

                if properties:
                    for param_name, param_info in properties.items():
                        param_type = param_info.get("type", "any")
                        # Check if it's required
                        required_str = (
                            "required" if param_name in required_params else "optional"
                        )

                        # Try to get a description from the parsed docstring or use a default
                        description = param_descriptions.get(
                            param_name, param_info.get("description", "")
                        )
                        if not description:
                            if param_name == "codebase_path":
                                description = "Path to the indexed codebase."
                            elif param_name == "file_path":
                                description = "The absolute path to the file."
                            elif param_name == "function_name":
                                description = "The name of the function to analyze."
                            elif param_name == "llm_model":
                                description = "The LLM model to use for the operation."
                            elif param_name == "llm_provider":
                                description = "The LLM provider (e.g., Groq, OpenRouter, Google AI)."
                            elif param_name == "api_key":
                                description = (
                                    "The API key for the specified LLM provider."
                                )
                            elif param_name == "path":
                                description = "The absolute path to the directory."

                        if description:
                            content.append(
                                f"- `{param_name}` ({param_type}, {required_str}): {description}"
                            )
                        else:
                            content.append(
                                f"- `{param_name}` ({param_type}, {required_str})"
                            )
                else:
                    content.append("None")
                content.append("")

            # Add example usage
            if tool_name in tool_examples:
                content.append("**Example Usage:**")
                content.append(tool_examples[tool_name])
                content.append("")

            # Returns section
            if returns_description:
                content.append("**Returns:**")
                # Indent the returns description
                indented_returns = textwrap.indent(returns_description, "    ")
                content.append(indented_returns)
                content.append("")

            content.append("")  # Extra newline for spacing

    return "\n".join(content)


def main():
    """Main function to generate the tools reference."""
    print("Generating tools reference...")

    # Ensure the docs directory exists
    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Generate the content
    content = generate_tools_reference()

    # Write to file
    output_file = docs_dir / "tools_reference.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Tools reference generated successfully at {output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to run an audit of the codebase using the project's own tools.
"""

import sys
import json
import subprocess
from pathlib import Path

# Add the project root to the path so we can import from codesage_mcp
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_tool(tool_name, arguments=None):
    """Run a tool via the MCP server and return the result."""
    if arguments is None:
        arguments = {}

    # Construct the curl command
    curl_cmd = [
        "curl",
        "-X",
        "POST",
        "http://127.0.0.1:8000/mcp",
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
        ),
    ]

    try:
        result = subprocess.run(
            curl_cmd, capture_output=True, text=True, cwd=project_root
        )
        if result.returncode == 0:
            response = json.loads(result.stdout)
            if "result" in response:
                return response["result"]
            elif "error" in response:
                print(f"Error from {tool_name}: {response['error']}")
                return None
        else:
            print(f"Error running {tool_name}: {result.stderr}")
            return None
    except Exception as e:
        print(f"Exception running {tool_name}: {e}")
        return None


def main():
    """Main function to run the audit."""
    print("Starting codebase audit...")

    # Ensure output directory exists
    output_dir = project_root / "audit_results"
    output_dir.mkdir(exist_ok=True)

    # 1. count_lines_of_code
    print("Running count_lines_of_code...")
    result = run_tool("count_lines_of_code")
    if result:
        with open(output_dir / "count_lines_of_code.json", "w") as f:
            json.dump(result, f, indent=2)
        print("  Done.")
    else:
        print("  Failed.")

    # 2. list_undocumented_functions
    # We need to run this for each Python file
    print("Running list_undocumented_functions...")
    # Get a list of Python files from the index
    # For simplicity, we'll just run it on a few key files
    # A more comprehensive version would iterate through all indexed Python files
    python_files = [
        "codesage_mcp/main.py",
        "codesage_mcp/tools.py",
        "codesage_mcp/indexing.py",
        "codesage_mcp/searching.py",
        "codesage_mcp/llm_analysis.py",
    ]

    all_undocumented = {}
    for file_path in python_files:
        print(f"  Checking {file_path}...")
        result = run_tool("list_undocumented_functions", {"file_path": file_path})
        if result and "undocumented_functions" in result:
            all_undocumented[file_path] = result["undocumented_functions"]
        elif result and "error" in result:
            print(f"    Error: {result['error']}")

    with open(output_dir / "list_undocumented_functions.json", "w") as f:
        json.dump(all_undocumented, f, indent=2)
    print("  Done.")

    # 3. find_duplicate_code
    print("Running find_duplicate_code...")
    result = run_tool(
        "find_duplicate_code",
        {"codebase_path": ".", "min_similarity": 0.8, "min_lines": 10},
    )
    if result:
        with open(output_dir / "find_duplicate_code.json", "w") as f:
            json.dump(result, f, indent=2)
        print("  Done.")
    else:
        print("  Failed.")

    # 4. get_dependencies_overview
    print("Running get_dependencies_overview...")
    result = run_tool("get_dependencies_overview")
    if result:
        with open(output_dir / "get_dependencies_overview.json", "w") as f:
            json.dump(result, f, indent=2)
        print("  Done.")
    else:
        print("  Failed.")

    # 5. search_file_content for TODO, FIXME, BUG, OPTIMIZE
    print("Running search_file_content for TODO, FIXME, BUG, OPTIMIZE...")
    markers = ["TODO", "FIXME", "BUG", "OPTIMIZE"]
    all_markers = {}

    for marker in markers:
        print(f"  Searching for {marker}...")
        # This is a bit tricky because search_file_content is not a direct tool
        # We would need to implement a custom search or use a different approach
        # For now, let's simulate this by using a simple grep command
        try:
            grep_result = subprocess.run(
                ["grep", "-r", "--include=*.py", marker, "."],
                capture_output=True,
                text=True,
                cwd=project_root,
            )
            if (
                grep_result.returncode == 0 or grep_result.returncode == 1
            ):  # 0 = matches found, 1 = no matches
                matches = (
                    grep_result.stdout.strip().split("\n")
                    if grep_result.stdout.strip()
                    else []
                )
                all_markers[marker] = [
                    m for m in matches if m
                ]  # Filter out empty strings
            else:
                print(f"    Error running grep for {marker}: {grep_result.stderr}")
                all_markers[marker] = []
        except Exception as e:
            print(f"    Exception running grep for {marker}: {e}")
            all_markers[marker] = []

    with open(output_dir / "search_file_content.json", "w") as f:
        json.dump(all_markers, f, indent=2)
    print("  Done.")

    # 6. analyze_codebase_improvements_tool
    print("Running analyze_codebase_improvements...")
    result = run_tool("analyze_codebase_improvements", {"codebase_path": "."})
    if result:
        with open(output_dir / "analyze_codebase_improvements.json", "w") as f:
            json.dump(result, f, indent=2)
        print("  Done.")
    else:
        print("  Failed.")

    print("Audit completed. Results are in the 'audit_results' directory.")


if __name__ == "__main__":
    main()

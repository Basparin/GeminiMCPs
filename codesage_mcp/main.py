import logging
import json
from typing import List, Dict, Any, Union, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError

from codesage_mcp.tools import (
    read_code_file_tool,
    search_codebase_tool,
    semantic_search_codebase_tool,
    find_duplicate_code_tool,
    get_configuration_tool,
    analyze_codebase_improvements_tool, # Import the new tool function
    suggest_code_improvements_tool,  # Import the new code improvement tool
    summarize_code_section_tool,
    get_file_structure_tool,
    index_codebase_tool,
    list_undocumented_functions_tool,
    count_lines_of_code_tool,
    configure_api_key_tool,
    get_dependencies_overview_tool,
    profile_code_performance_tool,  # Import the new profiling tool
    generate_unit_tests_tool,  # Import the new test generation tool
    auto_document_tool,  # Import the new auto documentation tool
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Union[Dict[str, Any], List[Any], None] = None
    id: Union[str, int, None] = None


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Any] = None
    id: Union[str, int, None] = None


def get_all_tools_definitions_as_object():
    # Return tools as an object (dictionary) keyed by tool name
    return {
        "read_code_file": {
            "name": "read_code_file",
            "description": "Reads and returns the content of a specified code file.",
            "inputSchema": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
            "type": "function",
        },
        "index_codebase": {
            "name": "index_codebase",
            "description": "Indexes a given codebase path for analysis.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "type": "function",
        },
        "search_codebase": {
            "name": "search_codebase",
            "description": (
                "Searches for a pattern within indexed code files, "
                "with optional exclusion patterns."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "codebase_path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "file_types": {"type": "array", "items": {"type": "string"}},
                    "exclude_patterns": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["codebase_path", "pattern"],
            },
            "type": "function",
        },
        "semantic_search_codebase": {
            "name": "semantic_search_codebase",
            "description": (
                "Performs a semantic search within the indexed codebase to find "
                "code snippets semantically similar to the given query."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "codebase_path": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["codebase_path", "query"],
            },
            "type": "function",
        },
        "find_duplicate_code": {
            "name": "find_duplicate_code",
            "description": (
                "Finds duplicate code sections within the indexed codebase."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "codebase_path": {"type": "string"},
                    "min_similarity": {"type": "number", "default": 0.8},
                    "min_lines": {"type": "integer", "default": 10},
                },
                "required": ["codebase_path"],
            },
            "type": "function",
        },
        "get_file_structure": {
            "name": "get_file_structure",
            "description": (
                "Provides a high-level overview of a file's structure "
                "within a given codebase."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "codebase_path": {"type": "string"},
                    "file_path": {"type": "string"},
                },
                "required": ["codebase_path", "file_path"],
            },
            "type": "function",
        },
        "summarize_code_section": {
            "name": "summarize_code_section",
            "description": "Summarizes a specific section of code using a chosen LLM.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "llm_model": {"type": "string"},
                    "function_name": {"type": "string"},
                    "class_name": {"type": "string"},
                },
                "required": ["file_path"],
            },
            "type": "function",
        },
        "list_undocumented_functions": {
            "name": "list_undocumented_functions",
            "description": (
                "Identifies and lists Python functions in a specified file that "
                "are missing docstrings."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
            "type": "function",
        },
        "count_lines_of_code": {
            "name": "count_lines_of_code",
            "description": (
                "Counts lines of code (LOC) in the indexed codebase, "
                "providing a summary by file type."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "configure_api_key": {
            "name": "configure_api_key",
            "description": (
                "Configures API keys for LLMs (e.g., Groq, OpenRouter, "
                "Google AI)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "llm_provider": {"type": "string"},
                    "api_key": {"type": "string"},
                },
                "required": ["llm_provider", "api_key"],
            },
            "type": "function",
        },
        "get_dependencies_overview": {
            "name": "get_dependencies_overview",
            "description": (
                "Analyzes Python files in the indexed codebase and extracts "
                "import statements, providing a high-level overview of internal "
                "and external dependencies."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "profile_code_performance": {
            "name": "profile_code_performance",
            "description": (
                "Profiles the performance of a specific function or the entire file "
                "using cProfile to measure execution time and resource usage."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "function_name": {"type": "string"},
                },
                "required": ["file_path"],
            },
            "type": "function",
        },
        "suggest_code_improvements": {
            "name": "suggest_code_improvements",
            "description": (
                "Analyzes a code section and suggests improvements by consulting "
                "external LLMs. It identifies potential code quality issues and "
                "provides suggestions for improvements."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["file_path"],
            },
            "type": "function",
        },
        "get_configuration": {
            "name": "get_configuration",
            "description": (
                "Returns the current configuration, with API keys masked for security."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "analyze_codebase_improvements": {
            "name": "analyze_codebase_improvements",
            "description": (
                "Analyzes the codebase for potential improvements and suggestions."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "codebase_path": {"type": "string"},
                },
                "required": ["codebase_path"],
            },
            "type": "function",
        },
        "generate_unit_tests": {
            "name": "generate_unit_tests",
            "description": (
                "Generates unit tests for functions in a Python file. The generated tests "
                "can be manually reviewed and added to the test suite."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "function_name": {"type": "string"},
                },
                "required": ["file_path"],
            },
            "type": "function",
        },
        "auto_document": {
            "name": "auto_document",
            "description": (
                "Automatically generates documentation for tools that lack detailed documentation. "
                "Analyzes tool functions in the codebase, extracts their signatures and docstrings, "
                "and uses LLMs to generate human-readable documentation in the existing format."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                },
                "required": [],
            },
            "type": "function",
        },
    }


# Map tool names to their functions
TOOL_FUNCTIONS = {
    "read_code_file": read_code_file_tool,
    "index_codebase": index_codebase_tool,
    "search_codebase": search_codebase_tool,
    "semantic_search_codebase": semantic_search_codebase_tool,
    "find_duplicate_code": find_duplicate_code_tool,
    "get_file_structure": get_file_structure_tool,
    "summarize_code_section": summarize_code_section_tool,
    "list_undocumented_functions": list_undocumented_functions_tool,
    "count_lines_of_code": count_lines_of_code_tool,
    "configure_api_key": configure_api_key_tool,
    "get_dependencies_overview": get_dependencies_overview_tool,
    "get_configuration": get_configuration_tool,
    "analyze_codebase_improvements": analyze_codebase_improvements_tool, # Register the new tool
    "suggest_code_improvements": suggest_code_improvements_tool,  # Register the new code improvement tool
    "generate_unit_tests": generate_unit_tests_tool,  # Register the new test generation tool
    "auto_document": auto_document_tool,  # Register the new auto documentation tool
    "profile_code_performance": profile_code_performance_tool, # Register the new profiling tool
}


@app.get("/")
async def root():
    return {"message": "CodeSage MCP Server is running!"}


@app.post("/mcp")
async def handle_jsonrpc_request(request: Request):
    """
    Handles JSON-RPC requests, including tool discovery and notifications.
    """
    try:
        body = await request.json()
        jsonrpc_request = JSONRPCRequest(**body)

        if jsonrpc_request.method == "initialize":
            response_result = {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": "CodeSage MCP Server", "version": "0.1.0"},
                "capabilities": {"tools": get_all_tools_definitions_as_object()},
            }
            return JSONRPCResponse(result=response_result, id=jsonrpc_request.id)

        elif jsonrpc_request.method == "notifications/initialized":
            logger.info(
                "Received 'notifications/initialized' notification. Acknowledging."
            )
            return {}

        elif jsonrpc_request.method == "tools/list":
            return JSONRPCResponse(
                result={"tools": list(get_all_tools_definitions_as_object().values())},
                id=jsonrpc_request.id,
            )

        elif jsonrpc_request.method == "tools/call":
            if not jsonrpc_request.params or not isinstance(
                jsonrpc_request.params, dict
            ):
                raise HTTPException(
                    status_code=400, detail="Invalid params for tools/call."
                )

            tool_name = jsonrpc_request.params.get("name")
            tool_args = jsonrpc_request.params.get("arguments", {})

            if tool_name not in TOOL_FUNCTIONS:
                raise HTTPException(
                    status_code=404, detail=f"Tool not found: {tool_name}"
                )

            tool_function = TOOL_FUNCTIONS[tool_name]
            try:
                tool_result = tool_function(**tool_args)
                return JSONRPCResponse(result=tool_result, id=jsonrpc_request.id)
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Error executing tool {tool_name}: {e}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown JSON-RPC method: {jsonrpc_request.method}",
            )
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Error processing JSON-RPC request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON-RPC request: {e}")

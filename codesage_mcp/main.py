import logging
import json
from typing import List, Dict, Any, Union, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError

from codesage_mcp.tools import (
    read_code_file_tool,
    search_codebase_tool,
    summarize_code_section_tool,
    get_file_structure_tool,
    index_codebase_tool,
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
            "description": "Searches for a pattern within indexed code files.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "codebase_path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "file_types": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["codebase_path", "pattern"],
            },
            "type": "function",
        },
        "get_file_structure": {
            "name": "get_file_structure",
            "description": "Provides a high-level overview of a file's structure.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
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
                },
                "required": ["file_path", "start_line", "end_line", "llm_model"],
            },
            "type": "function",
        },
    }


# Map tool names to their functions
TOOL_FUNCTIONS = {
    "read_code_file": read_code_file_tool,
    "index_codebase": index_codebase_tool,
    "search_codebase": search_codebase_tool,
    "get_file_structure": get_file_structure_tool,
    "summarize_code_section": summarize_code_section_tool,
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
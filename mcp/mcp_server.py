import sys
import json

def handle_request(request):
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "capabilities": {
                    "tools": {
                        "dynamicRegistration": True
                    }
                }
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "greet",
                        "description": "Greets a person by name.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the person to greet."
                                }
                            },
                            "required": ["name"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        tool_name = params.get("toolName")
        tool_args = params.get("args", {})

        if tool_name == "greet":
            name = tool_args.get("name", "World")
            greeting = f"Hello, {name}!"
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": greeting
                        }
                    ]
                }
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool '{tool_name}' not found."
                }
            }
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32600,
                "message": "Invalid Request"
            }
        }

def main():
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break # EOF
            request = json.loads(line)
            response = handle_request(request)
            sys.stdout.write(json.dumps(response) + '\n')
            sys.stdout.flush()
        except json.JSONDecodeError:
            # Handle malformed JSON
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": "Parse error"
                }
            }
            sys.stdout.write(json.dumps(error_response) + '\n')
            sys.stdout.flush()
        except Exception as e:
            # Catch any other unexpected errors
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32000,
                    "message": f"Server error: {str(e)}"
                }
            }
            sys.stdout.write(json.dumps(error_response) + '\n')
            sys.stdout.flush()

if __name__ == "__main__":
    main()

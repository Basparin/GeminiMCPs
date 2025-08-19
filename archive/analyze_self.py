import requests
import json

# Server URL
url = "http://127.0.0.1:8000/mcp"

# Step 1: Index the codebase
print("Indexing the codebase...")
index_payload = {
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "index_codebase",
        "arguments": {
            "path": "/home/basparin/Escritorio/GeminiMCPs"
        }
    },
    "id": "1"
}

response = requests.post(url, json=index_payload)
print("Index response:", response.json())

# Step 2: Analyze the codebase for improvements
print("\nAnalyzing the codebase for improvements...")
analyze_payload = {
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "analyze_codebase_improvements",
        "arguments": {
            "codebase_path": "/home/basparin/Escritorio/GeminiMCPs"
        }
    },
    "id": "2"
}

response = requests.post(url, json=analyze_payload)
analysis_result = response.json()
print("Analysis response:", json.dumps(analysis_result, indent=2))

# Step 3: Get configuration to check what LLMs are available
print("\nGetting configuration...")
config_payload = {
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "get_configuration",
        "arguments": {}
    },
    "id": "3"
}

response = requests.post(url, json=config_payload)
config_result = response.json()
print("Configuration response:", json.dumps(config_result, indent=2))

# Step 4: Look for undocumented functions in a specific file
print("\nLooking for undocumented functions in codebase_manager.py...")
undocumented_payload = {
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "list_undocumented_functions",
        "arguments": {
            "file_path": "/home/basparin/Escritorio/GeminiMCPs/codesage_mcp/codebase_manager.py"
        }
    },
    "id": "4"
}

response = requests.post(url, json=undocumented_payload)
undocumented_result = response.json()
print("Undocumented functions response:", json.dumps(undocumented_result, indent=2))

# Step 5: Check for duplicate code
print("\nChecking for duplicate code...")
duplicate_payload = {
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "find_duplicate_code",
        "arguments": {
            "codebase_path": "/home/basparin/Escritorio/GeminiMCPs",
            "min_similarity": 0.8,
            "min_lines": 10
        }
    },
    "id": "5"
}

response = requests.post(url, json=duplicate_payload)
duplicate_result = response.json()
print("Duplicate code response:", json.dumps(duplicate_result, indent=2))
import requests
import json
import time

# Server URL
url = "http://127.0.0.1:8000/mcp"

def main():
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

    # Step 5: Check for duplicate code (with timeout handling)
    print("\nChecking for duplicate code...")
    duplicate_payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "find_duplicate_code",
            "arguments": {
                "codebase_path": "/home/basparin/Escritorio/GeminiMCPs",
                "min_similarity": 0.9,  # Increased similarity threshold
                "min_lines": 20  # Increased line count threshold
            }
        },
        "id": "5"
    }

    try:
        response = requests.post(url, json=duplicate_payload, timeout=30)
        if response.status_code == 200:
            duplicate_result = response.json()
            print("Duplicate code response:", json.dumps(duplicate_result, indent=2))
        else:
            print(f"Duplicate code search failed with status code: {response.status_code}")
            print("Response text:", response.text)
    except requests.exceptions.Timeout:
        print("Duplicate code search timed out after 30 seconds")
    except Exception as e:
        print(f"Error during duplicate code search: {e}")
        if 'response' in locals():
            print("Response text:", response.text)

    # Summary of findings
    print("\n" + "="*50)
    print("SUMMARY OF FINDINGS")
    print("="*50)
    
    if "result" in analysis_result and "analysis" in analysis_result["result"]:
        analysis = analysis_result["result"]["analysis"]
        print(f"Total files indexed: {analysis.get('total_files', 'N/A')}")
        print(f"Python files: {analysis.get('python_files', 'N/A')}")
        print(f"TODO comments: {analysis.get('todo_comments', 'N/A')}")
        print(f"FIXME comments: {analysis.get('fixme_comments', 'N/A')}")
        print(f"Undocumented functions (estimated): {analysis.get('undocumented_functions', 'N/A')}")
        print(f"Large files (>500 lines): {len(analysis.get('large_files', []))}")
        
        if "suggestions" in analysis:
            print("\nSuggestions:")
            for suggestion in analysis["suggestions"]:
                print(f"  - {suggestion}")

    if "result" in undocumented_result:
        undocumented = undocumented_result["result"]
        print(f"\nDetailed undocumented functions in codebase_manager.py: {undocumented.get('message', 'N/A')}")
        if "undocumented_functions" in undocumented:
            for func in undocumented["undocumented_functions"]:
                print(f"  - {func.get('name', 'Unknown')} (line {func.get('line_number', 'Unknown')})

if __name__ == "__main__":
    main()
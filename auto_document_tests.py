#!/usr/bin/env python3
"""
Script to automatically generate documentation for test functions.

This script analyzes test files and generates appropriate docstrings for 
undocumented test functions based on their names and parameters.
"""

import ast
import sys
import os
import re


def generate_test_docstring(func_name, params):
    """Generate a docstring for a test function based on its name and parameters.
    
    Args:
        func_name (str): Name of the test function
        params (list): List of parameter names
    
    Returns:
        str: Generated docstring
    """
    # Convert snake_case to readable text
    readable_name = func_name.replace('test_', '').replace('_', ' ')
    readable_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', readable_name)  # Handle camelCase
    readable_name = readable_name.capitalize()
    
    # Special handling for common test prefixes
    if func_name.startswith('test_parse_llm_response'):
        readable_name = readable_name.replace('Parse llm response', 'Parse LLM response')
    
    docstring = f'Test {readable_name}.\n'
    
    # Add more detailed description based on the function name
    if 'valid' in func_name.lower():
        docstring += f'\nThis test verifies that {readable_name.lower()} behaves correctly.\n'
    elif 'invalid' in func_name.lower():
        docstring += f'\nThis test verifies that {readable_name.lower()} handles invalid input appropriately.\n'
    elif 'success' in func_name.lower():
        docstring += f'\nThis test verifies successful execution of {readable_name.lower()}.\n'
    elif 'error' in func_name.lower():
        docstring += f'\nThis test verifies error handling in {readable_name.lower()}.\n'
    elif 'empty' in func_name.lower():
        docstring += f'\nThis test verifies behavior with empty input in {readable_name.lower()}.\n'
    
    # Add parameter documentation if there are parameters
    if params:
        docstring += '\nArgs:\n'
        for param in params:
            if 'fixture' in param or 'manager' in param:
                docstring += f'    {param}: Mocked {param.replace("_", " ").title()} instance.\n'
            else:
                docstring += f'    {param}: Test parameter.\n'
    
    return docstring


def document_test_file(file_path):
    """Document all undocumented functions in a test file.
    
    Args:
        file_path (str): Path to the test file
    """
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse the file with AST
    try:
        tree = ast.parse(''.join(lines))
    except SyntaxError as e:
        print(f"Error parsing {file_path}: {e}")
        return
    
    # Track modifications
    modified = False
    offset = 0  # Track line offset due to insertions
    
    # Find all function definitions
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if the function has a docstring
            has_docstring = (
                node.body 
                and isinstance(node.body[0], ast.Expr) 
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            )
            
            if not has_docstring:
                # Get parameter names
                params = [arg.arg for arg in node.args.args]
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'params': params,
                    'node': node
                })
    
    # Sort functions by line number (descending) to avoid line number shifts
    functions.sort(key=lambda x: x['line'], reverse=True)
    
    # Generate documentation for undocumented functions
    for func in functions:
        docstring = generate_test_docstring(func['name'], func['params'])
        
        # Format the docstring with proper indentation
        formatted_docstring = '    """' + docstring.strip() + '"""
'
        
        # Insert the docstring after the function definition
        insert_line = func['line'] + 1  # Line after the function definition
        
        # Adjust for previous modifications
        adjusted_line = insert_line + offset
        
        # Insert the docstring
        lines.insert(adjusted_line - 1, formatted_docstring)
        offset += 1
        modified = True
        
        print(f"  - Documented {func['name']}")
    
    # Write the modified content back to the file
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"\nModified {file_path}")
    else:
        print(f"\nNo undocumented functions found in {file_path}")


def main():
    """Main function to document test files."""
    test_files = [
        'tests/test_parse_llm_response.py',
        'tests/test_main.py',
        'tests/test_todo_fixme_resolution.py',
        'tests/test_codebase_manager.py',
        'tests/test_generate_llm_api_wrapper.py'
    ]
    
    print("=== AUTO DOCUMENTING TEST FUNCTIONS ===")
    for test_file in test_files:
        if os.path.exists(test_file):
            document_test_file(test_file)
        else:
            print(f"File not found: {test_file}")


if __name__ == "__main__":
    main()
"""
CES Tools - Adapted CodeSage Tools for CES AI Orchestration

Provides essential tools for CES operations, adapted from CodeSage MCP tools
for AI orchestration and task execution.
"""

import os
import re
import ast
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CESTools:
    """CES Tools adapted from CodeSage for AI orchestration"""

    def __init__(self, codebase_path: str = "."):
        self.codebase_path = Path(codebase_path)
        self.logger = logging.getLogger(__name__)

    def read_code_file(self, file_path: str) -> Dict[str, Any]:
        """Read and return content of a code file (adapted from CodeSage)"""
        try:
            full_path = self.codebase_path / file_path
            if not full_path.exists():
                return {
                    "status": "error",
                    "error": f"File not found: {file_path}",
                    "content": None
                }

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return {
                "status": "success",
                "file_path": file_path,
                "content": content,
                "line_count": len(content.splitlines())
            }

        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "content": None
            }

    def search_codebase(self, pattern: str, file_types: List[str] = None) -> Dict[str, Any]:
        """Search codebase for pattern (adapted from CodeSage)"""
        try:
            results = []
            file_types = file_types or ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h']

            for root, dirs, files in os.walk(self.codebase_path):
                for file in files:
                    if any(file.endswith(ext) for ext in file_types):
                        file_path = Path(root) / file
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                lines = content.splitlines()

                                for line_num, line in enumerate(lines, 1):
                                    if re.search(pattern, line, re.IGNORECASE):
                                        results.append({
                                            "file": str(file_path.relative_to(self.codebase_path)),
                                            "line": line_num,
                                            "content": line.strip(),
                                            "match": re.search(pattern, line, re.IGNORECASE).group()
                                        })

                        except Exception as e:
                            self.logger.warning(f"Error reading {file_path}: {e}")

            return {
                "status": "success",
                "pattern": pattern,
                "results": results,
                "total_matches": len(results)
            }

        except Exception as e:
            self.logger.error(f"Error searching codebase: {e}")
            return {
                "status": "error",
                "error": str(e),
                "results": []
            }

    def get_file_structure(self, file_path: str) -> Dict[str, Any]:
        """Get high-level structure of a file (adapted from CodeSage)"""
        try:
            full_path = self.codebase_path / file_path
            if not full_path.exists():
                return {
                    "status": "error",
                    "error": f"File not found: {file_path}"
                }

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse Python AST for structure
            if file_path.endswith('.py'):
                return self._analyze_python_structure(content, file_path)
            else:
                # Basic structure for other files
                lines = content.splitlines()
                return {
                    "status": "success",
                    "file_path": file_path,
                    "total_lines": len(lines),
                    "structure": {
                        "type": "generic",
                        "sections": []
                    }
                }

        except Exception as e:
            self.logger.error(f"Error analyzing file structure: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _analyze_python_structure(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze Python file structure using AST"""
        try:
            tree = ast.parse(content)

            structure = {
                "classes": [],
                "functions": [],
                "imports": []
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    structure["classes"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    })
                elif isinstance(node, ast.FunctionDef):
                    structure["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": len(node.args.args)
                    })
                elif isinstance(node, ast.Import):
                    structure["imports"].extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    structure["imports"].extend([f"{module}.{alias.name}" for alias in node.names])

            return {
                "status": "success",
                "file_path": file_path,
                "structure": structure,
                "total_lines": len(content.splitlines())
            }

        except SyntaxError as e:
            return {
                "status": "error",
                "error": f"Syntax error in Python file: {e}",
                "structure": {}
            }

    def count_lines_of_code(self) -> Dict[str, Any]:
        """Count lines of code by file type (adapted from CodeSage)"""
        try:
            stats = {}
            total_lines = 0

            for root, dirs, files in os.walk(self.codebase_path):
                for file in files:
                    file_path = Path(root) / file
                    ext = file_path.suffix

                    if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.css', '.html']:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])

                                if ext not in stats:
                                    stats[ext] = {"files": 0, "lines": 0}
                                stats[ext]["files"] += 1
                                stats[ext]["lines"] += code_lines
                                total_lines += code_lines

                        except Exception as e:
                            self.logger.warning(f"Error reading {file_path}: {e}")

            return {
                "status": "success",
                "stats": stats,
                "total_lines": total_lines
            }

        except Exception as e:
            self.logger.error(f"Error counting lines of code: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def get_dependencies_overview(self) -> Dict[str, Any]:
        """Get overview of project dependencies (adapted from CodeSage)"""
        try:
            dependencies = {
                "python": [],
                "javascript": [],
                "other": []
            }

            # Check for Python requirements
            req_files = ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile']
            for req_file in req_files:
                req_path = self.codebase_path / req_file
                if req_path.exists():
                    try:
                        if req_file == 'requirements.txt':
                            with open(req_path, 'r') as f:
                                for line in f:
                                    if line.strip() and not line.startswith('#'):
                                        pkg = line.split('==')[0].split('>=')[0].split('<')[0].strip()
                                        dependencies["python"].append(pkg)
                        elif req_file == 'setup.py':
                            # Basic parsing of setup.py
                            with open(req_path, 'r') as f:
                                content = f.read()
                                # Look for install_requires
                                if 'install_requires' in content:
                                    dependencies["python"].append("setup.py dependencies (parsed)")
                    except Exception as e:
                        self.logger.warning(f"Error parsing {req_file}: {e}")

            # Check for JavaScript dependencies
            package_json = self.codebase_path / 'package.json'
            if package_json.exists():
                try:
                    with open(package_json, 'r') as f:
                        data = json.load(f)
                        if 'dependencies' in data:
                            dependencies["javascript"].extend(list(data['dependencies'].keys()))
                except Exception as e:
                    self.logger.warning(f"Error parsing package.json: {e}")

            return {
                "status": "success",
                "dependencies": dependencies,
                "total_dependencies": sum(len(deps) for deps in dependencies.values())
            }

        except Exception as e:
            self.logger.error(f"Error getting dependencies overview: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def analyze_codebase_improvements(self) -> Dict[str, Any]:
        """Analyze codebase for potential improvements (adapted from CodeSage)"""
        try:
            improvements = {
                "code_quality": [],
                "performance": [],
                "security": [],
                "maintainability": []
            }

            # Walk through Python files
            for root, dirs, files in os.walk(self.codebase_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                lines = content.splitlines()

                                # Basic code quality checks
                                for i, line in enumerate(lines, 1):
                                    # Check for TODO/FIXME
                                    if 'TODO' in line.upper() or 'FIXME' in line.upper():
                                        improvements["maintainability"].append({
                                            "file": str(file_path.relative_to(self.codebase_path)),
                                            "line": i,
                                            "issue": "TODO/FIXME comment",
                                            "suggestion": "Address pending task"
                                        })

                                    # Check for print statements (potential debug code)
                                    if re.search(r'\bprint\s*\(', line):
                                        improvements["code_quality"].append({
                                            "file": str(file_path.relative_to(self.codebase_path)),
                                            "line": i,
                                            "issue": "Print statement in production code",
                                            "suggestion": "Replace with proper logging"
                                        })

                        except Exception as e:
                            self.logger.warning(f"Error analyzing {file_path}: {e}")

            return {
                "status": "success",
                "improvements": improvements,
                "total_issues": sum(len(issues) for issues in improvements.values())
            }

        except Exception as e:
            self.logger.error(f"Error analyzing codebase improvements: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


# Global CES tools instance
_ces_tools_instance = None


def get_ces_tools(codebase_path: str = ".") -> CESTools:
    """Get the global CES tools instance"""
    global _ces_tools_instance
    if _ces_tools_instance is None or _ces_tools_instance.codebase_path != Path(codebase_path):
        _ces_tools_instance = CESTools(codebase_path)
    return _ces_tools_instance
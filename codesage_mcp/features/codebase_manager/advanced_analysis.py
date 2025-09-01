"""
Advanced Analysis Module for CodeSage MCP Server.

This module provides advanced code analysis capabilities using the graph-based code model,
including enhanced dependency mapping and performance prediction.

Classes:
    AdvancedDependencyAnalyzer: Analyzes function-level dependencies and external library usage
    PerformancePredictor: Predicts performance bottlenecks based on code structure
    AdvancedAnalysisManager: Main manager class for advanced analysis operations
"""

import ast
import logging
import re
from collections import defaultdict, Counter
from typing import Dict, List, Any

from codesage_mcp.core.code_model import CodeGraph, CodeNode, NodeType, RelationshipType, LayerType
from codesage_mcp.config.config import ENABLE_CACHING
from codesage_mcp.features.caching.cache import get_cache_instance
from codesage_mcp.features.memory_management.memory_manager import get_memory_manager

# Set up logger
logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for performance
NESTED_COMPREHENSION_PATTERN = re.compile(r'\[.*for.*in.*\].*\[.*\]')
COMPREHENSION_PATTERN = re.compile(r'\[.*for.*\]')
STRING_CONCAT_PATTERN = re.compile(r'for.*\+.*(?:str|["\'])', re.IGNORECASE)


class AdvancedDependencyAnalyzer:
    """Analyzes function-level dependencies and external library usage using the code graph."""

    def __init__(self, graph: CodeGraph):
        self.graph = graph
        self.cache = get_cache_instance() if ENABLE_CACHING else None
        self.memory_manager = get_memory_manager()

    def analyze_function_dependencies(self, file_path: str, function_name: str = None) -> Dict[str, Any]:
        """
        Analyze function-level dependencies for a specific function or all functions in a file.

        Args:
            file_path: Path to the Python file
            function_name: Specific function name, or None for all functions

        Returns:
            Dictionary with dependency analysis results
        """
        try:
            # Get nodes for the file
            file_nodes = self.graph.get_file_nodes(file_path, LayerType.SEMANTIC)

            if not file_nodes:
                return {
                    "error": f"No code model found for {file_path}. Please generate code model first.",
                    "dependencies": {}
                }

            # Filter function nodes
            function_nodes = [node for node in file_nodes if node.node_type == NodeType.FUNCTION]

            if function_name:
                function_nodes = [node for node in function_nodes if node.name == function_name]
                if not function_nodes:
                    return {
                        "error": f"Function '{function_name}' not found in {file_path}",
                        "dependencies": {}
                    }

            dependencies = {}

            for func_node in function_nodes:
                func_deps = self._analyze_single_function_dependencies(func_node)
                dependencies[func_node.name] = func_deps

            return {
                "file_path": file_path,
                "function_name": function_name,
                "dependencies": dependencies,
                "summary": self._summarize_dependencies(dependencies)
            }

        except Exception as e:
            logger.error(f"Error analyzing function dependencies: {e}")
            return {
                "error": str(e),
                "dependencies": {}
            }

    def _analyze_single_function_dependencies(self, func_node: CodeNode) -> Dict[str, Any]:
        """Analyze dependencies for a single function node."""
        dependencies = {
            "direct_calls": [],
            "indirect_calls": [],
            "imports_used": [],
            "external_libraries": [],
            "internal_modules": [],
            "complexity_score": 0
        }

        # Get relationships for this function
        relationships = self.graph.get_node_relationships(func_node.id, LayerType.SEMANTIC)

        for rel in relationships:
            if rel.relationship_type == RelationshipType.CALLS:
                # Find the target node
                target_node = self.graph.get_node(rel.target_id)
                if target_node:
                    if target_node.node_type == NodeType.FUNCTION:
                        dependencies["direct_calls"].append({
                            "name": target_node.name,
                            "file": target_node.file_path,
                            "qualified_name": target_node.qualified_name
                        })
                    elif target_node.node_type == NodeType.METHOD:
                        dependencies["direct_calls"].append({
                            "name": target_node.name,
                            "file": target_node.file_path,
                            "qualified_name": target_node.qualified_name,
                            "type": "method"
                        })

        # Analyze function content for additional dependencies
        if func_node.content:
            content_deps = self._analyze_function_content_dependencies(func_node.content)
            dependencies.update(content_deps)

        # Calculate complexity score based on dependencies
        dependencies["complexity_score"] = self._calculate_dependency_complexity(dependencies)

        return dependencies

    def _analyze_function_content_dependencies(self, content: str) -> Dict[str, Any]:
        """Analyze function content for import usage and external dependencies."""
        imports_used = []
        external_libraries = set()
        internal_modules = set()

        try:
            tree = ast.parse(content)

            # Find all name usages
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    # Check if this name is from an import
                    # This is a simplified analysis - in practice, you'd need symbol resolution
                    if hasattr(node, 'id'):
                        name = node.id
                        # Check against known external libraries
                        if self._is_external_library(name):
                            external_libraries.add(name)
                        elif self._is_internal_module(name):
                            internal_modules.add(name)

            # Find import statements used within the function
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports_used.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        full_name = f"{module}.{alias.name}" if module else alias.name
                        imports_used.append(full_name)

        except SyntaxError:
            logger.warning("Could not parse function content for dependency analysis")

        return {
            "imports_used": imports_used,
            "external_libraries": list(external_libraries),
            "internal_modules": list(internal_modules)
        }

    def _is_external_library(self, name: str) -> bool:
        """Check if a name refers to an external library."""
        # Common external libraries - this could be expanded or made configurable
        external_libs = {
            'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn', 'tensorflow', 'torch',
            'requests', 'flask', 'django', 'fastapi', 'sqlalchemy', 'psycopg2', 'pymongo',
            'redis', 'celery', 'pytest', 'unittest', 'logging', 'os', 'sys', 'json',
            'datetime', 'collections', 'itertools', 'functools', 're', 'math', 'random'
        }
        return name in external_libs

    def _is_internal_module(self, name: str) -> bool:
        """Check if a name refers to an internal module."""
        # This would need to be populated based on the project's module structure
        # For now, return False - could be enhanced with project analysis
        return False

    def _calculate_dependency_complexity(self, dependencies: Dict[str, Any]) -> float:
        """Calculate a complexity score based on dependencies."""
        score = 0.0

        # Direct calls contribute to complexity
        score += len(dependencies.get("direct_calls", [])) * 1.0

        # External libraries add complexity
        score += len(dependencies.get("external_libraries", [])) * 0.5

        # Indirect dependencies add less complexity
        score += len(dependencies.get("indirect_calls", [])) * 0.3

        return round(score, 2)

    def _summarize_dependencies(self, dependencies: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of dependency analysis."""
        total_functions = len(dependencies)
        total_direct_calls = sum(len(func_deps.get("direct_calls", [])) for func_deps in dependencies.values())
        total_external_libs = len(set(
            lib for func_deps in dependencies.values()
            for lib in func_deps.get("external_libraries", [])
        ))

        avg_complexity = sum(
            func_deps.get("complexity_score", 0) for func_deps in dependencies.values()
        ) / total_functions if total_functions > 0 else 0

        return {
            "total_functions_analyzed": total_functions,
            "total_direct_calls": total_direct_calls,
            "unique_external_libraries": total_external_libs,
            "average_complexity_score": round(avg_complexity, 2)
        }

    def analyze_external_library_usage(self, file_path: str = None) -> Dict[str, Any]:
        """
        Analyze external library usage across files or a specific file.

        Args:
            file_path: Optional specific file path, or None for all files

        Returns:
            Dictionary with external library usage analysis
        """
        try:
            if file_path:
                files_to_analyze = [file_path]
            else:
                # Get all Python files from the graph
                all_files = set()
                for layer in self.graph.layers.values():
                    for node in layer.nodes.values():
                        if node.file_path.endswith('.py'):
                            all_files.add(node.file_path)
                files_to_analyze = list(all_files)

            library_usage = defaultdict(lambda: {"files": [], "functions": []})

            for file in files_to_analyze:
                file_nodes = self.graph.get_file_nodes(file, LayerType.SEMANTIC)

                for node in file_nodes:
                    if node.node_type in [NodeType.FUNCTION, NodeType.METHOD]:
                        deps = self._analyze_single_function_dependencies(node)
                        for lib in deps.get("external_libraries", []):
                            library_usage[lib]["files"].append(file)
                            library_usage[lib]["functions"].append({
                                "name": node.name,
                                "file": file
                            })

            # Remove duplicates
            for lib_data in library_usage.values():
                lib_data["files"] = list(set(lib_data["files"]))
                # Keep unique functions
                seen = set()
                unique_functions = []
                for func in lib_data["functions"]:
                    func_key = (func["name"], func["file"])
                    if func_key not in seen:
                        seen.add(func_key)
                        unique_functions.append(func)
                lib_data["functions"] = unique_functions

            return {
                "file_path": file_path,
                "library_usage": dict(library_usage),
                "summary": {
                    "total_libraries": len(library_usage),
                    "most_used_library": max(library_usage.keys(), key=lambda k: len(library_usage[k]["functions"])) if library_usage else None
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing external library usage: {e}")
            return {
                "error": str(e),
                "library_usage": {}
            }


class PerformancePredictor:
    """Predicts performance bottlenecks based on code structure analysis."""

    def __init__(self, graph: CodeGraph):
        self.graph = graph
        self.cache = get_cache_instance() if ENABLE_CACHING else None
        self.memory_manager = get_memory_manager()

    def predict_bottlenecks(self, file_path: str = None) -> Dict[str, Any]:
        """
        Predict potential performance bottlenecks in code.

        Args:
            file_path: Optional specific file path, or None for all files

        Returns:
            Dictionary with bottleneck predictions
        """
        try:
            if file_path:
                files_to_analyze = [file_path]
            else:
                # Get all Python files from the graph
                all_files = set()
                for layer in self.graph.layers.values():
                    for node in layer.nodes.values():
                        if node.file_path.endswith('.py'):
                            all_files.add(node.file_path)
                files_to_analyze = list(all_files)

            bottlenecks = []

            for file in files_to_analyze:
                file_bottlenecks = self._analyze_file_bottlenecks(file)
                bottlenecks.extend(file_bottlenecks)

            # Sort by severity
            bottlenecks.sort(key=lambda x: x.get("severity_score", 0), reverse=True)

            return {
                "file_path": file_path,
                "bottlenecks": bottlenecks[:20],  # Top 20 bottlenecks
                "summary": self._summarize_bottlenecks(bottlenecks)
            }

        except Exception as e:
            logger.error(f"Error predicting bottlenecks: {e}")
            return {
                "error": str(e),
                "bottlenecks": []
            }

    def _analyze_file_bottlenecks(self, file_path: str) -> List[Dict[str, Any]]:
        """Analyze a single file for potential bottlenecks."""
        bottlenecks = []

        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')

            # Get function nodes for this file
            file_nodes = self.graph.get_file_nodes(file_path, LayerType.SEMANTIC)
            function_nodes = [node for node in file_nodes if node.node_type == NodeType.FUNCTION]

            for func_node in function_nodes:
                func_bottlenecks = self._analyze_function_bottlenecks(func_node, lines)
                bottlenecks.extend(func_bottlenecks)

        except Exception as e:
            logger.exception(f"Could not analyze bottlenecks for {file_path}: {e}")

        return bottlenecks

    def _analyze_function_bottlenecks(self, func_node: CodeNode, lines: List[str]) -> List[Dict[str, Any]]:
        """Analyze a single function for potential bottlenecks."""
        bottlenecks = []

        try:
            # Extract function content
            start_line = func_node.start_line
            end_line = func_node.end_line

            if start_line <= len(lines):
                func_content = '\n'.join(lines[start_line-1:end_line])

                # Analyze for common bottleneck patterns
                bottleneck_checks = [
                    self._check_nested_loops,
                    self._check_large_data_structures,
                    self._check_inefficient_operations,
                    self._check_recursive_calls,
                    self._check_complex_comprehensions
                ]

                for check_func in bottleneck_checks:
                    func_bottlenecks = check_func(func_node, func_content)
                    bottlenecks.extend(func_bottlenecks)

        except Exception as e:
            logger.warning(f"Could not analyze function {func_node.name}: {e}")

        return bottlenecks

    def _check_nested_loops(self, func_node: CodeNode, content: str) -> List[Dict[str, Any]]:
        """Check for nested loops that could be performance bottlenecks."""
        bottlenecks = []

        try:
            tree = ast.parse(content)

            def find_nested_loops(node, depth=0, max_depth=0):
                current_depth = depth
                if isinstance(node, (ast.For, ast.While)):
                    current_depth = depth + 1
                    max_depth = max(max_depth, current_depth)

                for child in ast.iter_child_nodes(node):
                    max_depth = find_nested_loops(child, current_depth, max_depth)

                return max_depth

            max_nesting = find_nested_loops(tree)

            if max_nesting >= 3:
                bottlenecks.append({
                    "type": "nested_loops",
                    "function": func_node.name,
                    "file": func_node.file_path,
                    "line": func_node.start_line,
                    "description": f"Deeply nested loops (depth {max_nesting}) may cause performance issues",
                    "severity_score": min(max_nesting * 2, 10),
                    "suggestion": "Consider optimizing loop structure or using more efficient algorithms"
                })

        except SyntaxError:
            pass

        return bottlenecks

    def _check_large_data_structures(self, func_node: CodeNode, content: str) -> List[Dict[str, Any]]:
        """Check for operations on large data structures."""
        bottlenecks = []

        # Look for list/dict comprehensions that might be inefficient
        if NESTED_COMPREHENSION_PATTERN.search(content):
            bottlenecks.append({
                "type": "large_data_structures",
                "function": func_node.name,
                "file": func_node.file_path,
                "line": func_node.start_line,
                "description": "Nested list/dict comprehensions may be inefficient for large datasets",
                "severity_score": 6,
                "suggestion": "Consider using generator expressions or traditional loops for better memory efficiency"
            })

        return bottlenecks

    def _check_inefficient_operations(self, func_node: CodeNode, content: str) -> List[Dict[str, Any]]:
        """Check for inefficient operations."""
        bottlenecks = []

        # Check for string concatenation in loops using compiled regex
        if STRING_CONCAT_PATTERN.search(content):
            bottlenecks.append({
                "type": "inefficient_string_operations",
                "function": func_node.name,
                "file": func_node.file_path,
                "line": func_node.start_line,
                "description": "String concatenation in loops can be inefficient",
                "severity_score": 5,
                "suggestion": "Use ''.join() or list accumulation for better performance"
            })

        return bottlenecks

    def _check_recursive_calls(self, func_node: CodeNode, content: str) -> List[Dict[str, Any]]:
        """Check for recursive function calls that might cause stack overflow."""
        bottlenecks = []

        # Simple check for self-calls (this is basic - could be enhanced)
        func_name = func_node.name
        if func_name in content and "def " + func_name in content:
            # Count occurrences of function name (rough heuristic)
            name_count = content.count(func_name)
            if name_count > 2:  # Function definition + at least one call
                bottlenecks.append({
                    "type": "recursion",
                    "function": func_node.name,
                    "file": func_node.file_path,
                    "line": func_node.start_line,
                    "description": "Potential recursive function calls detected",
                    "severity_score": 7,
                    "suggestion": "Ensure recursion has proper base case and consider iterative alternatives for deep recursion"
                })

        return bottlenecks

    def _check_complex_comprehensions(self, func_node: CodeNode, content: str) -> List[Dict[str, Any]]:
        """Check for overly complex comprehensions."""
        bottlenecks = []

        # Look for very long comprehensions using compiled regex
        comprehensions = COMPREHENSION_PATTERN.findall(content)

        for comp in comprehensions:
            if len(comp) > 100:  # Very long comprehension
                bottlenecks.append({
                    "type": "complex_comprehension",
                    "function": func_node.name,
                    "file": func_node.file_path,
                    "line": func_node.start_line,
                    "description": "Very complex list/dict comprehension may be hard to read and maintain",
                    "severity_score": 4,
                    "suggestion": "Consider breaking down complex comprehensions into multiple steps or using traditional loops"
                })

        return bottlenecks

    def _summarize_bottlenecks(self, bottlenecks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of bottleneck analysis."""
        if not bottlenecks:
            return {"total_bottlenecks": 0, "severity_distribution": {}}

        severity_counts = Counter(b.get("severity_score", 0) for b in bottlenecks)
        type_counts = Counter(b.get("type", "unknown") for b in bottlenecks)

        return {
            "total_bottlenecks": len(bottlenecks),
            "severity_distribution": dict(severity_counts),
            "bottleneck_types": dict(type_counts),
            "high_severity_count": sum(1 for b in bottlenecks if b.get("severity_score", 0) >= 7)
        }


class AdvancedAnalysisManager:
    """Main manager class for advanced analysis operations."""

    def __init__(self, graph: CodeGraph = None):
        self.graph = graph or CodeGraph()
        self.dependency_analyzer = AdvancedDependencyAnalyzer(self.graph)
        self.performance_predictor = PerformancePredictor(self.graph)
        self.cache = get_cache_instance() if ENABLE_CACHING else None
        self.memory_manager = get_memory_manager()

    def run_comprehensive_analysis(self, file_path: str = None) -> Dict[str, Any]:
        """
        Run comprehensive advanced analysis on files.

        Args:
            file_path: Optional specific file path, or None for all files

        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            # Run dependency analysis
            dependency_results = self.dependency_analyzer.analyze_function_dependencies(file_path)

            # Run external library analysis
            library_results = self.dependency_analyzer.analyze_external_library_usage(file_path)

            # Run performance prediction
            bottleneck_results = self.performance_predictor.predict_bottlenecks(file_path)

            return {
                "file_path": file_path,
                "timestamp": self.graph.layers[LayerType.SEMANTIC].nodes[list(self.graph.layers[LayerType.SEMANTIC].nodes.keys())[0]].updated_at if self.graph.layers[LayerType.SEMANTIC].nodes else None,
                "dependency_analysis": dependency_results,
                "library_analysis": library_results,
                "performance_analysis": bottleneck_results,
                "summary": {
                    "total_functions_analyzed": dependency_results.get("summary", {}).get("total_functions_analyzed", 0),
                    "total_bottlenecks_found": len(bottleneck_results.get("bottlenecks", [])),
                    "total_libraries_used": library_results.get("summary", {}).get("total_libraries", 0)
                }
            }

        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                "error": str(e),
                "dependency_analysis": {},
                "library_analysis": {},
                "performance_analysis": {}
            }

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get statistics about the advanced analysis capabilities."""
        return {
            "graph_stats": self.graph.get_statistics(),
            "memory_stats": self.memory_manager.get_memory_stats() if self.memory_manager else {},
            "cache_enabled": self.cache is not None,
            "supported_analyses": [
                "function_dependency_analysis",
                "external_library_usage",
                "performance_bottleneck_prediction",
                "comprehensive_code_analysis"
            ]
        } }
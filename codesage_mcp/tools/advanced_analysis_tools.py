"""Advanced Analysis Tools Module for CodeSage MCP Server.

This module provides tools for advanced code analysis capabilities including
enhanced dependency mapping and performance prediction.
"""

from codesage_mcp.features.codebase_manager import AdvancedAnalysisManager
from codesage_mcp.core.code_model import CodeGraph
from codesage_mcp.core.utils import tool_error_handler

# Initialize the advanced analysis manager
_advanced_analysis_manager = None

def get_advanced_analysis_manager():
    """Get or create the advanced analysis manager instance."""
    global _advanced_analysis_manager
    if _advanced_analysis_manager is None:
        graph = CodeGraph()
        _advanced_analysis_manager = AdvancedAnalysisManager(graph)
    return _advanced_analysis_manager


@tool_error_handler
def analyze_function_dependencies_tool(file_path: str, function_name: str = None) -> dict:
    """Analyze function-level dependencies for a specific function or all functions in a file.

    Args:
        file_path (str): Path to the Python file to analyze.
        function_name (str, optional): Specific function name to analyze. If None, analyzes all functions.

    Returns:
        dict: Dependency analysis results with function dependencies, external libraries, and complexity scores.
    """
    manager = get_advanced_analysis_manager()
    result = manager.dependency_analyzer.analyze_function_dependencies(file_path, function_name)

    return {
        "message": f"Function dependency analysis completed for {file_path}" +
                  (f" (function: {function_name})" if function_name else ""),
        "analysis": result
    }


@tool_error_handler
def analyze_external_library_usage_tool(file_path: str = None) -> dict:
    """Analyze external library usage across files or a specific file.

    Args:
        file_path (str, optional): Specific file path to analyze. If None, analyzes all files.

    Returns:
        dict: External library usage analysis with library statistics and usage patterns.
    """
    manager = get_advanced_analysis_manager()
    result = manager.dependency_analyzer.analyze_external_library_usage(file_path)

    file_desc = f"file {file_path}" if file_path else "all files"
    return {
        "message": f"External library usage analysis completed for {file_desc}",
        "analysis": result
    }


@tool_error_handler
def predict_performance_bottlenecks_tool(file_path: str = None) -> dict:
    """Predict potential performance bottlenecks in code based on structural analysis.

    Args:
        file_path (str, optional): Specific file path to analyze. If None, analyzes all files.

    Returns:
        dict: Performance bottleneck predictions with severity scores and recommendations.
    """
    manager = get_advanced_analysis_manager()
    result = manager.performance_predictor.predict_bottlenecks(file_path)

    file_desc = f"file {file_path}" if file_path else "all files"
    return {
        "message": f"Performance bottleneck prediction completed for {file_desc}",
        "analysis": result
    }


@tool_error_handler
def run_comprehensive_advanced_analysis_tool(file_path: str = None) -> dict:
    """Run comprehensive advanced analysis combining dependency mapping and performance prediction.

    Args:
        file_path (str, optional): Specific file path to analyze. If None, analyzes all files.

    Returns:
        dict: Comprehensive analysis results including dependencies, libraries, and performance insights.
    """
    manager = get_advanced_analysis_manager()
    result = manager.run_comprehensive_analysis(file_path)

    file_desc = f"file {file_path}" if file_path else "all files"
    return {
        "message": f"Comprehensive advanced analysis completed for {file_desc}",
        "analysis": result
    }


@tool_error_handler
def get_advanced_analysis_stats_tool() -> dict:
    """Get statistics about the advanced analysis capabilities and current state.

    Returns:
        dict: Statistics about the advanced analysis system including supported analyses and performance metrics.
    """
    manager = get_advanced_analysis_manager()
    stats = manager.get_analysis_stats()

    return {
        "message": "Advanced analysis statistics retrieved successfully",
        "stats": stats
    }
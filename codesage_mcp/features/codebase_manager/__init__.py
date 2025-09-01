"""
Codebase Manager Package for CodeSage MCP Server.

This package provides the main CodebaseManager class and advanced analysis capabilities
for managing and analyzing codebases.
"""

from .codebase_manager import CodebaseManager, codebase_manager, get_llm_analysis_manager
from .advanced_analysis import AdvancedDependencyAnalyzer, PerformancePredictor, AdvancedAnalysisManager

__all__ = [
    'CodebaseManager',
    'codebase_manager',
    'get_llm_analysis_manager',
    'AdvancedDependencyAnalyzer',
    'PerformancePredictor',
    'AdvancedAnalysisManager'
]
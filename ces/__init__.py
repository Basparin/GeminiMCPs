"""
Cognitive Enhancement System (CES) - Bootstrap Edition

A human-AI collaborative development system that enhances cognitive capabilities
through intelligent task delegation, context management, and adaptive learning.

This module provides the main entry point for CES functionality.
"""

__version__ = "0.1.0"
__author__ = "CES Development Team"

from .core.cognitive_agent import CognitiveAgent
from .ai_orchestrator.ai_assistant import AIOrchestrator
from .codesage_integration import CodeSageIntegration, CESToolExtensions

__all__ = ["CognitiveAgent", "AIOrchestrator", "CodeSageIntegration", "CESToolExtensions"]
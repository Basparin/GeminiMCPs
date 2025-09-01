"""Main Module for CodeSage MCP Server.

This module defines the FastAPI application and handles JSON-RPC requests for the CodeSage MCP Server.
It registers all available tools and provides endpoints for the Gemini CLI to interact with.

The server exposes a set of tools to the Gemini CLI:
- Codebase Indexing
- File Reading
- Code Search
- File Structure Overview
- LLM-Powered Code Summarization
- Duplicate Code Detection
- And many more...

It also integrates with various Large Language Models (LLMs) like Groq, OpenRouter, and Google AI
for specialized tasks like code analysis and summarization.
"""

import json
import time
from typing import List, Dict, Any, Union, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .logging_config import setup_logging, get_logger, log_exception, log_errors

from codesage_mcp.tools import (
    read_code_file_tool,
    search_codebase_tool,
    semantic_search_codebase_tool,
    find_duplicate_code_tool,
    get_configuration_tool,
    analyze_codebase_improvements_tool,  # Import the new tool function
    suggest_code_improvements_tool,  # Import the new code improvement tool
    summarize_code_section_tool,
    get_file_structure_tool,
    index_codebase_tool,
    list_undocumented_functions_tool,
    count_lines_of_code_tool,
    configure_api_key_tool,
    get_dependencies_overview_tool,
    profile_code_performance_tool,  # Import the new profiling tool
    generate_unit_tests_tool,  # Import the new test generation tool
    auto_document_tool,  # Import the new auto documentation tool
    resolve_todo_fixme_tool,  # Import the new TODO/FIXME resolution tool
    parse_llm_response_tool,  # Import the new LLM response parsing tool
    generate_llm_api_wrapper_tool,  # Import the new LLM API wrapper generation tool
    generate_boilerplate_tool,  # Import the new boilerplate generation tool
    get_cache_statistics_tool,  # Import the new cache statistics tool
    get_performance_metrics_tool,  # Import the new performance monitoring tool
    get_performance_report_tool,  # Import the new performance report tool
    get_usage_patterns_tool,  # Import the new usage patterns tool
    get_predictive_analytics_tool,  # Import the new predictive analytics tool
    collect_user_feedback_tool,  # Import the new user feedback tools
    get_feedback_summary_tool,
    get_user_insights_tool,
    analyze_feedback_patterns_tool,
    get_feedback_driven_recommendations_tool,
    get_user_satisfaction_metrics_tool,
    analyze_performance_trends_tool,  # Import the new trend analysis tools
    get_optimization_opportunities_tool,
    predict_performance_capacity_tool,
    forecast_performance_trends_tool,
    get_performance_baseline_comparison_tool,
    analyze_cache_effectiveness_tool,  # Import the new cache analysis tools
    get_cache_optimization_recommendations_tool,
    get_cache_performance_metrics_tool,
    get_cache_access_patterns_tool,
    get_cache_memory_efficiency_tool,
    analyze_memory_patterns_tool,  # Import the new memory pattern tools
    get_adaptive_memory_management_tool,
    optimize_memory_for_load_tool,
    get_memory_pressure_analysis_tool,
    get_memory_optimization_opportunities_tool,
    get_adaptive_cache_status_tool,  # Import the new adaptive cache tools
    trigger_cache_adaptation_tool,
    get_cache_sizing_recommendations_tool,
    analyze_cache_adaptation_effectiveness_tool,
    get_cache_adaptation_rules_tool,
    get_workload_analysis_tool,  # Import the new workload-adaptive memory tools
    get_memory_allocation_status_tool,
    trigger_workload_adaptation_tool,
    get_workload_optimization_recommendations_tool,
    analyze_workload_performance_impact_tool,
    get_workload_adaptation_rules_tool,
    get_prefetch_analysis_tool,  # Import the new intelligent prefetch tools
    trigger_prefetching_tool,
    get_prefetch_performance_metrics_tool,
    analyze_prefetch_patterns_tool,
    get_prefetch_configuration_tool,
    update_prefetch_configuration_tool,
    get_performance_tuning_analysis_tool,  # Import the new automatic performance tuning tools
    trigger_performance_tuning_tool,
    get_tuning_recommendations_tool,
    analyze_tuning_effectiveness_tool,
    get_tuning_configuration_tool,
    update_tuning_configuration_tool,
    get_workload_pattern_analysis_tool,  # Import the new workload pattern recognition tools
    trigger_pattern_based_allocation_tool,
    get_resource_allocation_status_tool,
    forecast_workload_patterns_tool,
    analyze_pattern_effectiveness_tool,
    get_pattern_recognition_configuration_tool,
    update_pattern_recognition_configuration_tool,
    analyze_continuous_improvement_opportunities_tool,  # Import the new continuous improvement tools
    implement_automated_improvements_tool,
    monitor_improvement_effectiveness_tool,
    detect_performance_regressions_tool,  # Import the new regression detection tool
)
from codesage_mcp.tools.advanced_analysis_tools import (
    analyze_function_dependencies_tool,
    analyze_external_library_usage_tool,
    predict_performance_bottlenecks_tool,
    run_comprehensive_advanced_analysis_tool,
    get_advanced_analysis_stats_tool,
)
from codesage_mcp.utils import create_error_response
from codesage_mcp.gemini_compatibility import (
    get_compatibility_handler,
    create_gemini_compatible_error_response,
    adapt_response_for_gemini,
    ResponseFormat,
    GeminiCompatibilityHandler
)
from codesage_mcp.performance_monitor import get_performance_monitor, get_usage_analyzer
from codesage_mcp.user_feedback import get_user_feedback_collector
from codesage_mcp.trend_analysis import get_trend_analyzer
from codesage_mcp.auto_performance_tuner import get_auto_performance_tuner
from codesage_mcp.adaptive_cache_manager import get_adaptive_cache_manager
from codesage_mcp.memory_manager import get_memory_manager
from codesage_mcp.workload_pattern_recognition import get_workload_pattern_recognition
from codesage_mcp.regression_detector import get_regression_detector
from codesage_mcp.codebase_manager import get_llm_analysis_manager

# Configure structured logging
setup_logging(
    level="INFO",
    log_file="logs/codesage.log",
    json_format=True
)
logger = get_logger(__name__)


class GeminiCompatibleJSONResponse(JSONResponse):
    """Custom JSON response that excludes None fields for Gemini CLI compatibility."""

    def render(self, content: Any) -> bytes:
        # Recursively remove None fields from the response
        def remove_none_fields(obj):
            if isinstance(obj, dict):
                return {k: remove_none_fields(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                return [remove_none_fields(item) for item in obj]
            else:
                return obj

        cleaned_content = remove_none_fields(content)
        return super().render(cleaned_content)

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("Incoming request", method=request.method, url=str(request.url), headers=dict(request.headers))
    response = await call_next(request)
    return response

# Initialize performance monitoring
performance_monitor = get_performance_monitor()
usage_analyzer = get_usage_analyzer()
feedback_collector = get_user_feedback_collector()
trend_analyzer = get_trend_analyzer()

# Initialize self-optimization features
auto_performance_tuner = get_auto_performance_tuner()
adaptive_cache_manager = get_adaptive_cache_manager()
memory_manager = get_memory_manager()
workload_pattern_recognition = get_workload_pattern_recognition()


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

    def model_dump(self, *args, **kwargs):
        """Override model_dump method to comply with JSON-RPC 2.0 specification."""
        # Use exclude_none=True to prevent None fields from being serialized
        data = super().model_dump(*args, exclude_none=True, **kwargs)

        # Additional validation to prevent schema issues
        if 'error' in data and 'result' in data:
            logger.error("CRITICAL: Response contains both 'error' and 'result' fields after model_dump!")
            data.pop('result', None)

        return data

    def dict(self, *args, **kwargs):
        """Deprecated dict method for backward compatibility."""
        return self.model_dump(*args, **kwargs)

    @classmethod
    def create_compatible_response(cls,
                                   result: Optional[Any] = None,
                                   error: Optional[Dict[str, Any]] = None,
                                   request_id: Union[str, int, None] = None) -> 'JSONRPCResponse':
        """Create a response using Gemini compatibility handler."""
        compatibility_handler = get_compatibility_handler()

        # Ensure error codes are numeric (JSON-RPC 2.0 specification)
        if error is not None and isinstance(error.get('code'), str):
            # Convert string error code to numeric using compatibility handler
            error = compatibility_handler.adapt_error_response(error, ResponseFormat.GEMINI_NUMERIC_ERRORS)

        # Use compatibility handler to create adapted response
        adapted_response = compatibility_handler.create_compatible_response(
            result=result,
            error=error,
            request_id=request_id
        )

        # Create response object from adapted data, but filter out None values
        filtered_response = {k: v for k, v in adapted_response.items() if v is not None}
        return cls(**filtered_response)


def get_all_tools_definitions_as_object():
    """
    Returns a dictionary of all available tool definitions, keyed by tool name.

    This function is used internally to provide tool metadata for the
    `initialize` and `tools/list` JSON-RPC methods. It aggregates the
    definitions of all registered tools into a single object for easy access.

    Returns:
        dict: A dictionary where keys are tool names (e.g., 'read_code_file')
              and values are dictionaries containing the tool's metadata
              (name, description, inputSchema, type).

    Note:
        The actual tool implementations are mapped separately in the
        `TOOL_FUNCTIONS` dictionary.
    """
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
            "description": (
                "Enhanced search tool with graph-based semantic search and dependency-aware results. "
                "Supports regex, semantic, and graph-based search modes with configurable context depth."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "codebase_path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "file_types": {"type": "array", "items": {"type": "string"}},
                    "exclude_patterns": {"type": "array", "items": {"type": "string"}},
                    "search_mode": {
                        "type": "string",
                        "enum": ["regex", "semantic", "graph"],
                        "default": "regex"
                    },
                    "context_depth": {"type": "integer", "default": 1, "minimum": 1, "maximum": 3},
                    "include_dependencies": {"type": "boolean", "default": True},
                },
                "required": ["codebase_path", "pattern"],
            },
            "type": "function",
        },
        "semantic_search_codebase": {
            "name": "semantic_search_codebase",
            "description": (
                "Performs a semantic search within the indexed codebase to find "
                "code snippets semantically similar to the given query."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "codebase_path": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["codebase_path", "query"],
            },
            "type": "function",
        },
        "find_duplicate_code": {
            "name": "find_duplicate_code",
            "description": (
                "Finds duplicate code sections within the indexed codebase."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "codebase_path": {"type": "string"},
                    "min_similarity": {"type": "number", "default": 0.8},
                    "min_lines": {"type": "integer", "default": 10},
                },
                "required": ["codebase_path"],
            },
            "type": "function",
        },
        "get_file_structure": {
            "name": "get_file_structure",
            "description": (
                "Provides a high-level overview of a file's structure "
                "within a given codebase."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "codebase_path": {"type": "string"},
                    "file_path": {"type": "string"},
                },
                "required": ["codebase_path", "file_path"],
            },
            "type": "function",
        },
        "summarize_code_section": {
            "name": "summarize_code_section",
            "description": (
                "Enhanced code summarization with performance insights and dependency analysis. "
                "Provides LLM-based summary along with performance bottleneck predictions and dependency mapping."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "llm_model": {"type": "string"},
                    "function_name": {"type": "string"},
                    "class_name": {"type": "string"},
                    "include_performance_insights": {"type": "boolean", "default": True},
                    "include_dependency_analysis": {"type": "boolean", "default": True},
                },
                "required": ["file_path"],
            },
            "type": "function",
        },
        "list_undocumented_functions": {
            "name": "list_undocumented_functions",
            "description": (
                "Identifies and lists Python functions in a specified file that "
                "are missing docstrings."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
            "type": "function",
        },
        "count_lines_of_code": {
            "name": "count_lines_of_code",
            "description": (
                "Counts lines of code (LOC) in the indexed codebase, "
                "providing a summary by file type."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"codebase_path": {"type": "string"}},
                "required": ["codebase_path"],
            },
            "type": "function",
        },
        "configure_api_key": {
            "name": "configure_api_key",
            "description": (
                "Configures API keys for LLMs (e.g., Groq, OpenRouter, Google AI)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "llm_provider": {"type": "string"},
                    "api_key": {"type": "string"},
                },
                "required": ["llm_provider", "api_key"],
            },
            "type": "function",
        },
        "get_dependencies_overview": {
            "name": "get_dependencies_overview",
            "description": (
                "Analyzes Python files in the indexed codebase and extracts "
                "import statements, providing a high-level overview of internal "
                "and external dependencies."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"codebase_path": {"type": "string"}},
                "required": ["codebase_path"],
            },
            "type": "function",
        },
        "profile_code_performance": {
            "name": "profile_code_performance",
            "description": (
                "Profiles the performance of a specific function or the entire file "
                "using cProfile to measure execution time and resource usage."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "function_name": {"type": "string"},
                },
                "required": ["file_path"],
            },
            "type": "function",
        },
        "suggest_code_improvements": {
            "name": "suggest_code_improvements",
            "description": (
                "Analyzes a code section and suggests improvements by consulting "
                "external LLMs. It identifies potential code quality issues and "
                "provides suggestions for improvements."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["file_path"],
            },
            "type": "function",
        },
        "get_configuration": {
            "name": "get_configuration",
            "description": (
                "Returns the current configuration, with API keys masked for security."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "analyze_codebase_improvements": {
            "name": "analyze_codebase_improvements",
            "description": (
                "Analyzes the codebase for potential improvements and suggestions."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "codebase_path": {"type": "string"},
                },
                "required": ["codebase_path"],
            },
            "type": "function",
        },
        "generate_unit_tests": {
            "name": "generate_unit_tests",
            "description": (
                "Generates unit tests for functions in a Python file. The generated tests "
                "can be manually reviewed and added to the test suite."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "function_name": {"type": "string"},
                },
                "required": ["file_path"],
            },
            "type": "function",
        },
        "auto_document_tool": {
            "name": "auto_document_tool",
            "description": (
                "Automatically generates documentation for tools that lack detailed documentation. "
                "Analyzes tool functions in the codebase, extracts their signatures and docstrings, "
                "and uses LLMs to generate human-readable documentation in the existing format."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                },
                "required": [],
            },
            "type": "function",
        },
        "resolve_todo_fixme": {
            "name": "resolve_todo_fixme",
            "description": (
                "Analyzes a TODO/FIXME comment and suggests potential resolutions using LLMs."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "line_number": {"type": "integer"},
                },
                "required": ["file_path"],
            },
            "type": "function",
        },
        "generate_boilerplate": {
            "name": "generate_boilerplate",
            "description": (
                "Generates standardized boilerplate code for new modules, tools, or tests. "
                "Supports file headers, module templates, tool functions, test scaffolding, "
                "classes, and functions."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "boilerplate_type": {"type": "string"},
                    "file_path": {"type": "string"},
                    "module_name": {"type": "string"},
                    "function_name": {"type": "string"},
                    "class_name": {"type": "string"},
                },
                "required": ["boilerplate_type"],
            },
            "type": "function",
        },
        "parse_llm_response": {
            "name": "parse_llm_response",
            "description": (
                "Parses the content of an LLM response, extracting and validating JSON data."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "llm_response_content": {"type": "string"},
                },
                "required": ["llm_response_content"],
            },
            "type": "function",
        },
        "generate_llm_api_wrapper": {
            "name": "generate_llm_api_wrapper",
            "description": (
                "Generates Python wrapper code for interacting with various LLM APIs."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "llm_provider": {"type": "string"},
                    "model_name": {"type": "string"},
                    "api_key_env_var": {"type": "string"},
                    "output_file_path": {"type": "string"},
                },
                "required": ["llm_provider", "model_name"],
            },
            "type": "function",
        },
        "get_cache_statistics": {
            "name": "get_cache_statistics",
            "description": (
                "Returns comprehensive statistics about the intelligent caching system, "
                "including hit rates, cache sizes, and performance metrics."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "get_performance_metrics": {
            "name": "get_performance_metrics",
            "description": (
                "Returns current real-time performance metrics including response times, "
                "resource utilization, throughput, and error rates."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "get_performance_report": {
            "name": "get_performance_report",
            "description": (
                "Generates a comprehensive performance report with metrics summary, "
                "baseline status, alerts, and optimization recommendations."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "get_usage_patterns": {
            "name": "get_usage_patterns",
            "description": (
                "Analyzes and returns usage patterns across different user profiles "
                "to identify optimization opportunities."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "get_predictive_analytics": {
            "name": "get_predictive_analytics",
            "description": (
                "Provides predictive analytics including resource usage forecasting, "
                "anomaly detection, and optimization recommendations."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "collect_user_feedback": {
            "name": "collect_user_feedback",
            "description": (
                "Collect user feedback for analysis and improvement. Supports different "
                "feedback types including bug reports, feature requests, performance issues, "
                "usability feedback, and general feedback."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "feedback_type": {
                        "type": "string",
                        "enum": ["bug_report", "feature_request", "performance_issue", "usability_feedback", "general_feedback"]
                    },
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "satisfaction_level": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "User satisfaction level (1=Very Dissatisfied, 5=Very Satisfied)"
                    },
                    "user_id": {"type": "string", "default": "anonymous"},
                    "metadata": {"type": "object"}
                },
                "required": ["feedback_type", "title", "description"],
            },
            "type": "function",
        },
        "get_feedback_summary": {
            "name": "get_feedback_summary",
            "description": (
                "Get a summary of user feedback data including statistics, trends, "
                "and analysis by type and user."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "feedback_type": {
                        "type": "string",
                        "enum": ["bug_report", "feature_request", "performance_issue", "usability_feedback", "general_feedback"]
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_user_insights": {
            "name": "get_user_insights",
            "description": (
                "Get insights about a specific user's behavior, satisfaction trends, "
                "engagement level, and usage patterns."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
            "type": "function",
        },
        "analyze_feedback_patterns": {
            "name": "analyze_feedback_patterns",
            "description": (
                "Analyze patterns in user feedback to identify trends, common themes, "
                "and improvement opportunities."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "get_feedback_driven_recommendations": {
            "name": "get_feedback_driven_recommendations",
            "description": (
                "Generate prioritized recommendations for product improvement based on "
                "user feedback analysis and usage patterns."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "get_user_satisfaction_metrics": {
            "name": "get_user_satisfaction_metrics",
            "description": (
                "Get comprehensive user satisfaction metrics including scores, trends, "
                "distribution, and key drivers of satisfaction/dissatisfaction."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "analyze_performance_trends": {
            "name": "analyze_performance_trends",
            "description": (
                "Analyze performance trends for specific metrics or all metrics over time, "
                "including trend direction, slope analysis, confidence levels, and predictions."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "Optional specific metric to analyze (e.g., 'response_time_ms', 'memory_usage_percent')"
                    },
                    "analysis_window_days": {
                        "type": "integer",
                        "default": 30,
                        "description": "Number of days to analyze (default: 30)"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_optimization_opportunities": {
            "name": "get_optimization_opportunities",
            "description": (
                "Identify and prioritize optimization opportunities based on performance trends, "
                "including detailed analysis, implementation guidance, and expected benefits."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "predict_performance_capacity": {
            "name": "predict_performance_capacity",
            "description": (
                "Predict maximum workload capacity for target performance levels, "
                "including capacity headroom analysis and scaling recommendations."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target_response_time_ms": {
                        "type": "integer",
                        "default": 100,
                        "description": "Target response time in milliseconds (default: 100ms)"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "forecast_performance_trends": {
            "name": "forecast_performance_trends",
            "description": (
                "Forecast future performance trends using predictive analytics, "
                "including confidence intervals and trend analysis."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "Name of the metric to forecast"
                    },
                    "forecast_days": {
                        "type": "integer",
                        "default": 30,
                        "description": "Number of days to forecast (default: 30)"
                    }
                },
                "required": ["metric_name"],
            },
            "type": "function",
        },
        "get_performance_baseline_comparison": {
            "name": "get_performance_baseline_comparison",
            "description": (
                "Compare current performance against established baselines, "
                "including compliance status and recommendations for improvement."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "analyze_cache_effectiveness": {
            "name": "analyze_cache_effectiveness",
            "description": (
                "Analyze cache effectiveness in real-world scenarios for specific cache types "
                "or all caches, including performance metrics, effectiveness ratings, and recommendations."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cache_type": {
                        "type": "string",
                        "description": "Optional specific cache type to analyze (e.g., 'embedding', 'search', 'file')"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_cache_optimization_recommendations": {
            "name": "get_cache_optimization_recommendations",
            "description": (
                "Get cache optimization recommendations based on effectiveness analysis, "
                "including detailed implementation steps, expected benefits, and prioritization."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "get_cache_performance_metrics": {
            "name": "get_cache_performance_metrics",
            "description": (
                "Get detailed cache performance metrics for monitoring and analysis, "
                "including hit/miss rates, memory usage, latency, and performance trends."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cache_type": {
                        "type": "string",
                        "description": "Optional specific cache type to analyze"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_cache_access_patterns": {
            "name": "get_cache_access_patterns",
            "description": (
                "Analyze cache access patterns to identify optimization opportunities, "
                "including hit/miss patterns, temporal access patterns, and key frequency analysis."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cache_type": {
                        "type": "string",
                        "description": "Optional specific cache type to analyze"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_cache_memory_efficiency": {
            "name": "get_cache_memory_efficiency",
            "description": (
                "Analyze cache memory efficiency and provide optimization recommendations, "
                "including memory usage analysis, efficiency scores, and reallocation suggestions."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "analyze_memory_patterns": {
            "name": "analyze_memory_patterns",
            "description": (
                "Analyze memory usage patterns under varying loads, including statistics, "
                "detected patterns, trends, and optimization opportunities."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "analysis_window_hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "Number of hours to analyze (default: 24)"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_adaptive_memory_management": {
            "name": "get_adaptive_memory_management",
            "description": (
                "Get adaptive memory management recommendations based on current patterns, "
                "including memory pressure assessment and adaptation strategies."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "optimize_memory_for_load": {
            "name": "optimize_memory_for_load",
            "description": (
                "Optimize memory settings for a specific load level, including optimal "
                "memory settings and detailed implementation plans."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "load_level": {
                        "type": "string",
                        "enum": ["idle", "light", "moderate", "heavy", "critical"],
                        "description": "Target load level for optimization"
                    }
                },
                "required": ["load_level"],
            },
            "type": "function",
        },
        "get_memory_pressure_analysis": {
            "name": "get_memory_pressure_analysis",
            "description": (
                "Analyze current memory pressure and provide detailed insights, "
                "including pressure sources, mitigation strategies, and predictive analysis."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "get_memory_optimization_opportunities": {
            "name": "get_memory_optimization_opportunities",
            "description": (
                "Identify memory optimization opportunities based on usage patterns, "
                "including pattern-based and load-aware optimization strategies."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "get_adaptive_cache_status": {
            "name": "get_adaptive_cache_status",
            "description": (
                "Get the current status of adaptive cache management, including "
                "adaptation activity, recent decisions, rules status, and effectiveness metrics."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "trigger_cache_adaptation": {
            "name": "trigger_cache_adaptation",
            "description": (
                "Trigger manual cache adaptation for specified cache type or all caches, "
                "including adaptation decisions, expected impacts, and implementation status."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cache_type": {
                        "type": "string",
                        "description": "Optional specific cache type ('embedding', 'search', 'file')"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["performance_based", "usage_pattern_based", "load_aware", "predictive", "hybrid"],
                        "default": "hybrid",
                        "description": "Adaptation strategy to use"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_cache_sizing_recommendations": {
            "name": "get_cache_sizing_recommendations",
            "description": (
                "Get cache sizing recommendations for a specific cache type using different strategies, "
                "including optimal sizes, implementation plans, and expected performance impact."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cache_type": {
                        "type": "string",
                        "enum": ["embedding", "search", "file"],
                        "description": "Cache type to analyze"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["performance_based", "usage_pattern_based", "load_aware", "predictive", "hybrid"],
                        "default": "hybrid",
                        "description": "Sizing strategy to use"
                    }
                },
                "required": ["cache_type"],
            },
            "type": "function",
        },
        "analyze_cache_adaptation_effectiveness": {
            "name": "analyze_cache_adaptation_effectiveness",
            "description": (
                "Analyze the effectiveness of cache adaptations over time, "
                "including success rates, performance impact, trends, and improvement recommendations."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time_window_hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "Number of hours to analyze (default: 24)"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_cache_adaptation_rules": {
            "name": "get_cache_adaptation_rules",
            "description": (
                "Get information about cache adaptation rules and their performance, "
                "including rule definitions, success rates, effectiveness analysis, and tuning recommendations."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "get_workload_analysis": {
            "name": "get_workload_analysis",
            "description": (
                "Get comprehensive workload analysis including current patterns, trends, and predictions, "
                "with insights, health scores, and optimization recommendations."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "get_memory_allocation_status": {
            "name": "get_memory_allocation_status",
            "description": (
                "Get current memory allocation status and effectiveness analysis, "
                "including allocation plans, performance impact, and optimization opportunities."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "trigger_workload_adaptation": {
            "name": "trigger_workload_adaptation",
            "description": (
                "Trigger manual workload adaptation with specified strategy, "
                "including adaptation decision, expected impact, and implementation status."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["conservative", "balanced", "aggressive", "predictive", "adaptive"],
                        "default": "adaptive",
                        "description": "Memory allocation strategy to use"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_workload_optimization_recommendations": {
            "name": "get_workload_optimization_recommendations",
            "description": (
                "Get workload optimization recommendations based on analysis, "
                "including pattern optimization, memory allocation strategies, and implementation roadmap."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "analyze_workload_performance_impact": {
            "name": "analyze_workload_performance_impact",
            "description": (
                "Analyze the performance impact of workload adaptations over time, "
                "including effectiveness metrics, performance trends, and strategy recommendations."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time_window_hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "Number of hours to analyze (default: 24)"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_workload_adaptation_rules": {
            "name": "get_workload_adaptation_rules",
            "description": (
                "Get information about workload adaptation rules and their effectiveness, "
                "including rule definitions, performance metrics, and tuning recommendations."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "get_prefetch_analysis": {
            "name": "get_prefetch_analysis",
            "description": (
                "Get comprehensive intelligent prefetching analysis including patterns, metrics, "
                "current candidates, effectiveness analysis, and optimization opportunities."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "trigger_prefetching": {
            "name": "trigger_prefetching",
            "description": (
                "Trigger intelligent prefetching with specified strategy and parameters, "
                "including candidate selection, execution results, and performance impact."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["pattern_based", "predictive", "collaborative", "hybrid"],
                        "default": "hybrid",
                        "description": "Prefetching strategy to use"
                    },
                    "max_candidates": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of files to prefetch"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_prefetch_performance_metrics": {
            "name": "get_prefetch_performance_metrics",
            "description": (
                "Get detailed prefetching performance metrics over a specified time window, "
                "including success rates, accuracy, performance impact, and trend analysis."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time_window_hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "Number of hours to analyze (default: 24)"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "analyze_prefetch_patterns": {
            "name": "analyze_prefetch_patterns",
            "description": (
                "Analyze prefetching patterns and their effectiveness, "
                "including pattern discovery, effectiveness analysis, evolution, and optimization recommendations."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern_type": {
                        "type": "string",
                        "enum": ["temporal", "spatial", "sequential", "all"],
                        "default": "all",
                        "description": "Type of patterns to analyze"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_prefetch_configuration": {
            "name": "get_prefetch_configuration",
            "description": (
                "Get current prefetching configuration and provide tuning recommendations, "
                "including current settings, effectiveness analysis, and performance impact."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "update_prefetch_configuration": {
            "name": "update_prefetch_configuration",
            "description": (
                "Update prefetching configuration with new settings, "
                "including validation, impact assessment, and rollback plan."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "config_updates": {
                        "type": "object",
                        "description": "Dictionary of configuration updates"
                    }
                },
                "required": ["config_updates"],
            },
            "type": "function",
        },
        "get_performance_tuning_analysis": {
            "name": "get_performance_tuning_analysis",
            "description": (
                "Get comprehensive automatic performance tuning analysis including current parameters, "
                "performance history, experiment results, and optimization opportunities."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "trigger_performance_tuning": {
            "name": "trigger_performance_tuning",
            "description": (
                "Trigger automatic performance tuning with specified strategy and parameters, "
                "including recommendations generation, experiment execution, and tunings application."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["bayesian_optimization", "gradient_descent", "genetic_algorithm", "reinforcement_learning", "hybrid"],
                        "default": "hybrid",
                        "description": "Tuning strategy to use"
                    },
                    "max_experiments": {
                        "type": "integer",
                        "default": 3,
                        "description": "Maximum number of tuning experiments to run"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_tuning_recommendations": {
            "name": "get_tuning_recommendations",
            "description": (
                "Get specific tuning recommendations based on current system state, "
                "including parameter-specific recommendations, implementation plans, and risk assessments."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "confidence_threshold": {
                        "type": "number",
                        "default": 0.6,
                        "description": "Minimum confidence score for recommendations (0-1)"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "analyze_tuning_effectiveness": {
            "name": "analyze_tuning_effectiveness",
            "description": (
                "Analyze the effectiveness of automatic performance tuning over time, "
                "including success rates, performance improvements, and optimization opportunities."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time_window_hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "Number of hours to analyze (default: 24)"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_tuning_configuration": {
            "name": "get_tuning_configuration",
            "description": (
                "Get current automatic performance tuning configuration and provide tuning recommendations, "
                "including current settings, effectiveness analysis, and performance impact."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "update_tuning_configuration": {
            "name": "update_tuning_configuration",
            "description": (
                "Update automatic performance tuning configuration with new settings, "
                "including validation, impact assessment, and rollback plan."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "config_updates": {
                        "type": "object",
                        "description": "Dictionary of configuration updates"
                    }
                },
                "required": ["config_updates"],
            },
            "type": "function",
        },
        "get_workload_pattern_analysis": {
            "name": "get_workload_pattern_analysis",
            "description": (
                "Get comprehensive workload pattern analysis including current patterns, metrics, "
                "resource allocations, and optimization opportunities."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "trigger_pattern_based_allocation": {
            "name": "trigger_pattern_based_allocation",
            "description": (
                "Trigger pattern-based resource allocation with specified parameters, "
                "including pattern detection, resource allocation recommendations, and implementation results."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern_type": {
                        "type": "string",
                        "enum": ["auto", "compute_intensive", "memory_intensive", "io_intensive", "bursty", "steady_state"],
                        "default": "auto",
                        "description": "Type of pattern to optimize for"
                    },
                    "resource_focus": {
                        "type": "string",
                        "enum": ["cpu", "memory", "balanced", "cache"],
                        "default": "balanced",
                        "description": "Resource allocation focus"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_resource_allocation_status": {
            "name": "get_resource_allocation_status",
            "description": (
                "Get current resource allocation status and effectiveness analysis, "
                "including allocation history, utilization analysis, and optimization opportunities."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "forecast_workload_patterns": {
            "name": "forecast_workload_patterns",
            "description": (
                "Forecast workload patterns and resource needs for the specified time horizon, "
                "including predicted patterns, resource forecasts, and proactive recommendations."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time_horizon_hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "Number of hours to forecast (default: 24)"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "analyze_pattern_effectiveness": {
            "name": "analyze_pattern_effectiveness",
            "description": (
                "Analyze the effectiveness of workload pattern recognition and resource allocation, "
                "including detection accuracy, allocation success rates, and performance impact."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time_window_hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "Number of hours to analyze (default: 24)"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "get_pattern_recognition_configuration": {
            "name": "get_pattern_recognition_configuration",
            "description": (
                "Get current workload pattern recognition configuration and provide tuning recommendations, "
                "including detection parameters, pattern types, and effectiveness analysis."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "update_pattern_recognition_configuration": {
            "name": "update_pattern_recognition_configuration",
            "description": (
                "Update workload pattern recognition configuration with new settings, "
                "including validation, impact assessment, and rollback plan."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "config_updates": {
                        "type": "object",
                        "description": "Dictionary of configuration updates"
                    }
                },
                "required": ["config_updates"],
            },
            "type": "function",
        },
        "analyze_continuous_improvement_opportunities": {
            "name": "analyze_continuous_improvement_opportunities",
            "description": (
                "Analyze production data to identify optimization opportunities and areas for improvement, "
                "including performance trends, user feedback insights, and automated recommendations."
            ),
            "inputSchema": {"type": "object", "properties": {}, "required": []},
            "type": "function",
        },
        "implement_automated_improvements": {
            "name": "implement_automated_improvements",
            "description": (
                "Implement automated improvements based on analysis results, "
                "with option for dry-run simulation."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dry_run": {
                        "type": "boolean",
                        "default": True,
                        "description": "If True, only simulate improvements without applying them"
                    }
                },
                "required": [],
            },
            "type": "function",
        },
        "monitor_improvement_effectiveness": {
            "name": "monitor_improvement_effectiveness",
            "description": (
                "Monitor the effectiveness of implemented improvements over a specified time window."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time_window_hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "Number of hours to analyze for improvement effectiveness"
                    }
                },
                "required": [],
            },
        "detect_performance_regressions": {
            "name": "detect_performance_regressions",
            "description": (
                "Detect performance regressions by comparing current benchmark results against baseline. "
                "Uses statistical analysis to identify significant performance changes and provides "
                "recommendations for remediation, including automated alerts and issue creation."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "current_results": {
                        "type": "object",
                        "description": "Current benchmark results to analyze (optional - uses latest if not provided)"
                    }
                },
                "required": []
            },
            "type": "function",
        },
        "analyze_function_dependencies": {
            "name": "analyze_function_dependencies",
            "description": (
                "Analyze function-level dependencies for a specific function or all functions in a file. "
                "Provides detailed dependency mapping including direct calls, indirect calls, external libraries, "
                "and complexity scoring based on dependency patterns."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the Python file to analyze"
                    },
                    "function_name": {
                        "type": "string",
                        "description": "Specific function name to analyze (optional - analyzes all if not provided)"
                    }
                },
                "required": ["file_path"]
            },
            "type": "function",
        },
        "analyze_external_library_usage": {
            "name": "analyze_external_library_usage",
            "description": (
                "Analyze external library usage across files or a specific file. "
                "Identifies which external libraries are used, their frequency, and usage patterns."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Specific file path to analyze (optional - analyzes all files if not provided)"
                    }
                },
                "required": []
            },
            "type": "function",
        },
        "predict_performance_bottlenecks": {
            "name": "predict_performance_bottlenecks",
            "description": (
                "Predict potential performance bottlenecks in code based on structural analysis. "
                "Identifies nested loops, inefficient operations, large data structures, and other "
                "performance-critical patterns with severity scoring and recommendations."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Specific file path to analyze (optional - analyzes all files if not provided)"
                    }
                },
                "required": []
            },
            "type": "function",
        },
        "run_comprehensive_advanced_analysis": {
            "name": "run_comprehensive_advanced_analysis",
            "description": (
                "Run comprehensive advanced analysis combining dependency mapping and performance prediction. "
                "Provides a complete analysis including function dependencies, external library usage, "
                "performance bottlenecks, and actionable insights."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Specific file path to analyze (optional - analyzes all files if not provided)"
                    }
                },
                "required": []
            },
            "type": "function",
        },
        "get_advanced_analysis_stats": {
            "name": "get_advanced_analysis_stats",
            "description": (
                "Get statistics about the advanced analysis capabilities and current state. "
                "Provides information about supported analyses, graph statistics, and system performance."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            },
            "type": "function",
        },
            "type": "function",
        },
    }


# Map tool names to their functions
TOOL_FUNCTIONS = {
    "read_code_file": read_code_file_tool,
    "index_codebase": index_codebase_tool,
    "search_codebase": search_codebase_tool,
    "semantic_search_codebase": semantic_search_codebase_tool,
    "find_duplicate_code": find_duplicate_code_tool,
    "get_file_structure": get_file_structure_tool,
    "summarize_code_section": summarize_code_section_tool,
    "list_undocumented_functions": list_undocumented_functions_tool,
    "count_lines_of_code": count_lines_of_code_tool,
    "configure_api_key": configure_api_key_tool,
    "get_dependencies_overview": get_dependencies_overview_tool,
    "get_configuration": get_configuration_tool,
    "analyze_codebase_improvements": analyze_codebase_improvements_tool,  # Register the new tool
    "suggest_code_improvements": suggest_code_improvements_tool,  # Register the new code improvement tool
    "generate_unit_tests": generate_unit_tests_tool,  # Register the new test generation tool
    "auto_document_tool": auto_document_tool,  # Register the new auto documentation tool
    "profile_code_performance": profile_code_performance_tool,  # Register the new profiling tool
    "resolve_todo_fixme": resolve_todo_fixme_tool,  # Register the new TODO/FIXME resolution tool
    "parse_llm_response": parse_llm_response_tool,  # Register the new LLM response parsing tool
    "generate_llm_api_wrapper": generate_llm_api_wrapper_tool,  # Register the new LLM API wrapper generation tool
    "generate_boilerplate": generate_boilerplate_tool,  # Register the new boilerplate generation tool
    "get_cache_statistics": get_cache_statistics_tool,  # Register the new cache statistics tool
    "get_performance_metrics": get_performance_metrics_tool,  # Register the new performance metrics tool
    "get_performance_report": get_performance_report_tool,  # Register the new performance report tool
    "get_usage_patterns": get_usage_patterns_tool,  # Register the new usage patterns tool
    "get_predictive_analytics": get_predictive_analytics_tool,  # Register the new predictive analytics tool
    "collect_user_feedback": collect_user_feedback_tool,  # Register the new user feedback tools
    "get_feedback_summary": get_feedback_summary_tool,
    "get_user_insights": get_user_insights_tool,
    "analyze_feedback_patterns": analyze_feedback_patterns_tool,
    "get_feedback_driven_recommendations": get_feedback_driven_recommendations_tool,
    "get_user_satisfaction_metrics": get_user_satisfaction_metrics_tool,
    "analyze_performance_trends": analyze_performance_trends_tool,  # Register the new trend analysis tools
    "get_optimization_opportunities": get_optimization_opportunities_tool,
    "predict_performance_capacity": predict_performance_capacity_tool,
    "forecast_performance_trends": forecast_performance_trends_tool,
    "get_performance_baseline_comparison": get_performance_baseline_comparison_tool,
    "analyze_cache_effectiveness": analyze_cache_effectiveness_tool,  # Register the new cache analysis tools
    "get_cache_optimization_recommendations": get_cache_optimization_recommendations_tool,
    "get_cache_performance_metrics": get_cache_performance_metrics_tool,
    "get_cache_access_patterns": get_cache_access_patterns_tool,
    "get_cache_memory_efficiency": get_cache_memory_efficiency_tool,
    "analyze_memory_patterns": analyze_memory_patterns_tool,  # Register the new memory pattern tools
    "get_adaptive_memory_management": get_adaptive_memory_management_tool,
    "optimize_memory_for_load": optimize_memory_for_load_tool,
    "get_memory_pressure_analysis": get_memory_pressure_analysis_tool,
    "get_memory_optimization_opportunities": get_memory_optimization_opportunities_tool,
    "get_adaptive_cache_status": get_adaptive_cache_status_tool,  # Register the new adaptive cache tools
    "trigger_cache_adaptation": trigger_cache_adaptation_tool,
    "get_cache_sizing_recommendations": get_cache_sizing_recommendations_tool,
    "analyze_cache_adaptation_effectiveness": analyze_cache_adaptation_effectiveness_tool,
    "get_cache_adaptation_rules": get_cache_adaptation_rules_tool,
    "get_workload_analysis": get_workload_analysis_tool,  # Register the new workload-adaptive memory tools
    "get_memory_allocation_status": get_memory_allocation_status_tool,
    "trigger_workload_adaptation": trigger_workload_adaptation_tool,
    "get_workload_optimization_recommendations": get_workload_optimization_recommendations_tool,
    "analyze_workload_performance_impact": analyze_workload_performance_impact_tool,
    "get_workload_adaptation_rules": get_workload_adaptation_rules_tool,
    "get_prefetch_analysis": get_prefetch_analysis_tool,  # Register the new intelligent prefetch tools
    "trigger_prefetching": trigger_prefetching_tool,
    "get_prefetch_performance_metrics": get_prefetch_performance_metrics_tool,
    "analyze_prefetch_patterns": analyze_prefetch_patterns_tool,
    "get_prefetch_configuration": get_prefetch_configuration_tool,
    "update_prefetch_configuration": update_prefetch_configuration_tool,
    "get_performance_tuning_analysis": get_performance_tuning_analysis_tool,  # Register the new automatic performance tuning tools
    "trigger_performance_tuning": trigger_performance_tuning_tool,
    "get_tuning_recommendations": get_tuning_recommendations_tool,
    "analyze_tuning_effectiveness": analyze_tuning_effectiveness_tool,
    "get_tuning_configuration": get_tuning_configuration_tool,
    "update_tuning_configuration": update_tuning_configuration_tool,
    "get_workload_pattern_analysis": get_workload_pattern_analysis_tool,  # Register the new workload pattern recognition tools
    "trigger_pattern_based_allocation": trigger_pattern_based_allocation_tool,
    "get_resource_allocation_status": get_resource_allocation_status_tool,
    "forecast_workload_patterns": forecast_workload_patterns_tool,
    "analyze_pattern_effectiveness": analyze_pattern_effectiveness_tool,
    "detect_performance_regressions": detect_performance_regressions_tool,
    "get_pattern_recognition_configuration": get_pattern_recognition_configuration_tool,
    "update_pattern_recognition_configuration": update_pattern_recognition_configuration_tool,
    "analyze_continuous_improvement_opportunities": analyze_continuous_improvement_opportunities_tool,
    "implement_automated_improvements": implement_automated_improvements_tool,
    "monitor_improvement_effectiveness": monitor_improvement_effectiveness_tool,
    "analyze_function_dependencies": analyze_function_dependencies_tool,
    "analyze_external_library_usage": analyze_external_library_usage_tool,
    "predict_performance_bottlenecks": predict_performance_bottlenecks_tool,
    "run_comprehensive_advanced_analysis": run_comprehensive_advanced_analysis_tool,
    "get_advanced_analysis_stats": get_advanced_analysis_stats_tool,
}


@app.get("/")
async def root_get():
    return {"message": "CodeSage MCP Server is running!"}

@app.post("/")
async def root_post():
    return {"message": "CodeSage MCP Server is running!"}


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint for monitoring and alerting.
    Returns metrics in Prometheus exposition format.
    """
    try:
        # Get current performance metrics
        current_metrics = performance_monitor.get_current_metrics()
        report = performance_monitor.get_performance_report()

        # Format metrics for Prometheus
        prometheus_output = []

        # Add basic info metric
        prometheus_output.append("# HELP codesage_mcp_info CodeSage MCP Server information")
        prometheus_output.append("# TYPE codesage_mcp_info gauge")
        prometheus_output.append('codesage_mcp_info{version="0.1.0"} 1')

        # Add uptime metric
        uptime_seconds = report.get("uptime_seconds", 0)
        prometheus_output.append("# HELP codesage_mcp_uptime_seconds Time since server start")
        prometheus_output.append("# TYPE codesage_mcp_uptime_seconds counter")
        prometheus_output.append(f"codesage_mcp_uptime_seconds {uptime_seconds}")

        # Add request metrics
        prometheus_output.append("# HELP codesage_mcp_requests_total Total number of requests processed")
        prometheus_output.append("# TYPE codesage_mcp_requests_total counter")
        prometheus_output.append(f"codesage_mcp_requests_total {performance_monitor.request_count}")

        prometheus_output.append("# HELP codesage_mcp_errors_total Total number of errors")
        prometheus_output.append("# TYPE codesage_mcp_errors_total counter")
        prometheus_output.append(f"codesage_mcp_errors_total {performance_monitor.error_count}")

        # Add response time metrics
        response_time = current_metrics.get("response_time_ms", {}).get("value")
        if response_time is not None:
            prometheus_output.append("# HELP codesage_mcp_response_time_ms Current response time in milliseconds")
            prometheus_output.append("# TYPE codesage_mcp_response_time_ms gauge")
            prometheus_output.append(f"codesage_mcp_response_time_ms {response_time}")

        # Add throughput metric
        throughput = current_metrics.get("throughput_rps", {}).get("value")
        if throughput is not None:
            prometheus_output.append("# HELP codesage_mcp_requests_per_second Current requests per second")
            prometheus_output.append("# TYPE codesage_mcp_requests_per_second gauge")
            prometheus_output.append(f"codesage_mcp_requests_per_second {throughput}")

        # Add resource utilization metrics
        memory_percent = current_metrics.get("memory_usage_percent", {}).get("value")
        if memory_percent is not None:
            prometheus_output.append("# HELP codesage_mcp_memory_usage_percent Current memory usage percentage")
            prometheus_output.append("# TYPE codesage_mcp_memory_usage_percent gauge")
            prometheus_output.append(f"codesage_mcp_memory_usage_percent {memory_percent}")

        cpu_percent = current_metrics.get("cpu_usage_percent", {}).get("value")
        if cpu_percent is not None:
            prometheus_output.append("# HELP codesage_mcp_cpu_usage_percent Current CPU usage percentage")
            prometheus_output.append("# TYPE codesage_mcp_cpu_usage_percent gauge")
            prometheus_output.append(f"codesage_mcp_cpu_usage_percent {cpu_percent}")

        # Add error rate metric
        error_rate = current_metrics.get("error_rate_percent", {}).get("value")
        if error_rate is not None:
            prometheus_output.append("# HELP codesage_mcp_error_rate_percent Current error rate percentage")
            prometheus_output.append("# TYPE codesage_mcp_error_rate_percent gauge")
            prometheus_output.append(f"codesage_mcp_error_rate_percent {error_rate}")

        # Add performance score
        performance_score = report.get("performance_score", 0)
        prometheus_output.append("# HELP codesage_mcp_performance_score Overall performance score (0-100)")
        prometheus_output.append("# TYPE codesage_mcp_performance_score gauge")
        prometheus_output.append(f"codesage_mcp_performance_score {performance_score}")

        # Add alert count
        alerts = report.get("recent_alerts", [])
        prometheus_output.append("# HELP codesage_mcp_active_alerts Number of active alerts")
        prometheus_output.append("# TYPE codesage_mcp_active_alerts gauge")
        prometheus_output.append(f"codesage_mcp_active_alerts {len(alerts)}")

        # Add connection pool metrics
        try:
            # Access the LLM analysis manager from the codebase manager
            llm_manager = get_llm_analysis_manager()
            conn_stats = llm_manager.get_connection_pool_stats()

            # Connection pool configuration
            prometheus_output.append("# HELP codesage_mcp_connection_pool_max_connections Maximum connections in pool")
            prometheus_output.append("# TYPE codesage_mcp_connection_pool_max_connections gauge")
            prometheus_output.append(f"codesage_mcp_connection_pool_max_connections {conn_stats['connection_pool']['max_connections']}")

            prometheus_output.append("# HELP codesage_mcp_connection_pool_timeout_seconds Request timeout in seconds")
            prometheus_output.append("# TYPE codesage_mcp_connection_pool_timeout_seconds gauge")
            prometheus_output.append(f"codesage_mcp_connection_pool_timeout_seconds {conn_stats['connection_pool']['timeout_seconds']}")

            # Provider-specific metrics
            for provider, stats in conn_stats['providers'].items():
                prometheus_output.append(f"# HELP codesage_mcp_{provider}_requests_total Total requests to {provider}")
                prometheus_output.append(f"# TYPE codesage_mcp_{provider}_requests_total counter")
                prometheus_output.append(f"codesage_mcp_{provider}_requests_total {stats['total_requests']}")

                prometheus_output.append(f"# HELP codesage_mcp_{provider}_requests_per_minute Current requests per minute to {provider}")
                prometheus_output.append(f"# TYPE codesage_mcp_{provider}_requests_per_minute gauge")
                prometheus_output.append(f"codesage_mcp_{provider}_requests_per_minute {stats['requests_per_minute']}")

                prometheus_output.append(f"# HELP codesage_mcp_{provider}_active_connections Active connections to {provider}")
                prometheus_output.append(f"# TYPE codesage_mcp_{provider}_active_connections gauge")
                prometheus_output.append(f"codesage_mcp_{provider}_active_connections {stats['active_connections']}")

                # Rate limiting metrics
                rate_stats = conn_stats['rate_limiting'][provider]
                prometheus_output.append(f"# HELP codesage_mcp_{provider}_rate_limited Whether {provider} is currently rate limited")
                prometheus_output.append(f"# TYPE codesage_mcp_{provider}_rate_limited gauge")
                prometheus_output.append(f"codesage_mcp_{provider}_rate_limited {1 if rate_stats['is_rate_limited'] else 0}")

        except Exception as e:
            logger.warning(f"Failed to collect connection pool metrics: {e}")

        # Return response with correct content type
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("\n".join(prometheus_output) + "\n", media_type="text/plain; version=0.0.4; charset=utf-8")

    except Exception as e:
        log_exception(e, logger)
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(f"# Error generating metrics: {str(e)}\n", media_type="text/plain; version=0.0.4; charset=utf-8")


@log_errors(request_id_param="id")
@app.post("/mcp")
async def handle_jsonrpc_request(request: Request):
    """
    Handles JSON-RPC requests, including tool discovery and notifications.
    Includes performance monitoring, usage pattern tracking, and Gemini CLI compatibility.
    """
    start_time = time.time()
    success = True
    tool_name = None
    user_id = "anonymous"  # Default user ID, could be extracted from headers

    try:
        body = await request.json()
        request_body = body
        request_headers = dict(request.headers)
        jsonrpc_request = JSONRPCRequest(**body)

        if jsonrpc_request.method == "initialize":
            response_result = {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": "CodeSage MCP Server", "version": "0.1.0"},
                "capabilities": {
                    "tools": get_all_tools_definitions_as_object(),
                    "prompts": {
                        "listChanged": False  # Server does not support dynamic prompt changes
                    }
                },
            }
            # Return raw dictionary to avoid Pydantic serialization issues
            compatibility_handler = get_compatibility_handler()
            response_data = compatibility_handler.create_compatible_response(
                result=response_result,
                request_id=jsonrpc_request.id,
                request_headers=request_headers,
                request_body=request_body
            )
            return GeminiCompatibleJSONResponse(content=response_data)

        elif jsonrpc_request.method == "notifications/initialized":
            logger.info(
                "Received 'notifications/initialized' notification. Acknowledging."
            )
            # Return raw dictionary to avoid Pydantic serialization issues
            compatibility_handler = get_compatibility_handler()
            response_data = compatibility_handler.create_compatible_response(
                result=None,
                request_id=jsonrpc_request.id,
                request_headers=request_headers,
                request_body=request_body
            )
            return GeminiCompatibleJSONResponse(content=response_data)

        elif jsonrpc_request.method == "tools/list":
            # Get tools in standard object format
            tools_object = get_all_tools_definitions_as_object()
            logger.debug(f"Retrieved {len(tools_object)} tools for listing")

            # Use compatibility handler to detect and adapt format
            compatibility_handler = get_compatibility_handler()
            detected_format = compatibility_handler.detect_response_format(
                request_headers, request_body
            )
            logger.debug(f"Detected response format: {detected_format}")

            adapted_tools = compatibility_handler.adapt_tools_response(
                tools_object,
                detected_format
            )
            logger.debug(f"Adapted tools response type: {type(adapted_tools)}")

            # Return raw dictionary to avoid Pydantic serialization issues
            response_data = compatibility_handler.create_compatible_response(
                result=adapted_tools,
                request_id=jsonrpc_request.id,
                request_headers=request_headers,
                request_body=request_body
            )
            return GeminiCompatibleJSONResponse(content=response_data)

        elif jsonrpc_request.method == "prompts/list":
            # Return empty prompts list since this server focuses on tools
            prompts_result = {"prompts": []}
            logger.debug("Returning empty prompts list")

            # Return raw dictionary to avoid Pydantic serialization issues
            compatibility_handler = get_compatibility_handler()
            response_data = compatibility_handler.create_compatible_response(
                result=prompts_result,
                request_id=jsonrpc_request.id,
                request_headers=request_headers,
                request_body=request_body
            )
            return GeminiCompatibleJSONResponse(content=response_data)

        elif jsonrpc_request.method == "tools/call":
            if not jsonrpc_request.params or not isinstance(
                jsonrpc_request.params, dict
            ):
                error_response = create_gemini_compatible_error_response(
                    "INVALID_PARAMS", "Invalid params for tools/call."
                )
                # Return raw dictionary to avoid Pydantic serialization issues
                compatibility_handler = get_compatibility_handler()
                response_data = compatibility_handler.create_compatible_response(
                    error=error_response,
                    request_id=jsonrpc_request.id,
                    request_headers=request_headers,
                    request_body=request_body
                )
                return GeminiCompatibleJSONResponse(content=response_data)

            tool_name = jsonrpc_request.params.get("name")
            tool_args = jsonrpc_request.params.get("arguments", {})

            if tool_name not in TOOL_FUNCTIONS:
                error_response = create_gemini_compatible_error_response(
                    "TOOL_NOT_FOUND", f"Tool not found: {tool_name}"
                )
                # Return raw dictionary to avoid Pydantic serialization issues
                compatibility_handler = get_compatibility_handler()
                response_data = compatibility_handler.create_compatible_response(
                    error=error_response,
                    request_id=jsonrpc_request.id,
                    request_headers=request_headers,
                    request_body=request_body
                )
                return GeminiCompatibleJSONResponse(content=response_data)

            tool_function = TOOL_FUNCTIONS[tool_name]
            try:
                # Record user action for pattern analysis
                usage_analyzer.record_user_action(
                    user_id=user_id,
                    action=tool_name,
                    metadata={
                        "tool_args": tool_args,
                        "request_id": str(jsonrpc_request.id)
                    }
                )

                tool_result = tool_function(**tool_args)
                # Return raw dictionary to avoid Pydantic serialization issues
                compatibility_handler = get_compatibility_handler()
                response_data = compatibility_handler.create_compatible_response(
                    result=tool_result,
                    request_id=jsonrpc_request.id,
                    request_headers=request_headers,
                    request_body=request_body
                )
                return GeminiCompatibleJSONResponse(content=response_data)
            except Exception as e:
                success = False
                log_exception(e, logger, request_id=str(jsonrpc_request.id), extra_context={"tool_name": tool_name})
                error_response = create_gemini_compatible_error_response(
                    "TOOL_EXECUTION_ERROR", f"Error executing tool {tool_name}: {e}"
                )
                # Return raw dictionary to avoid Pydantic serialization issues
                compatibility_handler = get_compatibility_handler()
                response_data = compatibility_handler.create_compatible_response(
                    error=error_response,
                    request_id=jsonrpc_request.id,
                    request_headers=request_headers,
                    request_body=request_body
                )
                return GeminiCompatibleJSONResponse(content=response_data)
        else:
            error_response = create_gemini_compatible_error_response(
                "UNKNOWN_METHOD", f"Unknown JSON-RPC method: {jsonrpc_request.method}"
            )
            # Return raw dictionary to avoid Pydantic serialization issues
            compatibility_handler = get_compatibility_handler()
            return compatibility_handler.create_compatible_response(
                error=error_response,
                request_id=jsonrpc_request.id,
                request_headers=request_headers,
                request_body=request_body
            )
    except (json.JSONDecodeError, ValidationError) as e:
        success = False
        # request_body may not be defined if JSON parsing failed
        request_id = None
        if 'request_body' in locals() and isinstance(request_body, dict):
            request_id = request_body.get('id')
        log_exception(e, logger, request_id=request_id)
        error_response = create_gemini_compatible_error_response(
            "INVALID_REQUEST", f"Invalid JSON-RPC request: {e}"
        )
        # Return raw dictionary to avoid Pydantic serialization issues
        compatibility_handler = get_compatibility_handler()
        response_data = compatibility_handler.create_compatible_response(
            error=error_response,
            request_id=request_id,
            request_headers=request_headers if 'request_headers' in locals() else {},
            request_body=request_body if 'request_body' in locals() else None
        )
        return GeminiCompatibleJSONResponse(content=response_data)
    finally:
        # Record performance metrics
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        performance_monitor.record_request(
            response_time_ms=response_time_ms,
            success=success,
            endpoint=f"/mcp",
            user_id=user_id
        )

        # Record metrics for trend analysis
        trend_analyzer.record_metric("response_time_ms", response_time_ms, end_time)
        trend_analyzer.record_metric("request_success", 1.0 if success else 0.0, end_time)
        if tool_name:
            trend_analyzer.record_metric(f"tool_usage_{tool_name}", 1.0, end_time)

        # Collect implicit feedback based on performance and success
        if tool_name:
            feedback_collector.collect_implicit_feedback(
                user_id=user_id,
                action=tool_name,
                response_time_ms=response_time_ms,
                success=success,
                metadata={
                    "request_id": str(jsonrpc_request.id) if 'jsonrpc_request' in locals() and jsonrpc_request else None,
                    "endpoint": "/mcp",
                    "tool_args": tool_args if 'tool_args' in locals() else None
                }
            )

        # Log performance data for monitoring
        logger.info(
            "Request processed",
            tool_name=tool_name,
            response_time_ms=round(response_time_ms, 2),
            success=success,
            user_id=user_id
        )

# Project: CodeSage MCP Server

## Executive Summary
The CodeSage Model Context Protocol (MCP) Server is a high-performance, production-ready platform designed to revolutionize code analysis and search capabilities. It functions as an intermediary, enabling the Qwen CLI to interact with larger codebases and integrate with various Large Language Models (LLMs) for specialized tasks. It boasts exceptional performance metrics in indexing speed, search response, memory usage, and cache hit rate.

## Definitions
*   **MCP:** Model Context Protocol
*   **LLM:** Large Language Model
*   **FAISS:** Facebook AI Similarity Search
*   **ASGI:** Asynchronous Server Gateway Interface
*   **CLI:** Command Line Interface

## Project Overview
The server is a FastAPI application that exposes a JSON-RPC endpoint (`/mcp`). This endpoint handles requests for tool discovery (`initialize`, `tools/list`) and tool execution (`tools/call`). It integrates with a `CodebaseManager` for file system operations, codebase indexing, and LLM integration, enabling advanced features such as intelligent codebase indexing, semantic search, duplicate code detection, smart code summarization, memory optimization, multi-strategy caching, incremental indexing, parallel processing, index compression, adaptive cache sizing, smart prefetching, usage pattern learning, comprehensive monitoring, and enterprise security.

**Key Technologies:**
*   **Python:** The primary programming language.
*   **FastAPI:** Used for building the web API that handles JSON-RPC requests.
*   **Uvicorn:** An ASGI server that runs the FastAPI application.
*   **FAISS:** For vector similarity search.
*   **Sentence Transformers:** For semantic understanding and embeddings.
*   **psutil:** For memory management and monitoring.
*   **python-dotenv:** For managing environment variables, especially for API keys.
*   **groq:** The Python client for the Groq LLM API.
*   **openai:** The official OpenAI Python client, used to interact with the OpenRouter API.
*   **google-generativeai:** The Python client for Google AI models.
*   **pytest:** For running the test suite.
*   **Docker:** For containerization and easy deployment.

## Building and Running

### Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd GeminiMCPs
    ```
2.  **Install dependencies into the virtual environment:**
    ```bash
    python3 -m venv venv # On Windows: python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate; On macOS: source venv/bin/activate
    pip install -r requirements.txt
    ```
    For performance optimization, it's recommended to run:
    ```bash
    pip install --upgrade pip
    pip install --no-cache-dir -r requirements.txt
    ```

### Running the Server
You can run the CodeSage MCP Server either directly using `uvicorn` or via Docker Compose.

#### Running Directly (using uvicorn)
To start the CodeSage MCP Server in development mode:
```bash
uvicorn codesage_mcp.main:app --host 127.0.0.1 --port 8000 --reload
```
Alternatively, you can use the `start_server.sh` script:
```bash
./start_server.sh # On Windows, you might need to use `bash start_server.sh` if Git Bash is installed.
```

#### Running with Docker Compose (Recommended for Production)
Ensure you have Docker and Docker Compose installed. From the project root directory, run:
```bash
docker compose up -d
```
This will build the Docker image (if not already built) and start the server in a container. The server will be accessible at `http://localhost:8000`.

### Environment Variables
Before running the server, you need to create a `.env` file in the project root. You can copy the `.env.example` file to get started and then edit it to add your API keys and performance settings:

```bash
cp .env.example .env # On Windows, you can use `copy .env.example .env`
```

Example `.env` content:
```bash
# LLM API Keys (choose your preferred provider)
GROQ_API_KEY="gsk_..."
OPENROUTER_API_KEY="sk-or-..."
GOOGLE_API_KEY="AIza..."

# Performance Tuning
CODESAGE_MEMORY_LIMIT=512MB
CODESAGE_CACHE_SIZE=1GB
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_PARALLEL_WORKERS=4

# Production Settings
CODESAGE_LOG_LEVEL=INFO
CODESAGE_MONITORING_ENABLED=true
CODESAGE_METRICS_PORT=9090
```

### Configuring Qwen CLI
To enable the Qwen CLI to utilize the CodeSage MCP Server, you need to add its configuration to your Qwen CLI `settings.json` file. This file is typically located at `~/.config/qwen-cli/settings.json` on Linux/macOS or `%APPDATA%\\qwen-cli\\settings.json` on Windows.

Add the following entry to the `mcpServers` array in your `settings.json`:
```json
{
  "mcpServers": [
    {
      "name": "codesage",
      "httpUrl": "http://127.0.0.1:8000/mcp",
      "trust": true
    }
  ]
}
```
After adding this configuration, restart your Qwen CLI session for the changes to take effect. You should then be able to discover and use the tools exposed by the CodeSage MCP Server.

## Development Conventions

**Tooling:**
The project leverages the key technologies listed in the "Project Overview" section, along with `ruff` for linting and formatting.

**Pre-commit Hooks:**
The project uses pre-commit hooks to enforce code quality and consistency. The hooks are defined in the `.pre-commit-config.yaml` file and include checks for large files, trailing whitespace, private keys, `ruff` for linting and formatting, a custom hook to automatically generate the `docs/tools_reference.md` file, and a custom hook to run the `pytest` test suite.

To use the pre-commit hooks, you need to install `pre-commit` and then run `pre-commit install` in the project root.

**Docstring Convention:**
The project follows the Google docstring convention. More details can be found in `docs/docstring_standard.md`.

**Code Structure:**
*   **`codesage_mcp/main.py`**: Contains the main FastAPI application logic, handling JSON-RPC requests and routing them to the appropriate tool functions.
*   **`codesage_mcp/tools/`**: Contains individual tool functions that the MCP server exposes to the Qwen CLI, organized into multiple files.
*   **`codesage_mcp/core/`**: Contains core structural components like exception definitions, logging, and data models.
*   **`codesage_mcp/features/`**: Contains distinct features like caching, memory management, and performance monitoring.
*   **`codesage_mcp/config.py`**: Handles loading API keys for various LLMs from environment variables.
*   **`tests/`**: Contains the test suite for the project.

## Current Tools Implemented (Detailed List)

This section provides a detailed list of all the tools currently exposed by the CodeSage MCP Server, organized by their functional categories within the `codesage_mcp/tools/` directory.

### 1. Codebase Analysis Tools (`codebase_analysis.py`)

These tools form the foundation for interacting with and understanding the codebase.

*   **`read_code_file_tool(file_path: str)`**
    *   **Description:** Reads and returns the content of a specified code file.
*   **`search_codebase_tool(...)`** *(Enhanced)*
    *   **Description:** Searches for a pattern within indexed code files with multiple modes (regex, semantic, graph) and enhanced results including dependencies.
    *   **Parameters:**
        *   `codebase_path` (str)
        *   `pattern` (str)
        *   `file_types` (list[str], optional)
        *   `exclude_patterns` (list[str], optional)
        *   `search_mode` (str): "regex", "semantic", or "graph"
        *   `context_depth` (int): Depth of related code (1-3)
        *   `include_dependencies` (bool): Include dependency info
*   **`get_file_structure_tool(codebase_path: str, file_path: str)`**
    *   **Description:** Provides a high-level overview of a file's structure.
*   **`index_codebase_tool(path: str)`**
    *   **Description:** Indexes a given codebase path for analysis.
*   **`list_undocumented_functions_tool(file_path: str)`**
    *   **Description:** Identifies and lists Python functions in a specified file that are missing docstrings.
*   **`count_lines_of_code_tool()`**
    *   **Description:** Counts lines of code (LOC) in the indexed codebase.
*   **`get_dependencies_overview_tool(codebase_path: str)`**
    *   **Description:** Analyzes Python files in the indexed codebase and extracts import statements.
*   **`find_duplicate_code_tool(codebase_path: str, min_similarity: float, min_lines: int)`**
    *   **Description:** Finds duplicate code sections within the indexed codebase.
*   **`analyze_codebase_improvements_tool(codebase_path: str)`**
    *   **Description:** Analyzes the codebase for potential improvements and suggestions (TODOs, FIXMEs, undocumented functions, large files, etc.).

### 2. LLM Analysis Tools (`llm_analysis.py`)

These tools leverage LLMs for deeper code understanding and processing.

*   **`summarize_code_section_tool(...)`** *(Enhanced)*
    *   **Description:** Summarizes a specific section of code using a chosen LLM, with performance insights and dependency analysis.
    *   **Parameters:**
        *   `file_path` (str)
        *   `start_line` (int, optional)
        *   `end_line` (int, optional)
        *   `llm_model` (str, optional)
        *   `function_name` (str, optional)
        *   `class_name` (str, optional)
        *   `include_performance_insights` (bool)
        *   `include_dependency_analysis` (bool)
*   **`semantic_search_codebase_tool(codebase_path: str, query: str, top_k: int)`**
    *   **Description:** Performs a semantic search within the indexed codebase to find code snippets semantically similar to the given query.
*   **`profile_code_performance_tool(file_path: str, function_name: str, llm_model: str)`**
    *   **Description:** Profiles the performance of a specific function or the entire file.
*   **`suggest_code_improvements_tool(file_path: str, start_line: int, end_line: int, llm_model: str)`**
    *   **Description:** Analyzes a code section and suggests improvements by consulting external LLMs.

### 3. Code Generation Tools (`code_generation.py`)

These tools assist in generating code, tests, and documentation.

*   **`generate_unit_tests_tool(file_path: str, function_name: str)`**
    *   **Description:** Generates unit tests for functions in a Python file.
*   **`auto_document_tool(tool_name: str)`**
    *   **Description:** Automatically generates documentation for tools that lack detailed documentation.
*   **`resolve_todo_fixme_tool(file_path: str, line_number: int)`**
    *   **Description:** Analyzes a TODO/FIXME comment and suggests potential resolutions using LLMs.
*   **`parse_llm_response_tool(llm_response_content: str)`**
    *   **Description:** Parses the content of an LLM response, extracting and validating JSON data.
*   **`generate_llm_api_wrapper_tool(...)`**
    *   **Description:** Generates Python wrapper code for interacting with various LLM APIs.
    *   **Parameters:**
        *   `llm_provider` (str)
        *   `model_name` (str)
        *   `api_key_env_var` (str, optional)
        *   `output_file_path` (str, optional)
*   **`generate_boilerplate_tool(...)`**
    *   **Description:** Generates standardized boilerplate code for new modules, tools, or tests.
    *   **Parameters:**
        *   `boilerplate_type` (str): "file_header", "module", "tool", "test", "class", "function"
        *   `file_path` (str, optional)
        *   `module_name` (str, optional)
        *   `function_name` (str, optional)
        *   `class_name` (str, optional)

### 4. Configuration Tools (`configuration.py`)

These tools manage the server's configuration and API keys.

*   **`configure_api_key_tool(llm_provider: str, api_key: str)`**
    *   **Description:** Configures API keys for LLMs (Groq, OpenRouter, Google AI).
*   **`get_configuration_tool()`**
    *   **Description:** Returns the current configuration, with API keys masked for security.
*   **`get_cache_statistics_tool()`**
    *   **Description:** Returns comprehensive statistics about the intelligent caching system.

### 5. Performance Monitoring Tools (`performance_monitoring.py`)

These tools provide insights into the server's performance and resource usage.

*   **`get_performance_metrics_tool()`**
    *   **Description:** Get current real-time performance metrics (response times, resource utilization, throughput, etc.).
*   **`get_performance_report_tool()`**
    *   **Description:** Generate a comprehensive performance report including current metrics, baselines, alerts, and recommendations.
*   **`get_usage_patterns_tool()`**
    *   **Description:** Analyze and return usage patterns across different user profiles.
*   **`get_predictive_analytics_tool()`**
    *   **Description:** Get predictive analytics for performance optimization (resource forecasting, anomaly detection, recommendations).
*   **`detect_performance_regressions_tool(current_results: Dict)`**
    *   **Description:** Detect performance regressions by comparing current results against baseline.

### 6. User Feedback Tools (`user_feedback_tools.py`)

These tools manage user feedback for continuous improvement.

*   **`collect_user_feedback_tool(...)`**
    *   **Description:** Collect user feedback for analysis and improvement.
    *   **Parameters:**
        *   `feedback_type` (str): "bug_report", "feature_request", etc.
        *   `title` (str)
        *   `description` (str)
        *   `satisfaction_level` (int, optional)
        *   `user_id` (str)
        *   `metadata` (Dict, optional)
*   **`get_feedback_summary_tool(user_id: str, feedback_type: str)`**
    *   **Description:** Get a summary of user feedback data.
*   **`get_user_insights_tool(user_id: str)`**
    *   **Description:** Get insights about a specific user's behavior and satisfaction.
*   **`analyze_feedback_patterns_tool()`**
    *   **Description:** Analyze patterns in user feedback to identify trends and improvement opportunities.
*   **`get_feedback_driven_recommendations_tool()`**
    *   **Description:** Generate recommendations based on user feedback analysis.
*   **`get_user_satisfaction_metrics_tool()`**
    *   **Description:** Get comprehensive user satisfaction metrics and trends.

### 7. Trend Analysis Tools (`trend_analysis_tools.py`)

These tools analyze performance trends and identify optimization opportunities.

*   **`analyze_performance_trends_tool(metric_name: str, analysis_window_days: int)`**
    *   **Description:** Analyze performance trends for specific metrics or all metrics.
*   **`get_optimization_opportunities_tool()`**
    *   **Description:** Identify and prioritize optimization opportunities based on performance trends.
*   **`predict_performance_capacity_tool(target_response_time_ms: int)`**
    *   **Description:** Predict maximum workload capacity for target performance levels.
*   **`forecast_performance_trends_tool(metric_name: str, forecast_days: int)`**
    *   **Description:** Forecast future performance trends using predictive analytics.
*   **`get_performance_baseline_comparison_tool()`**
    *   **Description:** Compare current performance against established baselines.

### 8. Cache Analysis Tools (`cache_analysis_tools.py`)

These tools analyze cache effectiveness and provide optimization recommendations.

*   **`analyze_cache_effectiveness_tool(cache_type: str)`**
    *   **Description:** Analyze cache effectiveness in real-world scenarios.
*   **`get_cache_optimization_recommendations_tool()`**
    *   **Description:** Get cache optimization recommendations based on effectiveness analysis.
*   **`get_cache_performance_metrics_tool(cache_type: str)`**
    *   **Description:** Get detailed cache performance metrics for monitoring and analysis.
*   **`get_cache_access_patterns_tool(cache_type: str)`**
    *   **Description:** Analyze cache access patterns to identify optimization opportunities.
*   **`get_cache_memory_efficiency_tool()`**
    *   **Description:** Analyze cache memory efficiency and provide optimization recommendations.

### 9. Memory Pattern Tools (`memory_pattern_tools.py`)

These tools monitor and manage memory usage patterns.

*   **`analyze_memory_patterns_tool(analysis_window_hours: int)`**
    *   **Description:** Analyze memory usage patterns under varying loads.
*   **`get_adaptive_memory_management_tool()`**
    *   **Description:** Get adaptive memory management recommendations based on current patterns.
*   **`optimize_memory_for_load_tool(load_level: str)`**
    *   **Description:** Optimize memory settings for a specific load level.
*   **`get_memory_pressure_analysis_tool()`**
    *   **Description:** Analyze current memory pressure and provide detailed insights.
*   **`get_memory_optimization_opportunities_tool()`**
    *   **Description:** Identify memory optimization opportunities based on usage patterns.

### 10. Adaptive Cache Tools (`adaptive_cache_tools.py`)

These tools manage dynamic cache sizing and adaptation.

*   **`get_adaptive_cache_status_tool()`**
    *   **Description:** Get the current status of adaptive cache management.
*   **`trigger_cache_adaptation_tool(cache_type: str, strategy: str)`**
    *   **Description:** Trigger manual cache adaptation for specified cache type or all caches.
*   **`get_cache_sizing_recommendations_tool(cache_type: str, strategy: str)`**
    *   **Description:** Get cache sizing recommendations for a specific cache type using different strategies.
*   **`analyze_cache_adaptation_effectiveness_tool(time_window_hours: int)`**
    *   **Description:** Analyze the effectiveness of cache adaptations over time.
*   **`get_cache_adaptation_rules_tool()`**
    *   **Description:** Get information about cache adaptation rules and their performance.

### 11. Workload Adaptive Memory Tools (`workload_adaptive_memory_tools.py`)

These tools adapt memory allocation based on workload.

*   **`get_workload_analysis_tool()`**
    *   **Description:** Get comprehensive workload analysis including current patterns and predictions.
*   **`get_memory_allocation_status_tool()`**
    *   **Description:** Get current memory allocation status and effectiveness analysis.
*   **`trigger_workload_adaptation_tool(strategy: str)`**
    *   **Description:** Trigger manual workload adaptation with specified strategy.
*   **`get_workload_optimization_recommendations_tool()`**
    *   **Description:** Get workload optimization recommendations based on analysis.
*   **`analyze_workload_performance_impact_tool(time_window_hours: int)`**
    *   **Description:** Analyze the performance impact of workload adaptations over time.
*   **`get_workload_adaptation_rules_tool()`**
    *   **Description:** Get information about workload adaptation rules and their effectiveness.

### 12. Intelligent Prefetch Tools (`intelligent_prefetch_tools.py`)

These tools manage intelligent prefetching of resources.

*   **`get_prefetch_analysis_tool()`**
    *   **Description:** Get comprehensive intelligent prefetching analysis.
*   **`trigger_prefetching_tool(strategy: str, max_candidates: int)`**
    *   **Description:** Trigger intelligent prefetching with specified strategy and parameters.
*   **`get_prefetch_performance_metrics_tool(time_window_hours: int)`**
    *   **Description:** Get detailed prefetching performance metrics over a specified time window.
*   **`analyze_prefetch_patterns_tool(pattern_type: str)`**
    *   **Description:** Analyze prefetching patterns and their effectiveness.
*   **`get_prefetch_configuration_tool()`**
    *   **Description:** Get current prefetching configuration and provide tuning recommendations.
*   **`update_prefetch_configuration_tool(config_updates: Dict)`**
    *   **Description:** Update prefetching configuration with new settings.

### 13. Auto Performance Tuning Tools (`auto_performance_tuning_tools.py`)

These tools automate performance tuning based on analysis.

*   **`get_performance_tuning_analysis_tool()`**
    *   **Description:** Get comprehensive automatic performance tuning analysis.
*   **`trigger_performance_tuning_tool(strategy: str, max_experiments: int)`**
    *   **Description:** Trigger automatic performance tuning with specified strategy and parameters.
*   **`get_tuning_recommendations_tool(confidence_threshold: float)`**
    *   **Description:** Get specific tuning recommendations based on current system state.
*   **`analyze_tuning_effectiveness_tool(time_window_hours: int)`**
    *   **Description:** Analyze the effectiveness of automatic performance tuning over time.
*   **`get_tuning_configuration_tool()`**
    *   **Description:** Get current automatic performance tuning configuration.
*   **`update_tuning_configuration_tool(config_updates: Dict)`**
    *   **Description:** Update automatic performance tuning configuration with new settings.

### 14. Workload Pattern Recognition Tools (`workload_pattern_recognition_tools.py`)

These tools recognize workload patterns and manage resources accordingly.

*   **`get_workload_pattern_analysis_tool()`**
    *   **Description:** Get comprehensive workload pattern analysis.
*   **`trigger_pattern_based_allocation_tool(pattern_type: str, resource_focus: str)`**
    *   **Description:** Trigger pattern-based resource allocation with specified parameters.
*   **`get_resource_allocation_status_tool()`**
    *   **Description:** Get current resource allocation status and effectiveness analysis.
*   **`forecast_workload_patterns_tool(time_horizon_hours: int)`**
    *   **Description:** Forecast workload patterns and resource needs.
*   **`analyze_pattern_effectiveness_tool(time_window_hours: int)`**
    *   **Description:** Analyze the effectiveness of workload pattern recognition.
*   **`get_pattern_recognition_configuration_tool()`**
    *   **Description:** Get current workload pattern recognition configuration.
*   **`update_pattern_recognition_configuration_tool(config_updates: Dict)`**
    *   **Description:** Update workload pattern recognition configuration with new settings.

### 15. Continuous Improvement Tools (`continuous_improvement.py`)

These tools drive continuous improvement through analysis and automation.

*   **`analyze_continuous_improvement_opportunities_tool()`**
    *   **Description:** Analyze production data to identify optimization opportunities.
*   **`implement_automated_improvements_tool(dry_run: bool)`**
    *   **Description:** Implement automated improvements based on analysis results.
*   **`monitor_improvement_effectiveness_tool(time_window_hours: int)`**
    *   **Description:** Monitor the effectiveness of implemented improvements.

## Development Workflow & Orchestration

This project follows a defined workflow as outlined in `AGENT_WORKFLOW.md`. Key aspects include:

*   **Iterative Deep Dive:** Understanding the codebase through initial scans, breadth-first exploration, and depth-first analysis.
*   **Extended Sequential Thinking:** Planning with pre-computation, dependency mapping, and error anticipation.
*   **Self-Verification Loops:** Running linters, type-checkers, and tests after modifications.
*   **Orchestration with External LLMs:** Delegating tasks to external LLMs with clear objectives and verifying their output.
*   **Git Workflow:** Making regular, descriptive commits. No direct pushes.
*   **Communication:** Concise, timely updates with explanations for critical commands.
*   **Consistency:** Adhering to project conventions, coding standards, and maintaining internal context.
*   **Modularization:** The project structure is being actively modularized as per `MODULARIZATION_PLAN.md` to separate structural base from features and tools.

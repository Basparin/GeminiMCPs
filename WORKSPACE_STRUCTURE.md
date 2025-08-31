## CodeSage MCP Server Workspace Structure

This document defines the required folder structure for the CodeSage MCP Server project, ensuring organization, consistency, and ease of navigation.

```
/project_root/
├── codesage_mcp/
│   ├── __init__.py             # Python package initializer
│   ├── main.py                 # Main FastAPI application entry point
│   ├── tools/                  # Directory for individual MCP tool implementations
│   │   ├── __init__.py
│   │   └── ... (e.g., read_code_file.py, index_codebase.py)
│   ├── config.py               # Application configuration settings
│   ├── codebase_manager.py     # Core logic for codebase interaction (indexing, searching)
│   ├── indexing.py             # Logic for codebase indexing (e.g., FAISS integration)
│   ├── caching.py              # Caching system implementation
│   ├── llm_analysis.py         # LLM integration and analysis logic
│   ├── memory_manager.py       # Memory optimization and management
│   └── ... (other core application modules)
├── tests/
│   ├── __init__.py
│   ├── test_main.py            # Tests for main application logic
│   ├── test_tools.py           # Tests for MCP tools
│   ├── test_codebase_manager.py# Tests for codebase manager
│   ├── benchmark_performance.py# Performance benchmark tests
│   └── ... (other test files)
├── docs/
│   ├── architecture.md         # High-level architectural overview
│   ├── GEMINI.md               # Project overview for Gemini CLI
│   ├── WORKSPACE_PLAN.md       # Strategic plan for the workspace
│   ├── AGENT_WORKFLOW.md       # Gemini agent's workflow documentation
│   ├── tools_reference.md      # Reference for all implemented MCP tools
│   ├── deployment_guide.md     # Instructions for deploying the server
│   ├── monitoring_guide.md     # Guide for setting up and using monitoring
│   └── WORKSPACE_STRUCTURE.md  # This document, defining the workspace structure
├── scripts/
│   ├── deploy.sh               # Shell scripts for deployment
│   ├── health_check.sh         # Scripts for health checks
│   ├── setup_monitoring.sh     # Scripts to set up monitoring components
│   └── ... (other utility scripts)
├── config/
│   ├── templates/              # Configuration file templates
│   │   └── ...
│   └── ... (other general configuration files)
├── monitoring/
│   ├── prometheus.yml          # Prometheus configuration
│   ├── grafana/                # Grafana dashboards and configurations
│   │   └── ...
│   └── ... (other monitoring-related files)
├── archive/                    # For old or deprecated files/documents
├── benchmark_results/          # Stores results from performance benchmarks
├── audit_results/              # Stores results from code audits
├── venv/                       # Python virtual environment (should be in .gitignore)
├── .git/                       # Git repository metadata (should be in .gitignore)
├── .codesage/                  # Internal CodeSage specific files (should be in .gitignore)
├── .ruff_cache/                # Ruff linter cache (should be in .gitignore)
├── requirements.txt            # Python dependency list
├── Dockerfile                  # Docker build instructions for the application
├── docker-compose.yml          # Docker Compose configuration for multi-service setup
├── pyproject.toml              # Project metadata and build system configuration
├── .gitignore                  # Specifies intentionally untracked files to ignore
├── .pre-commit-config.yaml     # Configuration for pre-commit hooks
├── .pydocstyle.ini             # Configuration for pydocstyle (docstring conventions)
├── .env.example                # Example file for environment variables
└── README.md                   # Project README file
```

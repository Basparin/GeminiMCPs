## CodeSage MCP Server Workspace Structure
*Last Updated: 2025-08-31*

### Executive Summary
[Brief summary of the document's purpose and key takeaways.]

### Definitions
[Glossary of key terms used in this document.]

This document defines the required folder structure for the CodeSage MCP Server project, ensuring organization, consistency, and ease of navigation.

```
**Legend:**
*   `*` : Optional element

/project_root/
├── codesage_mcp/
│   ├── __init__.py             # Python package initializer
│   ├── main.py                 # Main FastAPI application entry point
│   ├── config/                 # Configuration related files
│   │   ├── __init__.py
│   │   └── config.py
│   ├── core/                   # Proposed: Core structural components (currently at top-level codesage_mcp)
│   │   ├── __init__.py
│   │   └── ... (e.g., gemini_compatibility.py, exceptions.py, utils.py)
│   ├── features/               # Proposed: Directory for distinct features
│   │   ├── __init__.py
│   │   ├── caching/            # Proposed: Caching system feature
│   │   │   ├── __init__.py
│   │   │   └── ... (e.g., cache.py, adaptive_cache_manager.py)
│   │   ├── memory_management/  # Proposed: Memory management feature
│   │   │   ├── __init__.py
│   │   │   └── ...
│   │   ├── performance_monitoring/ # Proposed: Performance monitoring feature
│   │   │   ├── __init__.py
│   │   │   └── ...
│   │   ├── codebase_manager/   # Proposed: Codebase manager feature
│   │   │   ├── __init__.py
│   │   │   └── codebase_manager.py
│   │   └── llm_analysis/       # Proposed: LLM analysis feature
│   │       ├── __init__.py
│   │       └── llm_analysis.py
│   ├── tools/                  # Directory for individual MCP tool implementations (already exists)
│   │   ├── __init__.py
│   │   └── ... (e.g., codebase_analysis.py, llm_analysis.py, configuration.py)
│   └── ... (other top-level structural files that don't fit in 'core' or 'config')
├── tests/
│   ├── __init__.py
│   ├── structural_base/        # Proposed: Tests for the core structural components
│   │   ├── __init__.py
│   │   └── ...
│   ├── features/               # Proposed: Tests for distinct features
│   │   ├── __init__.py
│   │   ├── caching/
│   │   │   ├── __init__.py
│   │   │   └── ...
│   │   ├── memory_management/
│   │   │   ├── __init__.py
│   │   │   └── ...
│   │   ├── performance_monitoring/
│   │   │   ├── __init__.py
│   │   │   └── ...
│   │   ├── codebase_manager/
│   │   │   ├── __init__.py
│   │   │   └── ...
│   │   └── llm_analysis/
│   │       ├── __init__.py
│   │       └── ...
│   ├── tools/                  # Proposed: Tests for individual MCP tool implementations
│   │   ├── __init__.py
│   │   └── ...
│   └── ... (other top-level test files like conftest.py)
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

### Structure Validation
To ensure adherence to this workspace structure and facilitate automated maintenance, it is recommended to implement a script for structure validation. This script can check for:
*   Presence of required directories and files.
*   Correct naming conventions.
*   Absence of unauthorized files in specific locations.
```

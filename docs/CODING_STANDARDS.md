# CodeSage MCP Server Coding Standards and Guidelines

## Overview

This comprehensive document compiles all guidelines for workspace organization, modularity, naming conventions, dependencies, and code formatting policies for the CodeSage MCP Server. These standards ensure consistency, maintainability, and scalability across the codebase.

**Last Updated:** 2025-08-31  
**Version:** 1.0.0

## Table of Contents

1. [Workspace Organization](#workspace-organization)
2. [Modularity Guidelines](#modularity-guidelines)
3. [Naming Conventions](#naming-conventions)
4. [Dependencies](#dependencies)
5. [Code Formatting Policies](#code-formatting-policies)
6. [Migration Tips](#migration-tips)
7. [External Resources](#external-resources)
8. [LLM-Friendly Update Prompts](#llm-friendly-update-prompts)

## Workspace Organization

### Directory Structure

The CodeSage MCP Server follows a structured workspace organization to ensure consistency and ease of navigation:

```
project_root/
├── codesage_mcp/                 # Main Python package
│   ├── __init__.py               # Package initializer
│   ├── main.py                   # FastAPI application entry point
│   ├── tools/                    # MCP tool implementations
│   │   ├── __init__.py
│   │   └── *_tools.py            # Individual tool modules
│   ├── config.py                 # Configuration management
│   ├── exceptions.py             # Custom exceptions
│   └── [core modules...]         # Business logic modules
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_*.py                 # Unit and integration tests
│   └── conftest.py               # Test configuration
├── docs/                         # Documentation
│   ├── *.md                      # Markdown documentation files
│   └── CODING_STANDARDS.md       # This file
├── scripts/                      # Utility scripts
├── config/                       # Configuration files
├── monitoring/                   # Monitoring and logging setup
├── benchmark_results/            # Performance benchmark data
└── requirements.txt              # Python dependencies
```

### Key Principles

- **Separation of Concerns**: Clear boundaries between application logic, tests, documentation, and configuration
- **Scalability**: Structure supports growth without requiring major reorganizations
- **Tool Integration**: Compatible with development tools, CI/CD pipelines, and deployment systems

### Rationale

This structure promotes:
- Easy navigation for developers
- Clear separation between production code and supporting files
- Simplified deployment and testing processes
- Better integration with development tools and IDEs

## Modularity Guidelines

### Core Principles

1. **Single Responsibility Principle**: Each module should have one primary responsibility
2. **Interface Segregation**: Design client-specific interfaces rather than general-purpose ones
3. **Dependency Inversion**: High-level modules should not depend on low-level modules

### Module Size Limits

- Core modules: ≤ 500 lines
- Utility modules: ≤ 300 lines
- Tool modules: ≤ 200 lines
- Test modules: ≤ 400 lines

### Import Hierarchy

```
Infrastructure Layer (Bottom)
├── config.py, exceptions.py, utils.py
├── logging_config.py, prometheus_client.py

Core Services Layer
├── cache.py, memory_manager.py
├── chunking.py, performance_monitor.py

Business Logic Layer
├── indexing.py, searching.py, llm_analysis.py
├── codebase_manager.py, adaptive_cache_manager.py

Interface Layer (Top)
├── tools/*.py, main.py
```

### Dependency Rules

**Allowed Import Directions:**
- Infrastructure → Infrastructure (limited)
- Core Services → Infrastructure only
- Business Logic → Core Services + Infrastructure
- Interface → Business Logic + Core Services + Infrastructure

**Prohibited:**
- Business logic importing from main.py
- Core modules importing from tools/
- Any module importing from main.py

### Circular Dependency Prevention

**Detection Tools:**
```bash
python -m py_compile codesage_mcp/*.py
python -c "import codesage_mcp; print('No circular imports detected')"
```

**Prevention Strategies:**
- Dependency injection instead of direct imports
- Event-driven architecture for cross-module communication
- Plugin architecture for dynamic loading

### Example: Dependency Injection

**Before (Problematic):**
```python
class IndexingManager:
    def __init__(self):
        self.cache = get_cache_instance()
```

**After (Fixed):**
```python
class IndexingManager:
    def __init__(self, cache: CacheInterface):
        self.cache = cache
```

## Naming Conventions

### Files and Directories

- **Python Files**: `snake_case` (e.g., `cache.py`, `memory_manager.py`)
- **Directories**: `snake_case` (e.g., `codesage_mcp/`, `benchmark_results/`)
- **Configuration Files**: Lowercase with underscores or hyphens (e.g., `pyproject.toml`)

### Classes

- **Convention**: `CamelCase` (PascalCase)
- **Examples**: `IntelligentCache`, `PerformanceMonitor`, `AdaptiveCacheManager`
- **Exceptions**: `CamelCase` ending with "Error" (e.g., `BaseMCPError`, `ToolExecutionError`)

### Functions and Methods

- **Convention**: `snake_case`
- **Examples**: `get_embedding()`, `store_embedding()`, `analyze_cache_effectiveness()`
- **Private Methods**: Prefixed with single underscore (e.g., `_load_persistent_cache()`)

### Variables and Constants

- **Variables**: `snake_case` (e.g., `max_size`, `cache_dir`)
- **Constants**: `UPPER_CASE` (e.g., `CHUNK_SIZE_TOKENS`, `DEFAULT_TIMEOUT`)

### Test Files

- **Files**: `test_*.py` (e.g., `test_cache.py`, `test_performance_benchmarks.py`)
- **Functions**: `test_*` (e.g., `test_get_embedding_hit()`)
- **Classes**: `Test*` (e.g., `TestIntelligentCache`)

### Acronyms

- Treat as regular words: `get_embedding()` (not `getEmbedding()`)
- Maintain consistency: `LLMAnalysisManager`, `JSONRPCRequest`

## Dependencies

### Core Dependencies

Based on `pyproject.toml`:

```toml
dependencies = [
    "fastapi",           # Web framework
    "uvicorn",           # ASGI server
    "python-dotenv",     # Environment variable management
    "groq",              # Groq API client
    "openai",            # OpenAI API client
    "google-generativeai", # Google Gemini API
    "sentence-transformers", # Text embeddings
    "faiss-cpu",         # Vector similarity search
    "python-multipart",  # File upload support
    "radon",             # Code complexity analysis
    "stdlib-list",       # Standard library detection
]
```

### Development Dependencies

```toml
dev = [
    "pytest",            # Testing framework
    "ruff",              # Linter and formatter
]
```

### Dependency Management Principles

1. **Minimal Dependencies**: Only include necessary packages
2. **Version Pinning**: Use specific versions for production stability
3. **Security Updates**: Regularly update dependencies for security patches
4. **Compatibility**: Ensure dependencies work together and with target Python versions

### Adding New Dependencies

1. Evaluate necessity and alternatives
2. Check license compatibility
3. Update `pyproject.toml`
4. Test integration thoroughly
5. Update documentation

## Code Formatting Policies

### Linting and Formatting

**Primary Tool:** Ruff (configured in `pyproject.toml`)

```toml
[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F"]  # pycodestyle and pyflakes rules
ignore = ["E501"]    # Ignore line length (handled by formatter)
```

### Docstring Standard

**Format:** Google-style docstrings

**Required Sections:**
- Brief description (one line)
- Detailed description (if needed)
- Args: Parameter documentation
- Returns: Return value documentation
- Raises: Exception documentation
- Examples: Usage examples (optional)
- Note: Additional information (optional)

**Example:**
```python
def get_embedding(file_path: str, content: str) -> Optional[np.ndarray]:
    """Retrieves cached embedding for a file.

    Args:
        file_path (str): Path to the file.
        content (str): File content for hash generation.

    Returns:
        Optional[np.ndarray]: Cached embedding if found, None otherwise.

    Raises:
        CacheError: If cache access fails.
    """
```

### Import Organization

```python
# 1. Standard library imports
import os
import sys
from typing import Dict, List, Optional

# 2. Third-party imports
import numpy as np
import faiss
from fastapi import FastAPI

# 3. Local imports (grouped by package)
from .config import get_required_env_var
from .exceptions import BaseMCPError
from .utils import safe_read_file
```

### Code Style Principles

1. **PEP 8 Compliance**: Follow Python Enhancement Proposal 8
2. **Line Length**: 88 characters (Black/Ruff default)
3. **Quotes**: Use double quotes for strings, single for character constants
4. **Trailing Commas**: Include in multi-line structures
5. **Blank Lines**: Use strategically for readability

## Migration Tips

### Phase 1: Assessment (Week 1-2)

1. **Audit Current Codebase**
   ```bash
   # Check for undocumented functions
   python -c "from codesage_mcp.tools.codebase_analysis import list_undocumented_functions_tool; print('Audit complete')"
   
   # Analyze import dependencies
   python -m py_compile codesage_mcp/*.py
   ```

2. **Identify Priority Areas**
   - Large modules (>500 lines)
   - Complex import chains
   - Inconsistent naming
   - Missing documentation

### Phase 2: Refactoring (Week 3-6)

1. **Break Large Modules**
   - Extract interfaces and protocols
   - Split responsibilities into smaller modules
   - Implement dependency injection

2. **Standardize Naming**
   - Update class names to CamelCase
   - Convert functions to snake_case
   - Fix constant naming

3. **Add Documentation**
   - Implement Google-style docstrings
   - Document all public APIs
   - Add module-level documentation

### Phase 3: Enforcement (Week 7-8)

1. **Configure Tools**
   ```yaml
   # .pre-commit-config.yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.1.0
       hooks:
         - id: ruff
         - id: ruff-format
   ```

2. **CI/CD Integration**
   ```yaml
   # .github/workflows/ci.yml
   - name: Lint
     run: ruff check .
   - name: Format
     run: ruff format --check .
   ```

### Phase 4: Monitoring (Ongoing)

1. **Regular Audits**: Monthly codebase reviews
2. **Automated Checks**: Pre-commit hooks and CI/CD
3. **Team Training**: Ensure all developers understand standards

## External Resources

### Python Standards
- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

### Tools and Libraries
- [Ruff - Fast Python Linter](https://github.com/astral-sh/ruff)
- [Black - Code Formatter](https://black.readthedocs.io/)
- [Pydocstyle - Docstring Linter](https://pydocstyle.readthedocs.io/)
- [MyPy - Static Type Checker](https://mypy.readthedocs.io/)

### Best Practices
- [The Hitchhiker's Guide to Python](https://docs.python-guide.org/)
- [Real Python Tutorials](https://realpython.com/)
- [Python Code Quality](https://pycodequ.al/)

## LLM-Friendly Update Prompts

### For Adding New Guidelines

```
You are updating the CodeSage MCP Server coding standards. A new requirement has emerged: [describe requirement].

Analyze the current standards in CODING_STANDARDS.md and determine:
1. Which section(s) need updates
2. What new guidelines should be added
3. How this fits with existing standards
4. Any conflicts or adjustments needed

Provide specific recommendations with:
- Exact text additions/modifications
- Rationale for changes
- Examples of implementation
- Migration considerations

Ensure the update maintains consistency with existing standards and improves overall code quality.
```

### For Reviewing Code Changes

```
Review this code change against the CodeSage MCP Server coding standards in CODING_STANDARDS.md:

[code change here]

Check for compliance with:
- Naming conventions
- Modularity principles
- Documentation standards
- Import organization
- Code formatting rules

Provide feedback on:
1. Any violations found
2. Suggested fixes
3. Whether the change aligns with project standards
4. Recommendations for improvement

Be specific about which standards are violated and how to correct them.
```

### For Onboarding New Developers

```
Generate an onboarding guide for new developers joining the CodeSage MCP Server project, based on the standards in CODING_STANDARDS.md.

Include:
1. Overview of key standards
2. Quick reference checklist
3. Common pitfalls to avoid
4. Tool setup instructions
5. Code review expectations
6. Resources for learning

Make it practical and actionable, with examples specific to this codebase.
```

### For Periodic Standards Review

```
Conduct a comprehensive review of the CodeSage MCP Server coding standards in CODING_STANDARDS.md.

Evaluate:
1. Relevance to current codebase
2. Completeness for new requirements
3. Clarity of guidelines
4. Tool and process effectiveness
5. Industry best practice alignment

Recommend updates for:
- Outdated sections
- Missing guidelines
- Improved clarity
- New tools or processes
- Industry standard changes

Provide specific, actionable recommendations with rationale.
```

---

This document serves as the authoritative source for all coding standards in the CodeSage MCP Server project. Regular updates ensure it remains current with evolving best practices and project needs.
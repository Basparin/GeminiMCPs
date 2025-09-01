# CodeSage MCP Server Modularity and Dependencies Guidelines

## Overview

This document establishes comprehensive guidelines for modularity and dependencies in the CodeSage MCP Server codebase. These guidelines ensure maintainable, scalable, and testable code that supports future integrations and development.

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Modularity Principles](#modularity-principles)
3. [Dependency Rules](#dependency-rules)
4. [Import Hierarchy](#import-hierarchy)
5. [Shared Components](#shared-components)
6. [Circular Dependency Prevention](#circular-dependency-prevention)
7. [Cross-Module Integration](#cross-module-integration)
8. [Migration Strategy](#migration-strategy)
9. [Examples](#examples)
10. [Enforcement](#enforcement)

## Current State Analysis

### Existing Structure
```
codesage_mcp/
├── main.py                    # Entry point, heavy imports
├── config/                    # Configuration management
│   ├── __init__.py
│   └── config.py
├── core/                      # Proposed: Core structural components
│   ├── __init__.py
│   └── ... (e.g., exceptions.py, utils.py, gemini_compatibility.py)
├── features/                  # Proposed: Directory for distinct features
│   ├── __init__.py
│   ├── caching/               # Caching system feature
│   │   ├── __init__.py
│   │   └── ...
│   ├── memory_management/     # Memory management feature
│   │   ├── __init__.py
│   │   └── ...
│   ├── performance_monitoring/ # Performance monitoring feature
│   │   ├── __init__.py
│   │   └── ...
│   ├── codebase_manager/      # Codebase manager feature
│   │   ├── __init__.py
│   │   └── codebase_manager.py
│   └── llm_analysis/          # LLM analysis feature
│       ├── __init__.py
│       └── llm_analysis.py
├── tools/                     # Tool implementations
│   ├── __init__.py
│   ├── *_tools.py             # Individual tool modules
└── [additional modules...]
```

### Current Issues Identified

1. **Heavy Interdependencies**: `main.py` imports from 20+ modules
2. **Potential Circular Imports**: Complex import chains (e.g., codesage_mcp/core/indexing ↔ codesage_mcp/features/caching/cache ↔ codesage_mcp/config/config)
3. **Large Modules**: Some modules exceed 2000 lines (codesage_mcp/core/indexing.py, codesage_mcp/features/caching/cache.py)
4. **Mixed Responsibilities**: Modules handle multiple concerns
5. **Deep Import Hierarchies**: Tools import from core modules that import from tools

## Modularity Principles

### 1. Single Responsibility Principle
Each module should have one primary responsibility:

**Good Examples:**
- `config.py` - Configuration management only
- `exceptions.py` - Exception definitions only
- `utils.py` - Pure utility functions

**Anti-Patterns to Avoid:**
- Mixing business logic with infrastructure (e.g., caching + business rules)
- Combining data access with presentation logic
- Modules that handle both synchronous and asynchronous operations

### 2. Interface Segregation
Design interfaces that are client-specific rather than general-purpose:

```python
# Good: Specific interfaces
class CacheInterface(Protocol):
    def get_embedding(self, file_path: str, content: str) -> Optional[np.ndarray]:
        ...

class IndexingInterface(Protocol):
    def index_codebase(self, path: str, model) -> List[str]:
        ...
```

### 3. Dependency Inversion
High-level modules should not depend on low-level modules:

```python
# Good: Depend on abstractions
class CodebaseManager:
    def __init__(self, indexer: IndexingInterface, cache: CacheInterface):
        self.indexer = indexer
        self.cache = cache
```

### 4. Module Size Limits
- **Core modules**: ≤ 500 lines
- **Utility modules**: ≤ 300 lines
- **Tool modules**: ≤ 200 lines
- **Test modules**: ≤ 400 lines

## Dependency Rules

### 1. Import Direction Rules

**Allowed Import Directions:**
```
config.py → (no imports from business logic)
exceptions.py → (minimal imports)
utils.py → (minimal imports)
logging_config.py → exceptions.py, config.py

codesage_mcp/features/caching/cache.py → codesage_mcp/config/config.py, codesage_mcp/core/exceptions.py, codesage_mcp/core/utils.py
codesage_mcp/core/indexing.py → codesage_mcp/config/config.py, codesage_mcp/core/exceptions.py, codesage_mcp/features/caching/cache.py, codesage_mcp/features/memory_management/memory_manager.py
codesage_mcp/features/memory_management/memory_manager.py → codesage_mcp/config/config.py, codesage_mcp/core/exceptions.py

codesage_mcp/features/codebase_manager/codebase_manager.py → codesage_mcp/core/indexing.py, codesage_mcp/core/searching.py, codesage_mcp/features/llm_analysis/llm_analysis.py
codesage_mcp/features/llm_analysis/llm_analysis.py → codesage_mcp/config/config.py, codesage_mcp/core/exceptions.py, codesage_mcp/core/utils.py

codesage_mcp/tools/*.py → codesage_mcp/features/codebase_manager/codebase_manager.py, codesage_mcp/features/llm_analysis/llm_analysis.py, codesage_mcp/core/utils.py
codesage_mcp/main.py → codesage_mcp/tools/*.py, codesage_mcp/config/config.py, codesage_mcp/core/exceptions.py
```

**Prohibited Import Directions:**
- Business logic modules importing from `main.py`
- Core modules importing from `tools/`
- Any module importing from `main.py`

### 2. Import Statement Organization

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

# 4. Relative imports (only within same package)
from .cache import get_cache_instance
```

### 3. Import Aliasing Rules

```python
# Good: Clear aliases for complex names
from codesage_mcp.core.indexing import IndexingManager as IndexManager

# Bad: Unclear aliases
from codesage_mcp.indexing import IndexingManager as IM
```

## Import Hierarchy

### Layer Definitions

1. **Infrastructure Layer** (Bottom)
   - `config.py`, `exceptions.py`, `utils.py`
   - `logging_config.py`, `prometheus_client.py`

2. **Core Services Layer**
    - `codesage_mcp/features/caching/cache.py`, `codesage_mcp/features/memory_management/memory_manager.py`
    - `codesage_mcp/core/chunking.py`, `codesage_mcp/features/performance_monitoring/performance_monitor.py`

3. **Business Logic Layer**
    - `codesage_mcp/core/indexing.py`, `codesage_mcp/core/searching.py`, `codesage_mcp/features/llm_analysis/llm_analysis.py`
    - `codesage_mcp/features/codebase_manager/codebase_manager.py`, `codesage_mcp/features/caching/adaptive_cache_manager.py`

4. **Interface Layer** (Top)
    - `codesage_mcp/tools/*.py`, `codesage_mcp/main.py`

### Hierarchical Import Rules

```python
# Infrastructure → Infrastructure (limited)
from .config import ENABLE_CACHING
from .exceptions import BaseMCPError

# Core Services → Infrastructure only
from codesage_mcp.config.config import MAX_MEMORY_MB
from codesage_mcp.core.exceptions import IndexingError

# Business Logic → Core Services + Infrastructure
from codesage_mcp.features.caching.cache import get_cache_instance
from codesage_mcp.features.memory_management.memory_manager import get_memory_manager
from codesage_mcp.config.config import CHUNK_SIZE_TOKENS

# Interface → Business Logic + Core Services + Infrastructure
from codesage_mcp.features.codebase_manager.codebase_manager import get_llm_analysis_manager
from codesage_mcp.features.caching.cache import IntelligentCache
from codesage_mcp.config.config import get_configuration_status
```

## Shared Components

### 1. Common Interfaces

Create abstract base classes for shared behavior:

```python
# codesage_mcp/interfaces.py
from abc import ABC, abstractmethod
from typing import Protocol, Optional
import numpy as np

class CacheInterface(Protocol):
    def get_embedding(self, file_path: str, content: str) -> Optional[np.ndarray]:
        ...

    def store_embedding(self, file_path: str, content: str, embedding: np.ndarray) -> None:
        ...

class IndexingInterface(Protocol):
    def index_codebase(self, path: str, model) -> List[str]:
        ...

    def search_similar(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        ...
```

### 2. Shared Data Models

```python
# codesage_mcp/models.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class FileMetadata:
    path: str
    size: int
    mtime: datetime
    hash: str
    indexed_at: Optional[datetime] = None

@dataclass
class SearchResult:
    file_path: str
    similarity_score: float
    chunk_content: str
    line_number: Optional[int] = None
```

### 3. Factory Pattern for Complex Objects

```python
# codesage_mcp/core/factories.py
from codesage_mcp.features.caching.cache import IntelligentCache
from codesage_mcp.core.indexing import IndexingManager

class ComponentFactory:
    _cache_instance: Optional[CacheInterface] = None
    _indexer_instance: Optional[IndexingInterface] = None

    @classmethod
    def get_cache(cls, config: Dict[str, Any]) -> CacheInterface:
        if cls._cache_instance is None:
            cls._cache_instance = IntelligentCache(config=config)
        return cls._cache_instance

    @classmethod
    def get_indexer(cls, cache: CacheInterface) -> IndexingInterface:
        if cls._indexer_instance is None:
            cls._indexer_instance = IndexingManager()
            # Inject dependencies
            cls._indexer_instance.cache = cache
        return cls._indexer_instance
```

## Circular Dependency Prevention

### Detection Tools

Use static analysis to detect circular imports:

```bash
# Check for circular imports
python -m py_compile codesage_mcp/*.py
python -c "import codesage_mcp; print('No circular imports detected')"

# Use pylint for import analysis
pylint codesage_mcp/ --disable=all --enable=imports
```

### Prevention Strategies

1. **Dependency Injection**: Pass dependencies rather than importing
2. **Event-Driven Architecture**: Use callbacks/events instead of direct calls
3. **Plugin Architecture**: Load components dynamically
4. **Interface Segregation**: Break large interfaces into smaller ones

### Example: Breaking Circular Import

**Before (Problematic):**
```python
# codesage_mcp/core/indexing.py
from codesage_mcp.features.caching.cache import get_cache_instance

class IndexingManager:
    def __init__(self):
        self.cache = get_cache_instance()

# codesage_mcp/features/caching/cache.py
from codesage_mcp.core.indexing import IndexingManager

def get_cache_instance():
    indexer = IndexingManager()  # Circular!
    return IntelligentCache(indexer=indexer)
```

**After (Fixed):**
```python
# codesage_mcp/core/indexing.py
class IndexingManager:
    def __init__(self, cache: CacheInterface):
        self.cache = cache

# codesage_mcp/features/caching/cache.py
def get_cache_instance() -> IntelligentCache:
    return IntelligentCache()

# main.py
cache = get_cache_instance()
indexer = IndexingManager(cache=cache)
```

## Cross-Module Integration

### 1. Event System

```python
# codesage_mcp/events.py
from typing import Callable, Dict, List, Any
from enum import Enum

class EventType(Enum):
    FILE_INDEXED = "file_indexed"
    CACHE_INVALIDATED = "cache_invalidated"
    PERFORMANCE_REGRESSION = "performance_regression"

class EventBus:
    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}

    def subscribe(self, event_type: EventType, callback: Callable):
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def publish(self, event_type: EventType, data: Any):
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Event callback failed: {e}")
```

### 2. Plugin System

```python
# codesage_mcp/plugins.py
from typing import Protocol, runtime_checkable
import importlib
import pkgutil

@runtime_checkable
class PluginInterface(Protocol):
    def initialize(self, context: Dict[str, Any]) -> None:
        ...

    def get_tools(self) -> List[Dict[str, Any]]:
        ...

class PluginManager:
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = plugin_dir
        self.plugins: List[PluginInterface] = []

    def load_plugins(self):
        for _, name, _ in pkgutil.iter_modules([self.plugin_dir]):
            try:
                module = importlib.import_module(f"{self.plugin_dir}.{name}")
                plugin_class = getattr(module, f"{name.capitalize()}Plugin")
                plugin = plugin_class()
                self.plugins.append(plugin)
            except Exception as e:
                logger.warning(f"Failed to load plugin {name}: {e}")
```

### 3. Service Locator Pattern

```python
# codesage_mcp/services.py
from typing import Dict, Any, Optional, TypeVar, Generic
from .interfaces import CacheInterface, IndexingInterface

T = TypeVar('T')

class ServiceLocator:
    _services: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, service: Any):
        cls._services[name] = service

    @classmethod
    def get(cls, name: str, service_type: type[T] = None) -> T:
        service = cls._services.get(name)
        if service_type and not isinstance(service, service_type):
            raise TypeError(f"Service {name} is not of type {service_type}")
        return service

# Usage
ServiceLocator.register("cache", IntelligentCache())
ServiceLocator.register("indexer", IndexingManager())

# In modules that need services
cache = ServiceLocator.get("cache", CacheInterface)
```

## Migration Strategy

### Phase 1: Analysis and Planning (Week 1-2)

1. **Import Graph Analysis**
   ```bash
   # Generate import dependency graph
   python -c "
   import codesage_mcp
   # Analyze and visualize import relationships
   "
   ```

2. **Identify High-Risk Modules**
   - Modules with >10 imports
   - Modules imported by >5 other modules
   - Modules with complex dependency chains

3. **Create Migration Roadmap**
   - Prioritize based on risk and impact
   - Identify safe refactoring opportunities

### Phase 2: Interface Extraction (Week 3-4)

1. **Extract Common Interfaces**
   ```python
   # Create interfaces.py
   from typing import Protocol
   from abc import abstractmethod

   class CacheProtocol(Protocol):
       @abstractmethod
       def get(self, key: str) -> Any: ...

   class IndexingProtocol(Protocol):
       @abstractmethod
       def index(self, path: str) -> List[str]: ...
   ```

2. **Create Factory Classes**
   ```python
   # Create factories.py
   class ServiceFactory:
       @staticmethod
       def create_cache() -> CacheProtocol:
           return IntelligentCache()

       @staticmethod
       def create_indexer() -> IndexingProtocol:
           return IndexingManager()
   ```

### Phase 3: Refactoring Implementation (Week 5-8)

1. **Bottom-Up Refactoring**
   - Start with infrastructure modules
   - Gradually move up the hierarchy
   - Update tests after each change

2. **Dependency Injection Implementation**
   ```python
   # Before
   class CodebaseManager:
       def __init__(self):
           self.cache = get_cache_instance()

   # After
   class CodebaseManager:
       def __init__(self, cache: CacheInterface):
           self.cache = cache
   ```

3. **Module Splitting**
   - Break large modules into smaller ones
   - Extract common functionality
   - Maintain backward compatibility

### Phase 4: Testing and Validation (Week 9-10)

1. **Comprehensive Testing**
   ```bash
   # Run full test suite
   pytest tests/ -v --cov=codesage_mcp

   # Check for import errors
   python -c "import codesage_mcp; print('Import successful')"
   ```

2. **Performance Validation**
   - Benchmark before/after refactoring
   - Memory usage analysis
   - Startup time measurement

## Examples

### Example 1: Refactoring Large Module

**Before:**
```python
# codesage_mcp/core/indexing.py (2000+ lines)
class IndexingManager:
    # 15+ methods, multiple responsibilities
    def index_codebase(self): pass
    def search_similar(self): pass
    def manage_memory(self): pass  # Wrong responsibility
    def handle_cache(self): pass   # Wrong responsibility
```

**After:**
```python
# indexing/core.py
class IndexingManager:
    def __init__(self, cache: CacheInterface, memory_mgr: MemoryInterface):
        self.cache = cache
        self.memory_mgr = memory_mgr

    def index_codebase(self): pass
    def search_similar(self): pass

# indexing/memory.py
class IndexingMemoryManager:
    def optimize_for_indexing(self): pass

# codesage_mcp/core/indexing_cache.py
class IndexingCacheManager:
    def prefetch_for_indexing(self): pass
```

### Example 2: Dependency Injection

**Before:**
```python
# main.py
from codesage_mcp.core.indexing import IndexingManager
from codesage_mcp.features.caching.cache import get_cache_instance

indexer = IndexingManager()
cache = get_cache_instance()
```

**After:**
```python
# main.py
from codesage_mcp.features.caching.cache import get_cache_instance
from codesage_mcp.core.indexing import IndexingManager

cache = get_cache_instance()
indexer = IndexingManager(cache=cache)
```

### Example 3: Event-Driven Communication

**Before:**
```python
# Direct coupling
class IndexingManager:
    def __init__(self, cache):
        self.cache = cache

    def on_file_changed(self, file_path):
        self.cache.invalidate_file(file_path)  # Direct call
```

**After:**
```python
# Event-driven
class IndexingManager:
    def __init__(self, event_bus):
        self.event_bus = event_bus

    def on_file_changed(self, file_path):
        self.event_bus.publish(EventType.FILE_CHANGED, {"path": file_path})
```

## Enforcement

### Automated Checks

1. **Pre-commit Hooks**
   ```yaml
   # .pre-commit-config.yaml
   repos:
     - repo: local
       hooks:
         - id: check-imports
           name: Check import hierarchy
           entry: python scripts/check_imports.py
           language: system
           files: ^codesage_mcp/
   ```

2. **CI/CD Pipeline**
   ```yaml
   # .github/workflows/checks.yml
   - name: Check Modularity
     run: |
       python scripts/validate_modularity.py
       python -m py_compile codesage_mcp/*.py
   ```

### Code Review Guidelines

1. **Import Rules**
   - [ ] No imports from higher layers
   - [ ] No circular import chains
   - [ ] Import statements properly grouped and sorted

2. **Module Size**
   - [ ] Core modules ≤ 500 lines
   - [ ] Functions ≤ 50 lines
   - [ ] Classes ≤ 200 lines

3. **Dependency Injection**
   - [ ] No direct instantiation of services
   - [ ] Dependencies passed via constructor
   - [ ] Use interfaces, not concrete classes

### Monitoring

1. **Import Analysis**
   ```python
   # scripts/analyze_imports.py
   def analyze_import_graph():
       # Generate visual import dependency graph
       # Identify problematic patterns
       # Suggest refactoring opportunities
   ```

2. **Complexity Metrics**
   ```python
   # scripts/check_complexity.py
   def check_module_complexity():
       # Cyclomatic complexity
       # Module coupling metrics
       # Import depth analysis
   ```

## Conclusion

These guidelines provide a comprehensive framework for maintaining clean, modular code in the CodeSage MCP Server. By following these principles, we ensure:

- **Maintainability**: Clear separation of concerns
- **Testability**: Isolated components with defined interfaces
- **Scalability**: Easy addition of new features
- **Reliability**: Reduced risk of circular dependencies and import errors

Regular review and enforcement of these guidelines will help maintain code quality as the project grows and evolves.
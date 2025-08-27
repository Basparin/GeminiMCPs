"""Codebase Manager Module.

This module provides the CodebaseManager class, which acts as a central
coordinator for managing codebases, indexing, searching, and interacting with
LLM analysis tools. It delegates specific responsibilities to specialized
modules like indexing, searching, and llm_analysis.

The CodebaseManager is responsible for:
- Coordinating the initialization and interaction between different subsystems.
- Providing a high-level interface for codebase operations.
- Maintaining compatibility with legacy code.

Classes:
    CodebaseManager: The main class for managing codebases and delegating tasks.
"""

import os
from groq import Groq
from openai import OpenAI
import google.generativeai as genai
import ast
from collections import defaultdict  # New import

# Importaciones para modelos y búsqueda
from sentence_transformers import SentenceTransformer

from .config import (
    GROQ_API_KEY,
    OPENROUTER_API_KEY,
    GOOGLE_API_KEY,
    ENABLE_INCREMENTAL_INDEXING,
    FORCE_FULL_REINDEX,
    validate_configuration,
    get_configuration_status
)

# Importar los nuevos módulos
from .indexing import IndexingManager
from .searching import SearchingManager
from .llm_analysis import LLMAnalysisManager
from .utils import safe_read_file
from .config import ENABLE_CACHING
from .cache import get_cache_instance
from .memory_manager import get_memory_manager


def _is_module_installed(module_name: str) -> bool:
    """Checks if a third-party module is installed."""
    try:
        import importlib.util

        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _get_stdlib_modules() -> set:
    """Get a list of standard library modules.

    Returns:
        set: A set of standard library module names.
    """
    # Get a list of standard library modules
    try:
        from stdlib_list import stdlib_list

        # Get stdlib modules for the current Python version
        import sys

        stdlib_modules = set(
            stdlib_list(f"{sys.version_info.major}.{sys.version_info.minor}")
        )
    except Exception:
        # Fallback if stdlib_list is not available or fails
        stdlib_modules = set()
        # Add some common stdlib modules for fallback
        stdlib_modules.update(
            [
                "os",
                "sys",
                "json",
                "ast",
                "collections",
                "itertools",
                "functools",
                "re",
                "io",
                "pathlib",
                "datetime",
                "math",
                "random",
                "urllib",
                "http",
                "sqlite3",
                "threading",
                "asyncio",
            ]
        )

    return stdlib_modules


def _get_module_type(
    module_name: str, current_codebase_path: str, stdlib_modules: set
) -> str:
    """Determines if a module is internal, standard library, or third-party."""
    if (
        os.path.exists(os.path.join(current_codebase_path, module_name))
        or os.path.exists(os.path.join(current_codebase_path, module_name + ".py"))
        or os.path.exists(
            os.path.join(current_codebase_path, module_name, "__init__.py")
        )
    ):
        return "internal"
    elif module_name in stdlib_modules:
        return "stdlib"
    else:
        return "third_party"


def _process_import_node(
    node: ast.AST,
    current_codebase_path: str,
    stdlib_modules: set,
    internal_dependencies: defaultdict,
    stdlib_dependencies: defaultdict,
    third_party_dependencies: defaultdict,
    all_stdlib_modules: set,
    all_third_party_modules: set,
    installed_third_party_modules: set,
    not_installed_third_party_modules: set,
    relative_file_path: str,
):
    """Processes an AST import node and updates dependency sets."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            module_type = _get_module_type(
                module_name, current_codebase_path, stdlib_modules
            )
            if module_type == "internal":
                internal_dependencies[relative_file_path].add(module_name)
            elif module_type == "stdlib":
                stdlib_dependencies[relative_file_path].add(module_name)
                all_stdlib_modules.add(module_name)
            else:  # third_party
                third_party_dependencies[relative_file_path].add(module_name)
                all_third_party_modules.add(module_name)
                if _is_module_installed(module_name):
                    installed_third_party_modules.add(module_name)
                else:
                    not_installed_third_party_modules.add(module_name)

    elif isinstance(node, ast.ImportFrom):
        module_name = node.module.split(".")[0] if node.module else ""
        if module_name:
            module_type = _get_module_type(
                module_name, current_codebase_path, stdlib_modules
            )
            if module_type == "internal":
                internal_dependencies[relative_file_path].add(module_name)
            elif module_type == "stdlib":
                stdlib_dependencies[relative_file_path].add(module_name)
                all_stdlib_modules.add(module_name)
            else:  # third_party
                third_party_dependencies[relative_file_path].add(module_name)
                all_third_party_modules.add(module_name)
                if _is_module_installed(module_name):
                    installed_third_party_modules.add(module_name)
                else:
                    not_installed_third_party_modules.add(module_name)


class CodebaseManager:
    """Central coordinator for managing codebases and delegating tasks to
    specialized modules.

        This class acts as a thin layer that coordinates interactions between different
        subsystems of the CodeSage MCP Server. It holds references to specialized managers
        (IndexingManager, SearchingManager, LLMAnalysisManager) and delegates specific
        responsibilities to them.

        Attributes:
            indexing_manager (IndexingManager): Manages codebase indexing.
            searching_manager (SearchingManager): Manages codebase searching.
            llm_analysis_manager (LLMAnalysisManager): Manages LLM-based code analysis.
            sentence_transformer_model (SentenceTransformer): Model for semantic operations.
            groq_client (Groq): Client for Groq API.
            openrouter_client (OpenAI): Client for OpenRouter API.
            google_ai_client (google.generativeai.GenerativeModel): Client for Google
    AI API.
    """

    def __init__(self):
        """Initializes the CodebaseManager with default settings and clients.

        Sets up the indexing manager, sentence transformer model,
        and API clients for Groq, OpenRouter, and Google AI.
        Validates configuration and logs any issues.
        """
        # Validate configuration on startup
        config_status = get_configuration_status()
        if not config_status["valid"]:
            print("Warning: Configuration issues detected:")
            for issue in config_status["issues"]:
                print(f"  - {issue}")
            print("Some LLM providers may not be available.")

        # Inicializar el gestor de indexación
        self.indexing_manager = IndexingManager()
        # Inicializar el gestor de búsqueda
        self.searching_manager = SearchingManager(self.indexing_manager)

        # Inicializar memory manager
        self.memory_manager = get_memory_manager()

        # Acceder a los atributos relevantes a través del indexing_manager
        # para compatibilidad con el código existente que los usa directamente.
        # Se usan propiedades para mantener la sincronización.

        # Model will be loaded on-demand instead of permanently
        self._sentence_transformer_model = None
        self._model_name = "all-MiniLM-L6-v2"

        # Initialize API clients with proper error handling
        self.groq_client = None
        if GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=GROQ_API_KEY)
                print("Groq client initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize Groq client: {e}")
        else:
            print("Groq API key not configured - Groq features will be unavailable.")

        self.openrouter_client = None
        if OPENROUTER_API_KEY:
            try:
                self.openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=OPENROUTER_API_KEY,
                )
                print("OpenRouter client initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize OpenRouter client: {e}")
        else:
            print("OpenRouter API key not configured - OpenRouter features will be unavailable.")

        self.google_ai_client = None
        if GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                self.google_ai_client = genai.GenerativeModel("gemini-2.0-flash")
                print("Google AI client initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize Google AI client: {e}")
        else:
            print("Google API key not configured - Google AI features will be unavailable.")

        # Inicializar el gestor de análisis con LLM
        self.llm_analysis_manager = LLMAnalysisManager(
            self.groq_client, self.openrouter_client, self.google_ai_client
        )

        # Initialize cache if enabled
        self.cache = get_cache_instance() if ENABLE_CACHING else None

    @property
    def sentence_transformer_model(self):
        """Get the sentence transformer model, loading it on-demand if needed."""
        if self._sentence_transformer_model is None:
            print(f"Loading sentence transformer model: {self._model_name}")
            self._sentence_transformer_model = self.memory_manager.load_model(self._model_name)
        return self._sentence_transformer_model

    # --- Properties to maintain compatibility with legacy code ---
    # These properties delegate to the indexing_manager to ensure
    # that legacy code accessing them directly continues to work.
    # They also ensure synchronization between the CodebaseManager's
    # view and the IndexingManager's state.
    @property
    def indexed_codebases(self):
        """Legacy property for accessing indexed codebases."""
        return self.indexing_manager.indexed_codebases

    @indexed_codebases.setter
    def indexed_codebases(self, value):
        """Legacy setter for indexed codebases."""
        self.indexing_manager.indexed_codebases = value

    @property
    def file_paths_map(self):
        """Legacy property for accessing file paths map."""
        return self.indexing_manager.file_paths_map

    @file_paths_map.setter
    def file_paths_map(self, value):
        """Legacy setter for file paths map."""
        self.indexing_manager.file_paths_map = value

    @property
    def faiss_index(self):
        """Legacy property for accessing FAISS index."""
        return self.indexing_manager.faiss_index

    @faiss_index.setter
    def faiss_index(self, value):
        """Legacy setter for FAISS index."""
        self.indexing_manager.faiss_index = value

    # --- Delegated Methods ---
    # These methods now correctly delegate all their responsibilities
    # to the corresponding specialized managers.
    # This keeps the CodebaseManager as a thin coordinator.

    def read_code_file(self, file_path: str) -> str:
        """Reads and returns the content of a specified code file.

        This method uses intelligent caching to improve performance for frequently accessed files.

        Args:
            file_path (str): Path to the file to read.

        Returns:
            str: Content of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        # Check cache first
        if self.cache:
            cached_content, cache_hit = self.cache.get_file_content(file_path)
            if cache_hit:
                print(f"Cache hit for file: {file_path}")
                return cached_content

        # Read from disk
        content = safe_read_file(file_path)

        # Store in cache for future use
        if self.cache:
            self.cache.store_file_content(file_path, content)
            print(f"Cached file content for: {file_path}")

            # Trigger smart prefetching based on usage patterns
            try:
                prefetch_stats = self.cache.smart_prefetch(file_path, self.indexing_manager.indexed_codebases, self.sentence_transformer_model)
                if prefetch_stats["prefetched"] > 0:
                    print(f"Smart prefetching completed: {prefetch_stats['prefetched']} files prefetched")
            except Exception as e:
                print(f"Warning: Smart prefetching failed: {e}")

        return content

    def index_codebase(self, path: str, force_full: bool = None) -> list[str]:
        """Indexes a given codebase path for analysis, respecting .gitignore rules.

        This method delegates the actual indexing logic to the IndexingManager.
        Supports both incremental and full indexing based on configuration.

        Args:
            path (str): Path to the codebase directory to index.
            force_full (bool, optional): Force full re-indexing. If None, uses config setting.

        Returns:
            list[str]: List of indexed file paths relative to the codebase root.
        """
        # Determine if we should use incremental indexing
        use_incremental = ENABLE_INCREMENTAL_INDEXING and not (force_full if force_full is not None else FORCE_FULL_REINDEX)

        if use_incremental and hasattr(self.indexing_manager, 'index_codebase_incremental'):
            # Use incremental indexing
            indexed_files, was_incremental = self.indexing_manager.index_codebase_incremental(
                path, self.sentence_transformer_model, force_full=force_full if force_full is not None else FORCE_FULL_REINDEX
            )
            if was_incremental:
                print(f"Used incremental indexing for {path}")
            else:
                print(f"Used full indexing for {path}")
        else:
            # Use traditional full indexing
            indexed_files = self.indexing_manager.index_codebase(
                path, self.sentence_transformer_model
            )
            print(f"Used full indexing for {path}")

        # Actualizar las referencias locales para compatibilidad con el código existente
        self.indexed_codebases = self.indexing_manager.indexed_codebases
        self.file_paths_map = self.indexing_manager.file_paths_map

        # Warm the cache with commonly accessed files
        if self.cache and path:
            try:
                warming_stats = self.cache.warm_cache(path, self.sentence_transformer_model)
                if warming_stats["files_warmed"] > 0 or warming_stats["embeddings_cached"] > 0:
                    print(f"Cache warming completed: {warming_stats['files_warmed']} files warmed, "
                          f"{warming_stats['embeddings_cached']} embeddings cached")
            except Exception as e:
                print(f"Warning: Cache warming failed: {e}")

        return indexed_files

    def force_full_reindex(self, path: str) -> list[str]:
        """Force a full re-indexing of a codebase, bypassing incremental indexing.

        This method is useful when you want to ensure a complete rebuild of the index,
        for example after major changes to the codebase structure or when incremental
        indexing might have inconsistencies.

        Args:
            path (str): Path to the codebase directory to re-index.

        Returns:
            list[str]: List of indexed file paths relative to the codebase root.
        """
        return self.index_codebase(path, force_full=True)

    def get_file_structure(self, codebase_path: str, file_path: str) -> list[str]:
        """Provides a high-level overview of a file's structure within a given codebase.

        This method uses the AST (Abstract Syntax Tree) module to parse the file and
        extract key structural elements like classes, functions, and methods. For
        non-Python files or Python files without identifiable structure, it returns
        the file path as a fallback.

        Args:
            codebase_path (str): Path to the indexed codebase.
            file_path (str): Relative path to the file within the codebase.

        Returns:
            list[str]: List of structural elements (classes, functions, methods)
                       in a hierarchical format, e.g., ['ClassName',
                       'ClassName.method_name', 'function_name'].
                       If the file is not a Python file or has no structure, returns
                       [file_path].
        """
        abs_file_path = os.path.join(codebase_path, file_path)
        if not os.path.exists(abs_file_path):
            raise FileNotFoundError(f"File not found: {abs_file_path}")

        # If it's not a Python file, return the file path as structure
        if not file_path.endswith(".py"):
            return [file_path]

        try:
            with open(abs_file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            structure = []

            # Traverse the AST to find classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    structure.append(node.name)
                    # Find methods within the class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            structure.append(f"{node.name}.{item.name}")
                elif isinstance(node, ast.FunctionDef) and not any(
                    isinstance(parent, ast.ClassDef)
                    for parent in ast.walk(tree)
                    if node in parent.body
                ):
                    # Add top-level functions (not methods)
                    structure.append(node.name)

            # If no structure was found, return the file path as a fallback
            if not structure:
                return [file_path]

            return sorted(structure)

        except SyntaxError:
            # If the file is not valid Python, return the filename as part of
            # the structure
            # to indicate that it was attempted to be analyzed.
            return [file_path]

    def semantic_search_codebase(self, query: str, top_k: int = 5) -> list[dict]:
        """Performs a semantic search within the indexed codebase to find code snippets
        semantically similar to the given query.

        This method delegates the actual search logic to the SearchingManager.

        Args:
            query (str): The semantic query to search for.
            top_k (int, optional): Number of results to return. Defaults to 5.

        Returns:
            list[dict]: List of semantically similar code snippets.
        """
        return self.searching_manager.semantic_search_codebase(
            query, self.sentence_transformer_model, top_k
        )

    def get_dependencies_overview(self, codebase_path: str) -> dict:
        """Analyzes Python files in the indexed codebase and extracts import statements,
        providing a high-level overview of internal and external dependencies.

        Args:
            codebase_path (str): Path to the indexed codebase.

        Returns:
            dict: Dependency overview with statistics and lists of internal, stdlib, and
                third-party dependencies, or an error message.
        """
        abs_codebase_path = os.path.abspath(codebase_path)
        if abs_codebase_path not in self.indexed_codebases:
            raise ValueError(
                f"Codebase at {abs_codebase_path} has not been indexed. "
                "Please index it first."
            )

        indexed_files = self.indexed_codebases[abs_codebase_path]["files"]

        internal_dependencies = defaultdict(set)
        stdlib_dependencies = defaultdict(set)
        third_party_dependencies = defaultdict(set)
        all_stdlib_modules = set()
        all_third_party_modules = set()
        installed_third_party_modules = set()
        not_installed_third_party_modules = set()

        stdlib_modules = _get_stdlib_modules()

        for relative_file_path in indexed_files:
            if not relative_file_path.endswith(".py"):
                continue

            file_path = os.path.join(abs_codebase_path, relative_file_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    _process_import_node(
                        node,
                        abs_codebase_path,
                        stdlib_modules,
                        internal_dependencies,
                        stdlib_dependencies,
                        third_party_dependencies,
                        all_stdlib_modules,
                        all_third_party_modules,
                        installed_third_party_modules,
                        not_installed_third_party_modules,
                        relative_file_path,
                    )

            except Exception:
                pass

        return {
            "message": "Dependency overview generated.",
            "total_python_files_analyzed": (
                len(internal_dependencies)
                + len(stdlib_dependencies)
                + len(third_party_dependencies)
            ),
            "total_internal_dependencies": sum(
                len(deps) for deps in internal_dependencies.values()
            ),
            "total_stdlib_dependencies": sum(
                len(deps) for deps in stdlib_dependencies.values()
            ),
            "total_third_party_dependencies": sum(
                len(deps) for deps in third_party_dependencies.values()
            ),
            "unique_stdlib_modules": sorted(list(all_stdlib_modules)),
            "unique_third_party_modules": sorted(list(all_third_party_modules)),
            "installed_third_party_modules": sorted(
                list(installed_third_party_modules)
            ),
            "not_installed_third_party_modules": sorted(
                list(not_installed_third_party_modules)
            ),
            "internal_dependencies_by_file": {
                k: sorted(list(v)) for k, v in internal_dependencies.items()
            },
            "stdlib_dependencies_by_file": {
                k: sorted(list(v)) for k, v in stdlib_dependencies.items()
            },
            "third_party_dependencies_by_file": {
                k: sorted(list(v)) for k, v in third_party_dependencies.items()
            },
        }

    def get_memory_stats(self) -> dict:
        """Get current memory usage statistics.

        Returns:
            dict: Memory statistics including usage, limits, and cache information.
        """
        return self.memory_manager.get_memory_stats()

    def cleanup_memory(self) -> dict:
        """Perform memory cleanup and return statistics.

        Returns:
            dict: Cleanup results and memory statistics.
        """
        before_stats = self.memory_manager.get_memory_stats()
        self.memory_manager.cleanup()
        after_stats = self.memory_manager.get_memory_stats()

        return {
            "cleanup_performed": True,
            "memory_freed_mb": before_stats['rss_mb'] - after_stats['rss_mb'],
            "before_cleanup": before_stats,
            "after_cleanup": after_stats
        }


# Create a global instance of the CodebaseManager
codebase_manager = CodebaseManager()

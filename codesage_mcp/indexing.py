"""
Indexing Module for CodeSage MCP Server.

This module contains the logic for indexing codebases, managing .gitignore
files, and persisting indexes to disk.

It provides the IndexingManager class, which is responsible for:
- Creating and maintaining FAISS indexes and associated metadata.
- Handling file system operations related to indexing.
- Respecting .gitignore rules during the indexing process.
- Persisting and loading indexes from disk.

Classes:
    IndexingManager: Manages codebase indexing.
"""

import os
import json
import fnmatch
from pathlib import Path
import faiss
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import caching system
from .config import ENABLE_CACHING
from .cache import get_cache_instance

# Import memory management
from .memory_manager import get_memory_manager
from .chunking import DocumentChunker, chunk_file


class IndexingManager:
    """Manages codebase indexing.

    This class is responsible for creating and maintaining FAISS indexes and
    associated metadata. It handles file system operations related to indexing
    and respects .gitignore rules during the indexing process. It also persists
    and loads indexes from disk.

    Attributes:
        index_dir_name (str): Name of the directory where indexes are stored.
        index_dir (Path): Path object for the index directory.
        index_file (Path): Path object for the index metadata file.
        faiss_index_file (Path): Path object for the FAISS index file.
        indexed_codebases (dict): Dictionary of indexed codebases.
        file_paths_map (dict): Mapping of FAISS IDs to file paths.
        faiss_index (faiss.Index): The FAISS index object.
    """

    def __init__(self, index_dir_name: str = ".codesage"):
        """Initializes the IndexingManager.

                  Args:
                      index_dir_name (str, optional): Name of the directory where indexes
          are stored.
                           Defaults to ".codesage".
          """
        self.index_dir_name = index_dir_name
        self.index_dir = Path(self.index_dir_name)
        self.index_file = self.index_dir / "codebase_index.json"
        self.faiss_index_file = self.index_dir / "codebase_index.faiss"
        self.metadata_file = self.index_dir / "codebase_metadata.json"
        self.indexed_codebases = {}
        self.file_paths_map = {}
        self.file_metadata = {}  # Stores file timestamps and metadata

        # Initialize cache if enabled
        self.cache = get_cache_instance() if ENABLE_CACHING else None

        # Initialize memory manager
        self.memory_manager = get_memory_manager()

        # Initialize chunker for document processing
        self.chunker = DocumentChunker()

        # Thread pool for parallel processing
        self._executor = None
        self._executor_lock = threading.RLock()

        # Dependency tracking for incremental indexing
        self._dependency_graph = {}  # file -> set of files that depend on it
        self._reverse_dependencies = {}  # file -> set of files it depends on

        self._initialize_index()

    def _initialize_index(self):
        """Inicializa el índice cargando datos existentes o creando uno nuevo."""
        self.index_dir.mkdir(exist_ok=True)
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    loaded_data = json.load(f)
                    self.indexed_codebases = loaded_data.get("indexed_codebases", {})
                    self.file_paths_map = loaded_data.get("file_paths_map", {})
            except (json.JSONDecodeError, IOError) as e:
                print(
                    f"Warning: Could not load codebase index. "
                    f"A new one will be created. Error: {e}"
                )
                self.indexed_codebases = {}
                self.file_paths_map = {}

        # Load metadata
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    self.file_metadata = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load metadata. Error: {e}")
                self.file_metadata = {}
        else:
            self.file_metadata = {}

        # Inicializar FAISS con soporte para memory mapping
        if self.faiss_index_file.exists():
            try:
                self.faiss_index = self.memory_manager.load_faiss_index(str(self.faiss_index_file))
                print(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            except Exception as e:
                print(f"Warning: Could not load FAISS index. Error: {e}")
                self.faiss_index = None

        if self.faiss_index is None:
            # Crear nuevo índice FAISS si no se cargó ninguno
            # Nota: Se necesita el modelo para obtener la dimensión. Se creará en
            # index_codebase si es necesario.
            pass

    def _save_index(self):
        """Guarda el índice actual en archivos."""
        data_to_save = {
            "indexed_codebases": self.indexed_codebases,
            "file_paths_map": self.file_paths_map,
        }
        with open(self.index_file, "w") as f:
            json.dump(data_to_save, f, indent=4)

        # Save metadata
        with open(self.metadata_file, "w") as f:
            json.dump(self.file_metadata, f, indent=4, default=str)

        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(self.faiss_index_file))

    def _get_gitignore_patterns(self, path: Path) -> list[str]:
        """Encuentra y parsea el archivo .gitignore en la ruta dada."""
        gitignore_file = path / ".gitignore"
        patterns = []
        if gitignore_file.exists():
            with open(gitignore_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        return patterns

    def _is_ignored(
        self,
        file_path: Path,
        gitignore_patterns: list[str],
        root_path: Path,
    ) -> bool:
        """Checks if a file or directory should be ignored based on .gitignore
        patterns."""
        relative_path = file_path.relative_to(root_path)
        relative_path_str = str(relative_path)
        file_name_str = file_path.name

        for pattern in gitignore_patterns:
            # Handle patterns ending with a slash (directories)
            if pattern.endswith("/"):
                # Check if the relative path starts with the directory pattern
                if relative_path_str.startswith(pattern):
                    return True
                # Check if any part of the relative path matches the directory name
                # e.g., "venv/" should ignore "foo/venv/bar"
                pattern_dir_name = pattern.rstrip("/")
                if pattern_dir_name in relative_path.parts:
                    return True

            # Handle general glob patterns against the full relative path
            if fnmatch.fnmatch(relative_path_str, pattern):
                return True

            # Handle patterns that match only the file/directory name
            # This covers cases like "foo" ignoring "bar/foo"
            if "/" not in pattern and fnmatch.fnmatch(file_name_str, pattern):
                return True

        return False

    def _get_file_mtime(self, file_path: Path) -> float:
        """Get the modification time of a file.

        Args:
            file_path: Path to the file

        Returns:
            Modification time as a float timestamp
        """
        try:
            return file_path.stat().st_mtime
        except OSError:
            return 0.0

    def _get_codebase_key(self, path: str) -> str:
        """Get the canonical key for a codebase path.

        Args:
            path: Path to the codebase

        Returns:
            Canonical absolute path string
        """
        return str(Path(path).resolve())

    def _detect_changed_files(self, codebase_path: str) -> Tuple[Set[str], Set[str], Set[str]]:
        """Detect added, modified, and deleted files in a codebase.

        Args:
            codebase_path: Path to the codebase

        Returns:
            Tuple of (added_files, modified_files, deleted_files) as sets of relative paths
        """
        codebase_key = self._get_codebase_key(codebase_path)
        root_path = Path(codebase_path)

        # Get current files in the codebase
        current_files = set()
        gitignore_patterns = self._get_gitignore_patterns(root_path)
        gitignore_patterns.extend([".git/", "venv/", self.index_dir_name + "/", ".gitignore"])

        for root, dirs, files in os.walk(codebase_path, topdown=True):
            current_path = Path(root)
            dirs[:] = [d for d in dirs if not self._is_ignored(current_path / d, gitignore_patterns, root_path)]

            for file in files:
                file_path = current_path / file
                if not self._is_ignored(file_path, gitignore_patterns, root_path):
                    relative_path = str(file_path.relative_to(root_path))
                    current_files.add(relative_path)

        # Get previously indexed files for this codebase
        previously_indexed = set()
        if codebase_key in self.indexed_codebases:
            previously_indexed = set(self.indexed_codebases[codebase_key].get("files", []))

        # Get metadata for this codebase
        codebase_metadata = self.file_metadata.get(codebase_key, {})

        added_files = set()
        modified_files = set()
        deleted_files = previously_indexed - current_files

        for file_path in current_files:
            abs_file_path = root_path / file_path
            current_mtime = self._get_file_mtime(abs_file_path)

            if file_path not in previously_indexed:
                added_files.add(file_path)
            elif file_path in codebase_metadata:
                stored_mtime = codebase_metadata[file_path].get("mtime", 0)
                if current_mtime > stored_mtime:
                    modified_files.add(file_path)
            else:
                # File exists but no metadata - treat as modified
                modified_files.add(file_path)

        # Include dependent files in the changed set
        all_modified_files = set(modified_files)
        all_added_files = set(added_files)

        # For each modified file, add its dependent files
        for modified_file in modified_files:
            dependent_files = self._get_dependent_files(modified_file, codebase_path)
            all_modified_files.update(dependent_files)
            print(f"File {modified_file} changed, will also re-index {len(dependent_files)} dependent files")

        return added_files, list(all_modified_files), deleted_files

    def _remove_deleted_embeddings(self, codebase_path: str, deleted_files: Set[str]) -> None:
        """Remove embeddings for deleted files from the index.

        Args:
            codebase_path: Path to the codebase
            deleted_files: Set of deleted file paths (relative)
        """
        if not deleted_files or not self.faiss_index:
            return

        codebase_key = self._get_codebase_key(codebase_path)
        indices_to_remove = []

        # Find FAISS indices for deleted files
        for faiss_id, file_path in self.file_paths_map.items():
            try:
                faiss_id_int = int(faiss_id)
                file_path_obj = Path(file_path)
                if file_path_obj.is_relative_to(codebase_path):
                    relative_path = str(file_path_obj.relative_to(codebase_path))
                    if relative_path in deleted_files:
                        indices_to_remove.append(faiss_id_int)
                        # Invalidate cache for deleted file
                        if self.cache:
                            self.cache.invalidate_file(file_path)
            except (ValueError, ValueError):
                continue

        if indices_to_remove:
            # Remove from FAISS index (this is a simplified approach)
            # In a production system, you'd want to rebuild the index more efficiently
            all_indices = set(range(self.faiss_index.ntotal))
            indices_to_keep = list(all_indices - set(indices_to_remove))

            if indices_to_keep:
                # Create new index with remaining vectors
                temp_index = faiss.IndexFlatL2(self.faiss_index.d)
                remaining_vectors = np.zeros((len(indices_to_keep), self.faiss_index.d), dtype=np.float32)

                for i, idx in enumerate(indices_to_keep):
                    self.faiss_index.reconstruct(idx, remaining_vectors[i])

                self.faiss_index = temp_index
                self.faiss_index.add(remaining_vectors)
            else:
                # All vectors removed, reset index
                self.faiss_index = faiss.IndexFlatL2(self.faiss_index.d)

            # Update file_paths_map
            new_file_paths_map = {}
            for faiss_id, file_path in self.file_paths_map.items():
                try:
                    faiss_id_int = int(faiss_id)
                    if faiss_id_int not in indices_to_remove:
                        # Adjust index for removed vectors
                        new_id = str(faiss_id_int - sum(1 for r in indices_to_remove if r < faiss_id_int))
                        new_file_paths_map[new_id] = file_path
                except ValueError:
                    continue
            self.file_paths_map = new_file_paths_map

    def _update_file_metadata(self, codebase_path: str, files: List[str]) -> None:
        """Update metadata for indexed files.

        Args:
            codebase_path: Path to the codebase
            files: List of relative file paths
        """
        codebase_key = self._get_codebase_key(codebase_path)
        root_path = Path(codebase_path)

        if codebase_key not in self.file_metadata:
            self.file_metadata[codebase_key] = {}

        for file_path in files:
            abs_file_path = root_path / file_path
            mtime = self._get_file_mtime(abs_file_path)
            self.file_metadata[codebase_key][file_path] = {
                "mtime": mtime,
                "indexed_at": datetime.now().isoformat()
            }

    def index_codebase_incremental(self, path: str, sentence_transformer_model, force_full: bool = False) -> Tuple[list[str], bool]:
        """Index a codebase incrementally, updating only changed files.

        Args:
            path: Path to the codebase directory
            sentence_transformer_model: Model for generating embeddings
            force_full: If True, force full re-indexing

        Returns:
            Tuple of (indexed_files, was_incremental) where was_incremental is True if incremental indexing was used
        """
        root_path = Path(path)
        if not root_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        codebase_key = self._get_codebase_key(path)

        # Check if we should do incremental indexing
        if force_full or codebase_key not in self.indexed_codebases:
            # Do full indexing
            indexed_files = self.index_codebase(path, sentence_transformer_model)
            self._update_file_metadata(path, indexed_files)
            return indexed_files, False

        # Detect changed files
        added_files, modified_files, deleted_files = self._detect_changed_files(path)

        if not added_files and not modified_files and not deleted_files:
            # No changes detected
            print(f"No changes detected in codebase at: {path}")
            existing_files = self.indexed_codebases[codebase_key].get("files", [])
            return existing_files, True

        print(f"Incremental indexing: {len(added_files)} added, {len(modified_files)} modified, {len(deleted_files)} deleted files")

        # Use batch processing for better performance
        return self._process_incremental_changes_batch(
            path, sentence_transformer_model, added_files, modified_files, deleted_files
        )

    def _process_incremental_changes_batch(self, path: str, sentence_transformer_model,
                                         added_files: Set[str], modified_files: Set[str],
                                         deleted_files: Set[str]) -> Tuple[list[str], bool]:
        """Process incremental changes in batches for better performance.

        Args:
            path: Path to the codebase directory
            sentence_transformer_model: Model for generating embeddings
            added_files: Set of added file paths
            modified_files: Set of modified file paths
            deleted_files: Set of deleted file paths

        Returns:
            Tuple of (indexed_files, was_incremental)
        """
        root_path = Path(path)
        codebase_key = self._get_codebase_key(path)

        # Remove embeddings for deleted files
        self._remove_deleted_embeddings(path, deleted_files)

        # Process added and modified files
        files_to_process = added_files | modified_files
        if not files_to_process:
            # Only deletions, update metadata and return
            self._update_file_metadata(path, [])
            self.indexed_codebases[codebase_key]["files"] = list(set(self.indexed_codebases[codebase_key].get("files", [])) - deleted_files)
            self._save_index()
            return list(set(self.indexed_codebases[codebase_key].get("files", [])) - deleted_files), True

        # Initialize FAISS if needed
        if self.faiss_index is None:
            embedding_size = sentence_transformer_model.get_sentence_embedding_dimension()
            # Use optimized index creation
            self.faiss_index = self.memory_manager.create_optimized_index(
                np.zeros((1, embedding_size), dtype=np.float32)  # Dummy data for initialization
            )

        gitignore_patterns = self._get_gitignore_patterns(root_path)
        gitignore_patterns.extend([".git/", "venv/", self.index_dir_name + "/", ".gitignore"])

        # Process files in batches for better performance
        batch_size = min(50, len(files_to_process))  # Adaptive batch size
        indexed_files = []
        all_new_embeddings = []
        all_new_file_paths = []

        current_faiss_id = self.faiss_index.ntotal if self.faiss_index else 0

        # Convert to list for batching
        files_list = list(files_to_process)

        # Determine if we should use parallel processing for this workload
        use_parallel = self._should_use_parallel_processing(len(files_list))

        for i in range(0, len(files_list), batch_size):
            batch_files = files_list[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(files_list) + batch_size - 1)//batch_size} ({len(batch_files)} files)")

            if use_parallel:
                batch_embeddings, batch_file_paths, batch_indexed_files = self._process_file_batch_parallel(
                    batch_files, root_path, gitignore_patterns, sentence_transformer_model
                )
            else:
                batch_embeddings, batch_file_paths, batch_indexed_files = self._process_file_batch(
                    batch_files, root_path, gitignore_patterns, sentence_transformer_model
                )

            all_new_embeddings.extend(batch_embeddings)
            all_new_file_paths.extend(batch_file_paths)
            indexed_files.extend(batch_indexed_files)

            # Periodic memory cleanup during large indexing operations
            if len(all_new_embeddings) > 1000:
                self.memory_manager.cleanup()

        # Add all new embeddings to FAISS in one operation
        if all_new_embeddings:
            embeddings_array = np.array(all_new_embeddings).astype("float32")
            self.faiss_index.add(embeddings_array)

            # Update file paths map
            for i, fp in enumerate(all_new_file_paths):
                self.file_paths_map[str(current_faiss_id + i)] = fp

        # Update codebase metadata
        existing_files = set(self.indexed_codebases[codebase_key].get("files", []))
        updated_files = (existing_files - deleted_files) | set(indexed_files)
        self.indexed_codebases[codebase_key] = {
            "status": "indexed",
            "files": list(updated_files),
        }

        # Update file metadata
        self._update_file_metadata(path, indexed_files)

        # Update dependency graph for new/modified files
        self._update_dependency_graph(path, indexed_files)

        self._save_index()

        print(f"Incrementally indexed {len(indexed_files)} files in codebase at: {path}")
        return list(updated_files), True

    def _process_file_batch(self, file_paths: List[str], root_path: Path,
                          gitignore_patterns: List[str], sentence_transformer_model) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """Process a batch of files for embedding generation.

        Args:
            file_paths: List of relative file paths to process
            root_path: Root path of the codebase
            gitignore_patterns: Patterns for files to ignore
            sentence_transformer_model: Model for generating embeddings

        Returns:
            Tuple of (embeddings, file_paths, indexed_files)
        """
        embeddings = []
        file_paths_list = []
        indexed_files = []

        for file_path in file_paths:
            abs_file_path = root_path / file_path
            if not self._is_ignored(abs_file_path, gitignore_patterns, root_path):
                try:
                    # Use chunked processing for large files
                    chunks = chunk_file(str(abs_file_path))

                    if chunks:
                        print(f"Processing {len(chunks)} chunks for {abs_file_path}")

                        for i, chunk in enumerate(chunks):
                            # Create chunk identifier
                            chunk_id = f"{file_path}:chunk_{i}"

                            # Check cache for existing embedding
                            embedding = None
                            cache_hit = False

                            if self.cache:
                                embedding, cache_hit = self.cache.get_embedding(chunk_id, chunk.content)

                            if not cache_hit:
                                # Generate new embedding for chunk
                                embedding = sentence_transformer_model.encode(chunk.content)

                                # Store in cache
                                if self.cache:
                                    self.cache.store_embedding(chunk_id, chunk.content, embedding)

                            embeddings.append(embedding)
                            file_paths_list.append(str(abs_file_path.resolve()))
                            indexed_files.append(file_path)

                            if cache_hit:
                                print(f"Cache hit for chunk {i} of {abs_file_path}")
                            else:
                                print(f"Generated embedding for chunk {i} of {abs_file_path}")
                    else:
                        # Fallback to whole file processing if chunking fails
                        with open(abs_file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Check cache for existing embedding
                        embedding = None
                        cache_hit = False

                        if self.cache:
                            embedding, cache_hit = self.cache.get_embedding(str(abs_file_path), content)

                        if not cache_hit:
                            # Generate new embedding
                            embedding = sentence_transformer_model.encode(content)

                            # Store in cache
                            if self.cache:
                                self.cache.store_embedding(str(abs_file_path), content, embedding)

                        embeddings.append(embedding)
                        file_paths_list.append(str(abs_file_path.resolve()))
                        indexed_files.append(file_path)

                        if cache_hit:
                            print(f"Cache hit for {abs_file_path}")
                        else:
                            print(f"Generated new embedding for {abs_file_path}")

                except Exception as e:
                    print(f"Warning: Could not process file {abs_file_path} for embedding: {e}")
                    continue

        return embeddings, file_paths_list, indexed_files

    def _analyze_file_dependencies(self, file_path: str, root_path: Path) -> Set[str]:
        """Analyze dependencies of a Python file by parsing import statements.

        Args:
            file_path: Relative path to the file
            root_path: Root path of the codebase

        Returns:
            Set of relative file paths that this file depends on
        """
        dependencies = set()
        abs_file_path = root_path / file_path

        # Only analyze Python files
        if not file_path.endswith('.py'):
            return dependencies

        try:
            with open(abs_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse AST to find import statements
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        # Try to resolve module to file path
                        dep_path = self._resolve_module_to_path(module_name, root_path)
                        if dep_path:
                            dependencies.add(dep_path)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        # Try to resolve module to file path
                        dep_path = self._resolve_module_to_path(module_name, root_path)
                        if dep_path:
                            dependencies.add(dep_path)

        except Exception as e:
            # If we can't parse the file, return empty dependencies
            print(f"Warning: Could not analyze dependencies for {file_path}: {e}")

        return dependencies

    def _resolve_module_to_path(self, module_name: str, root_path: Path) -> Optional[str]:
        """Resolve a Python module name to a relative file path.

        Args:
            module_name: Name of the Python module
            root_path: Root path of the codebase

        Returns:
            Relative file path if found, None otherwise
        """
        # Try different possible file extensions and locations
        possible_paths = [
            f"{module_name}.py",
            f"{module_name}/__init__.py",
            f"{module_name.replace('.', '/')}.py",
            f"{module_name.replace('.', '/')}/__init__.py"
        ]

        for path_str in possible_paths:
            candidate_path = root_path / path_str
            if candidate_path.exists():
                # Return relative path from root
                try:
                    return str(candidate_path.relative_to(root_path))
                except ValueError:
                    continue

        return None

    def _build_dependency_graph(self, codebase_path: str) -> None:
        """Build dependency graph for all files in a codebase.

        Args:
            codebase_path: Path to the codebase
        """
        root_path = Path(codebase_path)
        codebase_key = self._get_codebase_key(codebase_path)

        # Initialize dependency graphs
        self._dependency_graph[codebase_key] = {}
        self._reverse_dependencies[codebase_key] = {}

        # Get all indexed files for this codebase
        if codebase_key not in self.indexed_codebases:
            return

        indexed_files = self.indexed_codebases[codebase_key].get("files", [])

        for file_path in indexed_files:
            if file_path.endswith('.py'):
                # Analyze dependencies for this file
                dependencies = self._analyze_file_dependencies(file_path, root_path)

                # Update dependency graphs
                if file_path not in self._dependency_graph[codebase_key]:
                    self._dependency_graph[codebase_key][file_path] = set()
                if file_path not in self._reverse_dependencies[codebase_key]:
                    self._reverse_dependencies[codebase_key][file_path] = set()

                for dep in dependencies:
                    # Add forward dependency (file depends on dep)
                    self._reverse_dependencies[codebase_key][file_path].add(dep)

                    # Add reverse dependency (dep is depended upon by file)
                    if dep not in self._dependency_graph[codebase_key]:
                        self._dependency_graph[codebase_key][dep] = set()
                    self._dependency_graph[codebase_key][dep].add(file_path)

    def _get_dependent_files(self, file_path: str, codebase_path: str) -> Set[str]:
        """Get all files that depend on the given file (directly or indirectly).

        Args:
            file_path: Relative path to the file
            codebase_path: Path to the codebase

        Returns:
            Set of relative file paths that depend on the given file
        """
        codebase_key = self._get_codebase_key(codebase_path)

        if codebase_key not in self._dependency_graph:
            self._build_dependency_graph(codebase_path)

        if codebase_key not in self._dependency_graph:
            return set()

        # Use BFS to find all dependent files
        dependent_files = set()
        queue = [file_path]
        visited = set()

        while queue:
            current_file = queue.pop(0)
            if current_file in visited:
                continue
            visited.add(current_file)

            # Add direct dependents
            if current_file in self._dependency_graph[codebase_key]:
                for dependent in self._dependency_graph[codebase_key][current_file]:
                    if dependent not in visited:
                        dependent_files.add(dependent)
                        queue.append(dependent)

        return dependent_files

    def _update_dependency_graph(self, codebase_path: str, changed_files: List[str]) -> None:
        """Update dependency graph for changed files.

        Args:
            codebase_path: Path to the codebase
            changed_files: List of files that have been changed/added
        """
        root_path = Path(codebase_path)
        codebase_key = self._get_codebase_key(codebase_path)

        # Initialize dependency graphs if they don't exist
        if codebase_key not in self._dependency_graph:
            self._dependency_graph[codebase_key] = {}
        if codebase_key not in self._reverse_dependencies:
            self._reverse_dependencies[codebase_key] = {}

        # Update dependencies for changed files
        for file_path in changed_files:
            if file_path.endswith('.py'):
                # Remove old dependencies for this file
                if file_path in self._reverse_dependencies[codebase_key]:
                    old_deps = self._reverse_dependencies[codebase_key][file_path].copy()
                    for old_dep in old_deps:
                        if old_dep in self._dependency_graph[codebase_key]:
                            self._dependency_graph[codebase_key][old_dep].discard(file_path)

                # Analyze new dependencies
                new_dependencies = self._analyze_file_dependencies(file_path, root_path)

                # Update dependency graphs
                self._reverse_dependencies[codebase_key][file_path] = new_dependencies

                for dep in new_dependencies:
                    if dep not in self._dependency_graph[codebase_key]:
                        self._dependency_graph[codebase_key][dep] = set()
                    self._dependency_graph[codebase_key][dep].add(file_path)

    def _process_files_parallel(self, file_paths: List[str], root_path: Path, sentence_transformer_model) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """Process multiple files in parallel for full indexing.

        Args:
            file_paths: List of relative file paths to process
            root_path: Root path of the codebase
            sentence_transformer_model: Model for generating embeddings

        Returns:
            Tuple of (embeddings, file_paths, indexed_files)
        """
        # Get thread pool
        executor = self._get_thread_pool()

        # Submit all file processing tasks
        future_to_file = {}
        for file_path in file_paths:
            abs_file_path = root_path / file_path
            future = executor.submit(self._process_single_file, abs_file_path, file_path, sentence_transformer_model)
            future_to_file[future] = file_path

        # Collect results as they complete
        all_embeddings = []
        all_file_paths = []
        all_indexed_files = []

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                embeddings, file_paths_list, indexed_files = future.result()
                all_embeddings.extend(embeddings)
                all_file_paths.extend(file_paths_list)
                all_indexed_files.extend(indexed_files)
            except Exception as e:
                print(f"Warning: Failed to process file {file_path}: {e}")
                continue

        return all_embeddings, all_file_paths, all_indexed_files

    def _get_thread_pool(self, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """Get or create a thread pool for parallel processing.

        Args:
            max_workers: Maximum number of worker threads

        Returns:
            ThreadPoolExecutor instance
        """
        with self._executor_lock:
            if self._executor is None or self._executor._shutdown:
                # Determine optimal number of workers based on system and workload
                if max_workers is None:
                    import multiprocessing
                    cpu_count = multiprocessing.cpu_count()
                    # Use 75% of available CPUs, minimum 2, maximum 8
                    max_workers = max(2, min(8, int(cpu_count * 0.75)))

                self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="IndexingWorker")
            return self._executor

    def _should_use_parallel_processing(self, num_files: int, file_sizes: Optional[List[int]] = None) -> bool:
        """Determine if parallel processing would be beneficial.

        Args:
            num_files: Number of files to process
            file_sizes: Optional list of file sizes for more accurate decision

        Returns:
            True if parallel processing should be used
        """
        # Use parallel processing for more than 10 files or large files
        if num_files > 10:
            return True

        # Check for large files if sizes are provided
        if file_sizes:
            total_size_mb = sum(file_sizes) / (1024 * 1024)
            if total_size_mb > 50:  # Over 50MB total
                return True

        return False

    def _process_file_batch_parallel(self, file_paths: List[str], root_path: Path,
                                   gitignore_patterns: List[str], sentence_transformer_model) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """Process a batch of files in parallel for better performance.

        Args:
            file_paths: List of relative file paths to process
            root_path: Root path of the codebase
            gitignore_patterns: Patterns for files to ignore
            sentence_transformer_model: Model for generating embeddings

        Returns:
            Tuple of (embeddings, file_paths, indexed_files)
        """
        if not self._should_use_parallel_processing(len(file_paths)):
            # Fall back to sequential processing for small batches
            return self._process_file_batch(file_paths, root_path, gitignore_patterns, sentence_transformer_model)

        print(f"Using parallel processing for {len(file_paths)} files")

        # Get thread pool
        executor = self._get_thread_pool()

        # Submit all file processing tasks
        future_to_file = {}
        for file_path in file_paths:
            abs_file_path = root_path / file_path
            if not self._is_ignored(abs_file_path, gitignore_patterns, root_path):
                future = executor.submit(self._process_single_file, abs_file_path, file_path, sentence_transformer_model)
                future_to_file[future] = file_path

        # Collect results as they complete
        all_embeddings = []
        all_file_paths = []
        all_indexed_files = []

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                embeddings, file_paths_list, indexed_files = future.result()
                all_embeddings.extend(embeddings)
                all_file_paths.extend(file_paths_list)
                all_indexed_files.extend(indexed_files)
            except Exception as e:
                print(f"Warning: Failed to process file {file_path}: {e}")
                continue

        return all_embeddings, all_file_paths, all_indexed_files

    def _process_single_file(self, abs_file_path: Path, relative_file_path: str, sentence_transformer_model) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """Process a single file for embedding generation (used in parallel processing).

        Args:
            abs_file_path: Absolute path to the file
            relative_file_path: Relative path from codebase root
            sentence_transformer_model: Model for generating embeddings

        Returns:
            Tuple of (embeddings, file_paths, indexed_files)
        """
        embeddings = []
        file_paths_list = []
        indexed_files = []

        try:
            # Use chunked processing for large files
            chunks = chunk_file(str(abs_file_path))

            if chunks:
                for i, chunk in enumerate(chunks):
                    # Create chunk identifier
                    chunk_id = f"{relative_file_path}:chunk_{i}"

                    # Check cache for existing embedding
                    embedding = None
                    cache_hit = False

                    if self.cache:
                        embedding, cache_hit = self.cache.get_embedding(chunk_id, chunk.content)

                    if not cache_hit:
                        # Generate new embedding for chunk
                        embedding = sentence_transformer_model.encode(chunk.content)

                        # Store in cache
                        if self.cache:
                            self.cache.store_embedding(chunk_id, chunk.content, embedding)

                    embeddings.append(embedding)
                    file_paths_list.append(str(abs_file_path.resolve()))
                    indexed_files.append(relative_file_path)
            else:
                # Fallback to whole file processing if chunking fails
                with open(abs_file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check cache for existing embedding
                embedding = None
                cache_hit = False

                if self.cache:
                    embedding, cache_hit = self.cache.get_embedding(str(abs_file_path), content)

                if not cache_hit:
                    # Generate new embedding
                    embedding = sentence_transformer_model.encode(content)

                    # Store in cache
                    if self.cache:
                        self.cache.store_embedding(str(abs_file_path), content, embedding)

                embeddings.append(embedding)
                file_paths_list.append(str(abs_file_path.resolve()))
                indexed_files.append(relative_file_path)

        except Exception as e:
            print(f"Warning: Could not process file {abs_file_path} for embedding: {e}")

        return embeddings, file_paths_list, indexed_files

    def index_codebase(self, path: str, sentence_transformer_model) -> list[str]:
        """
        Indexa un directorio de codebase, respetando las reglas de .gitignore.

        Args:
            path (str): Ruta al directorio de la codebase.
            sentence_transformer_model: Modelo para generar embeddings.

        Returns:
            list[str]: Lista de rutas de archivos indexados.
        """
        # Esta función será una versión adaptada de la de codebase_manager.py
        # Se necesitará ajustar las referencias a self.sentence_transformer_model, etc.
        root_path = Path(path)
        if not root_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        gitignore_patterns = self._get_gitignore_patterns(root_path)
        # Explicitly add .git/ and venv/ to ignore patterns
        gitignore_patterns.append(".git/")
        gitignore_patterns.append("venv/")
        gitignore_patterns.append(self.index_dir_name + "/")
        gitignore_patterns.append(".gitignore")

        indexed_files = []
        new_embeddings = []
        new_file_paths = []
        # Inicializar el modelo y FAISS si es necesario
        self.sentence_transformer_model = sentence_transformer_model
        if self.faiss_index is None:
            embedding_size = (
                self.sentence_transformer_model.get_sentence_embedding_dimension()
            )
            # Use optimized index creation
            self.faiss_index = self.memory_manager.create_optimized_index(
                np.zeros((1, embedding_size), dtype=np.float32)  # Dummy data for initialization
            )

        current_faiss_id = self.faiss_index.ntotal if self.faiss_index else 0

        # Collect all files to process first
        all_files_to_process = []
        for root, dirs, files in os.walk(path, topdown=True):
            current_path = Path(root)

            # Filter directories in-place to prevent os.walk from
            # descending into ignored ones
            dirs[:] = [
                d
                for d in dirs
                if not self._is_ignored(current_path / d, gitignore_patterns, root_path)
            ]

            for file in files:
                file_path = current_path / file
                if not self._is_ignored(file_path, gitignore_patterns, root_path):
                    all_files_to_process.append(str(file_path.relative_to(root_path)))

        # Determine if we should use parallel processing
        use_parallel = self._should_use_parallel_processing(len(all_files_to_process))

        if use_parallel:
            print(f"Using parallel processing for {len(all_files_to_process)} files")
            # Process files in parallel
            new_embeddings, new_file_paths, indexed_files = self._process_files_parallel(
                all_files_to_process, root_path, sentence_transformer_model
            )
        else:
            # Process files sequentially
            for relative_file_path in all_files_to_process:
                file_path = root_path / relative_file_path
                try:
                    # Use chunked processing for large files
                    chunks = chunk_file(str(file_path))

                    if chunks:
                        print(f"Processing {len(chunks)} chunks for {file_path}")

                        for i, chunk in enumerate(chunks):
                            # Create chunk identifier
                            chunk_id = f"{relative_file_path}:chunk_{i}"

                            # Check cache for existing embedding
                            embedding = None
                            cache_hit = False

                            if self.cache:
                                embedding, cache_hit = self.cache.get_embedding(chunk_id, chunk.content)

                            if not cache_hit:
                                # Generate new embedding for chunk
                                embedding = self.sentence_transformer_model.encode(chunk.content)

                                # Store in cache
                                if self.cache:
                                    self.cache.store_embedding(chunk_id, chunk.content, embedding)

                            new_embeddings.append(embedding)
                            new_file_paths.append(str(file_path.resolve()))
                            indexed_files.append(relative_file_path)

                            if cache_hit:
                                print(f"Cache hit for chunk {i} of {file_path}")
                            else:
                                print(f"Generated embedding for chunk {i} of {file_path}")
                    else:
                        # Fallback to whole file processing if chunking fails
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Check cache for existing embedding
                        embedding = None
                        cache_hit = False

                        if self.cache:
                            embedding, cache_hit = self.cache.get_embedding(str(file_path), content)

                        if not cache_hit:
                            # Generate new embedding
                            embedding = self.sentence_transformer_model.encode(content)

                            # Store in cache
                            if self.cache:
                                self.cache.store_embedding(str(file_path), content, embedding)

                        new_embeddings.append(embedding)
                        new_file_paths.append(str(file_path.resolve()))
                        indexed_files.append(relative_file_path)

                        if cache_hit:
                            print(f"Cache hit for {file_path}")
                        else:
                            print(f"Generated new embedding for {file_path}")

                except Exception as e:
                    print(
                        f"Warning: Could not process file {file_path} "
                        f"for embedding: {e}"
                    )
                    continue

        if new_embeddings:
            # Add new embeddings to FAISS index
            self.faiss_index.add(np.array(new_embeddings).astype("float32"))
            # Update file_paths_map
            for i, fp in enumerate(new_file_paths):
                self.file_paths_map[str(current_faiss_id + i)] = fp

        abs_path_key = str(root_path.resolve())
        self.indexed_codebases[abs_path_key] = {
            "status": "indexed",
            "files": indexed_files,
        }

        # Update file metadata for incremental indexing support
        self._update_file_metadata(path, indexed_files)

        # Build initial dependency graph
        self._build_dependency_graph(path)

        self._save_index()

        print(f"Indexed {len(indexed_files)} files in codebase at: {path}")
        # print(f"DEBUG: Indexed files: {indexed_files}")  # Added logging
        return indexed_files

    def get_indexing_stats(self, codebase_path: str = None) -> dict:
        """Get indexing statistics including memory usage and performance metrics.

        Args:
            codebase_path: Optional path to get stats for specific codebase

        Returns:
            dict: Statistics about indexing performance and memory usage
        """
        stats = {
            "memory_stats": self.memory_manager.get_memory_stats(),
            "index_stats": {},
            "chunking_stats": {},
            "cache_stats": self.cache.get_stats() if self.cache else {}
        }

        if self.faiss_index:
            stats["index_stats"] = {
                "total_vectors": self.faiss_index.ntotal,
                "dimension": self.faiss_index.d,
                "index_type": type(self.faiss_index).__name__,
                "is_trained": hasattr(self.faiss_index, 'is_trained') and self.faiss_index.is_trained
            }

        if codebase_path:
            abs_path = str(Path(codebase_path).resolve())
            if abs_path in self.indexed_codebases:
                codebase_info = self.indexed_codebases[abs_path]
                stats["codebase_stats"] = {
                    "total_files": len(codebase_info.get("files", [])),
                    "status": codebase_info.get("status", "unknown")
                }

        return stats

    def optimize_index(self, codebase_path: str = None) -> dict:
        """Optimize the FAISS index for better performance.

        Args:
            codebase_path: Optional path to optimize index for specific codebase

        Returns:
            dict: Optimization results
        """
        # Use the comprehensive optimization method
        return self.optimize_index_comprehensive(codebase_path)

    def get_index_health(self, codebase_path: str = None) -> Dict[str, Any]:
        """Get comprehensive index health statistics.

        Args:
            codebase_path: Optional path to get health for specific codebase

        Returns:
            Dictionary with index health metrics
        """
        health_stats = self._analyze_index_health()

        # Add codebase-specific information if provided
        if codebase_path:
            abs_path = str(Path(codebase_path).resolve())
            if abs_path in self.indexed_codebases:
                codebase_info = self.indexed_codebases[abs_path]
                health_stats["codebase_info"] = {
                    "total_files": len(codebase_info.get("files", [])),
                    "status": codebase_info.get("status", "unknown")
                }

        return health_stats

    def cleanup(self) -> None:
        """Clean up resources used by the indexing manager."""
        with self._executor_lock:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

        # Save persistent cache if available
        if self.cache:
            self.cache.save_persistent_cache()

    def _analyze_index_health(self) -> Dict[str, Any]:
        """Analyze the health and fragmentation of the FAISS index.

        Returns:
            Dictionary with index health metrics
        """
        if not self.faiss_index:
            return {"healthy": False, "reason": "no index"}

        health_stats = {
            "total_vectors": self.faiss_index.ntotal,
            "dimension": self.faiss_index.d,
            "index_type": type(self.faiss_index).__name__,
            "healthy": True,
            "fragmentation_ratio": 0.0,
            "needs_optimization": False,
            "recommendations": []
        }

        # Check for fragmentation (simplified - in practice you'd analyze the index structure)
        if hasattr(self.faiss_index, 'ntotal') and hasattr(self.faiss_index, 'd'):
            # Estimate fragmentation based on vector count and memory usage
            estimated_memory_mb = (self.faiss_index.ntotal * self.faiss_index.d * 4) / (1024 * 1024)
            health_stats["estimated_memory_mb"] = estimated_memory_mb

            # If we have significantly more vectors than expected for the memory usage,
            # it might indicate fragmentation
            expected_vectors = (estimated_memory_mb * 1024 * 1024) / (self.faiss_index.d * 4 * 1.5)  # 1.5x overhead
            if self.faiss_index.ntotal > expected_vectors * 1.2:  # 20% threshold
                health_stats["fragmentation_ratio"] = self.faiss_index.ntotal / expected_vectors
                health_stats["needs_optimization"] = True
                health_stats["recommendations"].append("High fragmentation detected - consider rebuilding index")

        # Check if index needs training (for IVF indexes)
        if hasattr(self.faiss_index, 'is_trained'):
            health_stats["is_trained"] = self.faiss_index.is_trained
            if not self.faiss_index.is_trained:
                health_stats["needs_optimization"] = True
                health_stats["recommendations"].append("Index not trained - needs training")

        # Performance recommendations
        if self.faiss_index.ntotal > 10000:
            health_stats["recommendations"].append("Large index - consider using IVF or HNSW for better performance")

        return health_stats

    def _rebuild_index_optimized(self, embeddings: np.ndarray) -> None:
        """Rebuild the FAISS index with optimal settings based on data characteristics.

        Args:
            embeddings: Numpy array of embeddings to index
        """
        if embeddings.shape[0] == 0:
            return

        # Determine optimal index type based on data size and characteristics
        n_vectors = embeddings.shape[0]
        dimension = embeddings.shape[1]

        if n_vectors < 1000:
            # Small dataset - use simple flat index
            self.faiss_index = faiss.IndexFlatL2(dimension)
        elif n_vectors < 10000:
            # Medium dataset - use IVF with small nlist
            nlist = min(100, max(4, n_vectors // 39))
            quantizer = faiss.IndexFlatL2(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            # Large dataset - use IVF with optimized parameters
            nlist = min(1024, max(100, n_vectors // 39))
            quantizer = faiss.IndexFlatL2(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        # Train the index if necessary
        if hasattr(self.faiss_index, 'is_trained') and not self.faiss_index.is_trained:
            print(f"Training index with {n_vectors} vectors...")
            self.faiss_index.train(embeddings)

        # Add vectors to the index
        self.faiss_index.add(embeddings)

        print(f"Rebuilt index: {type(self.faiss_index).__name__} with {n_vectors} vectors")

    def optimize_index_comprehensive(self, codebase_path: str = None) -> Dict[str, Any]:
        """Perform comprehensive index optimization including defragmentation and rebuilding.

        Args:
            codebase_path: Optional path to optimize index for specific codebase

        Returns:
            Dictionary with optimization results
        """
        if not self.faiss_index:
            return {"success": False, "message": "No index to optimize"}

        print("Starting comprehensive index optimization...")

        # Analyze current index health
        health_stats = self._analyze_index_health()

        if not health_stats["needs_optimization"]:
            return {
                "success": True,
                "message": "Index is already optimized",
                "health_stats": health_stats
            }

        try:
            # Trigger memory cleanup before optimization
            self.memory_manager.cleanup()

            # Extract all current embeddings
            print("Extracting embeddings for optimization...")
            all_embeddings = []
            valid_ids = []

            for faiss_id in range(self.faiss_index.ntotal):
                try:
                    # Reconstruct the vector
                    vector = np.zeros(self.faiss_index.d, dtype=np.float32)
                    self.faiss_index.reconstruct(faiss_id, vector)
                    all_embeddings.append(vector)
                    valid_ids.append(faiss_id)
                except Exception as e:
                    print(f"Warning: Could not reconstruct vector {faiss_id}: {e}")
                    continue

            if not all_embeddings:
                return {"success": False, "message": "No valid embeddings found"}

            embeddings_array = np.array(all_embeddings)

            # Rebuild index with optimal settings
            print("Rebuilding index with optimal settings...")
            self._rebuild_index_optimized(embeddings_array)

            # Update file paths map (remap IDs)
            new_file_paths_map = {}
            for i, old_id in enumerate(valid_ids):
                if str(old_id) in self.file_paths_map:
                    new_file_paths_map[str(i)] = self.file_paths_map[str(old_id)]

            self.file_paths_map = new_file_paths_map

            # Save the optimized index
            self._save_index()

            optimization_results = {
                "success": True,
                "message": "Index optimization completed successfully",
                "vectors_optimized": len(all_embeddings),
                "new_index_type": type(self.faiss_index).__name__,
                "health_improvements": health_stats["recommendations"],
                "before_optimization": health_stats
            }

            print(f"Index optimization completed: {optimization_results}")
            return optimization_results

        except Exception as e:
            return {
                "success": False,
                "message": f"Optimization failed: {e}",
                "health_stats": health_stats
            }

    def defragment_index(self, threshold: float = 0.2) -> Dict[str, Any]:
        """Defragment the index by removing gaps and rebuilding if fragmentation is high.

        Args:
            threshold: Fragmentation threshold above which to defragment (0.0 to 1.0)

        Returns:
            Dictionary with defragmentation results
        """
        health_stats = self._analyze_index_health()

        if not health_stats["needs_optimization"] or health_stats["fragmentation_ratio"] < (1.0 + threshold):
            return {
                "defragmented": False,
                "reason": "Fragmentation below threshold",
                "fragmentation_ratio": health_stats["fragmentation_ratio"]
            }
    
        def compress_index(self, compression_type: str = "auto", target_memory_mb: Optional[int] = None) -> Dict[str, Any]:
            """Compress the FAISS index to reduce memory usage.
    
            Args:
                compression_type: Type of compression ("auto", "pq", "ivf_pq", "scalar_quant")
                target_memory_mb: Target memory usage in MB (optional)
    
            Returns:
                Dictionary with compression results
            """
            if not self.faiss_index:
                return {"success": False, "message": "No index to compress"}
    
            print(f"Starting index compression (type: {compression_type})...")
    
            try:
                # Extract current embeddings
                print("Extracting embeddings for compression...")
                embeddings = []
                for faiss_id in range(self.faiss_index.ntotal):
                    try:
                        vector = np.zeros(self.faiss_index.d, dtype=np.float32)
                        self.faiss_index.reconstruct(faiss_id, vector)
                        embeddings.append(vector)
                    except Exception as e:
                        print(f"Warning: Could not reconstruct vector {faiss_id}: {e}")
                        continue
    
                if not embeddings:
                    return {"success": False, "message": "No valid embeddings found"}
    
                embeddings_array = np.array(embeddings)
                n_vectors = embeddings_array.shape[0]
                dimension = embeddings_array.shape[1]
    
                # Determine optimal compression based on data characteristics
                if compression_type == "auto":
                    if n_vectors > 10000:
                        compression_type = "ivf_pq"
                    elif dimension > 384:
                        compression_type = "pq"
                    else:
                        compression_type = "scalar_quant"
    
                original_memory_mb = (n_vectors * dimension * 4) / (1024 * 1024)
    
                # Create compressed index
                if compression_type == "pq":
                    # Product Quantization
                    m = min(dimension // 4, 64)  # Number of sub-quantizers
                    nbits = 8  # Bits per sub-quantizer
                    self.faiss_index = faiss.IndexPQ(dimension, m, nbits)
                elif compression_type == "ivf_pq":
                    # IVF + PQ
                    nlist = min(1024, max(100, n_vectors // 39))
                    m = min(dimension // 4, 64)
                    nbits = 8
                    quantizer = faiss.IndexFlatL2(dimension)
                    self.faiss_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
                elif compression_type == "scalar_quant":
                    # Scalar quantization
                    self.faiss_index = faiss.IndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_8bit)
                else:
                    return {"success": False, "message": f"Unknown compression type: {compression_type}"}
    
                # Train the compressed index if necessary
                if hasattr(self.faiss_index, 'is_trained') and not self.faiss_index.is_trained:
                    print("Training compressed index...")
                    self.faiss_index.train(embeddings_array)
    
                # Add vectors to compressed index
                self.faiss_index.add(embeddings_array)
    
                # Calculate compression statistics
                compressed_memory_mb = self._estimate_index_memory_usage()
    
                return {
                    "success": True,
                    "message": f"Index compressed using {compression_type}",
                    "compression_type": compression_type,
                    "original_memory_mb": original_memory_mb,
                    "compressed_memory_mb": compressed_memory_mb,
                    "compression_ratio": original_memory_mb / compressed_memory_mb if compressed_memory_mb > 0 else 0,
                    "vectors_compressed": n_vectors
                }
    
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Compression failed: {e}",
                    "compression_type": compression_type
                }
    
        def _estimate_index_memory_usage(self) -> float:
            """Estimate memory usage of the current FAISS index in MB.
    
            Returns:
                Estimated memory usage in MB
            """
            if not self.faiss_index:
                return 0
    
            try:
                # Rough estimation based on index type and parameters
                n_vectors = self.faiss_index.ntotal
                dimension = self.faiss_index.d
    
                if hasattr(self.faiss_index, 'pq'):
                    # PQ index
                    m = self.faiss_index.pq.M
                    nbits = self.faiss_index.pq.nbits
                    bytes_per_vector = (m * nbits) / 8
                elif hasattr(self.faiss_index, 'sq'):
                    # Scalar quantizer
                    bytes_per_vector = 1  # 8-bit quantization
                else:
                    # Flat index (no compression)
                    bytes_per_vector = dimension * 4  # float32
    
                # Add overhead for index structure
                total_bytes = n_vectors * bytes_per_vector * 1.2  # 20% overhead
                return total_bytes / (1024 * 1024)
    
            except Exception:
                # Fallback estimation
                return (self.faiss_index.ntotal * self.faiss_index.d * 4 * 1.2) / (1024 * 1024)
    
        def optimize_index_for_memory(self, target_memory_mb: int = 500) -> Dict[str, Any]:
            """Optimize index for specific memory target.
    
            Args:
                target_memory_mb: Target memory usage in MB
    
            Returns:
                Dictionary with optimization results
            """
            if not self.faiss_index:
                return {"success": False, "message": "No index to optimize"}
    
            current_memory_mb = self._estimate_index_memory_usage()
    
            if current_memory_mb <= target_memory_mb:
                return {
                    "success": True,
                    "message": "Index already within memory target",
                    "current_memory_mb": current_memory_mb,
                    "target_memory_mb": target_memory_mb
                }
    
            # Try different compression strategies
            strategies = [
                ("scalar_quant", "Scalar Quantization (8-bit)"),
                ("pq", "Product Quantization"),
                ("ivf_pq", "IVF + Product Quantization")
            ]
    
            for compression_type, description in strategies:
                try:
                    result = self.compress_index(compression_type)
                    if result["success"] and result["compressed_memory_mb"] <= target_memory_mb:
                        result["strategy_used"] = description
                        return result
                except Exception as e:
                    print(f"Warning: {description} failed: {e}")
                    continue
    
            return {
                "success": False,
                "message": f"Could not achieve target memory usage of {target_memory_mb}MB",
                "current_memory_mb": current_memory_mb,
                "target_memory_mb": target_memory_mb
            }

        print(f"Defragmenting index (fragmentation: {health_stats['fragmentation_ratio']:.2f})...")

        # Perform comprehensive optimization
        return self.optimize_index_comprehensive()

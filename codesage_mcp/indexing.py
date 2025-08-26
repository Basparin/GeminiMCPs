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
        self.indexed_codebases = {}
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

        # Inicializar FAISS
        if self.faiss_index_file.exists():
            try:
                self.faiss_index = faiss.read_index(str(self.faiss_index_file))
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
            self.faiss_index = faiss.IndexFlatL2(embedding_size)

        current_faiss_id = self.faiss_index.ntotal if self.faiss_index else 0

        for root, dirs, files in os.walk(path, topdown=True):
            current_path = Path(root)

            # print(f"DEBUG: Current directory: {current_path}") # Opcional: Logging
            # print(f"DEBUG: Dirs before filtering: {dirs}") # Opcional: Logging

            # Filter directories in-place to prevent os.walk from
            # descending into ignored ones
            dirs[:] = [
                d
                for d in dirs
                if not self._is_ignored(current_path / d, gitignore_patterns, root_path)
            ]
            # print(f"DEBUG: Dirs after filtering: {dirs}") # Opcional: Logging

            for file in files:
                file_path = current_path / file
                # print(f"DEBUG: Checking file: {file_path}") # Opcional: Logging
                if not self._is_ignored(file_path, gitignore_patterns, root_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        embedding = self.sentence_transformer_model.encode(content)
                        new_embeddings.append(embedding)
                        new_file_paths.append(str(file_path.resolve()))
                        indexed_files.append(str(file_path.relative_to(root_path)))
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
        self._save_index()

        print(f"Indexed {len(indexed_files)} files in codebase at: {path}")
        # print(f"DEBUG: Indexed files: {indexed_files}")  # Added logging
        return indexed_files

"""
Searching Module for CodeSage MCP Server.

This module contains the logic for searching and analyzing similarity within indexed codebases.
It provides the SearchingManager class, which is responsible for:
- Searching for text patterns.
- Performing semantic searches using FAISS embeddings.
- Finding duplicate code sections.
- Providing file structure overviews.

Classes:
    SearchingManager: Manages codebase search and similarity analysis.
"""

import re
import fnmatch
import numpy as np
from pathlib import Path

# Import caching system
from .config import ENABLE_CACHING
from .cache import get_cache_instance


class SearchingManager:
    """Manages codebase search and similarity analysis.

    This class is responsible for searching for text patterns, performing semantic searches
    using FAISS embeddings, and finding duplicate code sections within indexed codebases.

    Attributes:
        indexing_manager (IndexingManager): Reference to the IndexingManager for accessing indexed data.
    """

    def __init__(self, indexing_manager):
        """Initializes the SearchingManager.

        Args:
            indexing_manager: An instance of IndexingManager to access indexed data.
        """
        self.indexing_manager = indexing_manager
        # Initialize cache if enabled
        self.cache = get_cache_instance() if ENABLE_CACHING else None

    def search_codebase(
        self,
        codebase_path: str,
        pattern: str,
        file_types: list[str] = None,
        exclude_patterns: list[str] = None,
    ) -> list[dict]:
        """
        Busca un patrón dentro de los archivos de código indexados, con patrones de exclusión opcionales.

        Args:
            codebase_path (str): Ruta a la codebase indexada.
            pattern (str): Patrón regex a buscar.
            file_types (list[str], optional): Lista de extensiones de archivo a incluir en la búsqueda.
                Si es None, se incluyen todos los tipos de archivo.
            exclude_patterns (list[str], optional): Lista de patrones a excluir de la búsqueda.
                Los archivos que coincidan con estos patrones serán omitidos.

        Returns:
            list[dict]: Lista de resultados de búsqueda, cada uno contiene la ruta del archivo,
                número de línea y contenido de la línea donde se encontró el patrón.

        Raises:
            ValueError: Si la codebase no ha sido indexada.
            re.error: Si el patrón proporcionado no es una expresión regular válida.
        """
        abs_codebase_path = str(Path(codebase_path).resolve())
        if abs_codebase_path not in self.indexing_manager.indexed_codebases:
            raise ValueError(
                f"Codebase at {codebase_path} has not been indexed. "
                f"Please index it first."
            )

        indexed_files = self.indexing_manager.indexed_codebases[abs_codebase_path][
            "files"
        ]
        search_results = []

        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {pattern}. Error: {e}")

        for relative_file_path in indexed_files:
            file_path = Path(codebase_path) / relative_file_path

            # Apply file_types filter
            if file_types and file_path.suffix.lstrip(".") not in file_types:
                continue

            # Apply exclude_patterns filter
            if exclude_patterns:
                should_exclude = False
                for exclude_pattern in exclude_patterns:
                    # Check against the full relative path
                    if fnmatch.fnmatch(str(relative_file_path), exclude_pattern):
                        should_exclude = True
                        break
                    # Check against the file name only
                    if fnmatch.fnmatch(file_path.name, exclude_pattern):
                        should_exclude = True
                        break
                if should_exclude:
                    continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line_content in enumerate(f, 1):
                        if re.search(pattern, line_content):
                            search_results.append(
                                {
                                    "file_path": str(file_path),
                                    "line_number": line_num,
                                    "line_content": line_content.strip(),
                                }
                            )
            except FileNotFoundError:
                print(f"Warning: File not found during search: {file_path}. Skipping.")
                continue
            except Exception as e:
                print(f"Error processing file {file_path}: {e}. Skipping.")
                continue
        return search_results

    def semantic_search_codebase(
        self, query: str, sentence_transformer_model, top_k: int = 5
    ) -> list[dict]:
        """
        Realiza una búsqueda semántica dentro de la codebase indexada usando transformadores de oraciones.

        Args:
            query (str): La consulta semántica a buscar.
            sentence_transformer_model: El modelo para codificar la consulta.
            top_k (int, optional): Número de resultados similares principales a devolver. Por defecto es 5.

        Returns:
            list[dict]: Lista de resultados de búsqueda, cada uno contiene la ruta del archivo y
                puntuación de similitud. Puntuaciones más altas indican mayor similitud semántica.

        Note:
            Requiere que la codebase esté indexada primero. Devuelve una lista vacía si
            no existe índice o si no hay embeddings disponibles.
        """
        # Acceder al índice a través del IndexingManager
        faiss_index = self.indexing_manager.faiss_index
        file_paths_map = self.indexing_manager.file_paths_map

        if faiss_index is None or faiss_index.ntotal == 0:
            return []  # No index or no embeddings

        # Check cache for similar queries first
        query_embedding = sentence_transformer_model.encode(query)
        cached_results = None
        cache_hit = False

        if self.cache:
            cached_results, cache_hit = self.cache.get_search_results(query, query_embedding, top_k)

        if cache_hit and cached_results:
            print(f"Cache hit for search query: '{query}'")
            return cached_results

        # Perform search
        query_embedding_reshaped = np.array([query_embedding]).astype("float32")
        distances, indices = faiss_index.search(query_embedding_reshaped, top_k)

        search_results = []
        for i, dist in zip(indices[0], distances[0]):
            if i == -1:  # -1 indicates no result found
                continue
            file_path = file_paths_map.get(str(i))
            if file_path:
                search_results.append(
                    {
                        "file_path": file_path,
                        "score": float(dist),  # Convert numpy float to Python float
                    }
                )

        # Store results in cache
        if self.cache and search_results:
            self.cache.store_search_results(query, query_embedding, search_results)
            print(f"Cached search results for query: '{query}'")

        return search_results

    def find_duplicate_code(
        self,
        codebase_path: str,
        sentence_transformer_model,
        min_similarity: float = 0.8,
        min_lines: int = 10,
    ) -> list[dict]:
        """
        Encuentra secciones de código duplicadas dentro de la codebase indexada.

        Args:
            codebase_path (str): Ruta a la codebase indexada.
            sentence_transformer_model: El modelo para codificar secciones de código.
            min_similarity (float): Puntuación mínima de similitud para considerar
                fragmentos como duplicados.
            min_lines (int): Número mínimo de líneas que debe tener una sección de código.

        Returns:
            list[dict]: Lista de pares de código duplicado con rutas de archivo, números de
                línea y puntuaciones de similitud.
        """
        abs_codebase_path = str(Path(codebase_path).resolve())
        if abs_codebase_path not in self.indexing_manager.indexed_codebases:
            raise ValueError(
                f"Codebase at {codebase_path} has not been indexed. "
                f"Please index it first."
            )

        # Acceder al índice a través del IndexingManager
        faiss_index = self.indexing_manager.faiss_index
        file_paths_map = self.indexing_manager.file_paths_map

        if faiss_index is None or faiss_index.ntotal == 0:
            return []  # No index or no embeddings

        # Get all indexed file paths
        indexed_files = self.indexing_manager.indexed_codebases[abs_codebase_path][
            "files"
        ]
        duplicates = []

        # Filter out archived files to avoid false positives
        filtered_files = [f for f in indexed_files if not f.startswith("archive/")]

        # For each file, split into sections and compare
        for relative_file_path in filtered_files:
            # Skip archived files entirely
            if relative_file_path.startswith("archive/"):
                continue

            file_path = Path(codebase_path) / relative_file_path

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Split file into sections of at least min_lines
                for i in range(0, len(lines), min_lines):
                    section_lines = lines[i : i + min_lines]
                    if len(section_lines) < min_lines:
                        continue  # Skip sections shorter than min_lines

                    section_content = "".join(section_lines)

                    # Encode the section
                    section_embedding = sentence_transformer_model.encode(
                        section_content
                    )
                    section_embedding = np.array([section_embedding]).astype("float32")

                    # Search for similar sections in the entire codebase
                    distances, indices = faiss_index.search(
                        section_embedding, 10
                    )  # Search for top 10

                    # Check results
                    for j, dist in zip(indices[0], distances[0]):
                        if j == -1:  # -1 indicates no result found
                            continue

                        # Calculate similarity score (convert distance to similarity)
                        # Assuming L2 distance, convert to similarity
                        similarity = 1 - (dist / 2)

                        if similarity >= min_similarity:
                            # Get the file path of the matching section
                            matching_file_path = file_paths_map.get(str(j))

                            # Skip if it's the same file and section
                            # j is a FAISS ID, not a line number, so we need to compare file paths correctly
                            if matching_file_path == str(file_path):
                                # Same file, but we still want to report if it's a different section
                                # unless it's literally the same section (same FAISS ID)
                                # But since we're doing a self-search, FAISS will return the same section
                                # We should skip all self-matches entirely to avoid false positives
                                continue

                            duplicates.append(
                                {
                                    "file1": str(file_path),
                                    "file2": matching_file_path,
                                    "start_line1": int(i + 1),  # Convert to Python int
                                    "end_line1": int(
                                        i + min_lines
                                    ),  # Convert to Python int
                                    "start_line2": int(
                                        j + 1
                                    ),  # Convert to Python int, this is approximate
                                    "end_line2": int(
                                        j + min_lines
                                    ),  # Convert to Python int, this is approximate
                                    "similarity": float(
                                        similarity
                                    ),  # Convert to Python float
                                }
                            )

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        # Remove duplicates from the results list (since we're comparing
        # bidirectionally)
        seen_pairs = set()
        unique_duplicates = []
        for dup in duplicates:
            pair_key = tuple(sorted([dup["file1"], dup["file2"]]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                unique_duplicates.append(dup)

        return unique_duplicates

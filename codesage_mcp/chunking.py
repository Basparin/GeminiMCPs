"""
Document Chunking Module for CodeSage MCP Server.

This module provides functionality for splitting large documents into semantic chunks
for better memory efficiency and search granularity.

Classes:
    DocumentChunker: Handles document chunking with semantic awareness.
"""

import re
import tiktoken
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from .config import CHUNK_SIZE_TOKENS


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    content: str
    start_line: int
    end_line: int
    token_count: int
    chunk_type: str  # 'function', 'class', 'docstring', 'general'
    metadata: Dict[str, any]


class DocumentChunker:
    """Handles document chunking with semantic awareness."""

    def __init__(self, chunk_size_tokens: int = None):
        self.chunk_size_tokens = chunk_size_tokens or CHUNK_SIZE_TOKENS
        # Use tiktoken for accurate token counting
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except:
            # Fallback to approximate token counting
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Approximate token count (rough estimate)
            return len(text) // 4

    def split_into_chunks(self, content: str, file_path: str = None) -> List[DocumentChunk]:
        """Split document content into semantic chunks."""
        lines = content.split('\n')
        chunks = []

        # First, try to extract semantic units (functions, classes, etc.)
        semantic_chunks = self._extract_semantic_units(content, lines)

        if semantic_chunks:
            # Process semantic chunks
            for chunk in semantic_chunks:
                if self.count_tokens(chunk.content) <= self.chunk_size_tokens:
                    chunks.append(chunk)
                else:
                    # Split large semantic chunks further
                    sub_chunks = self._split_large_chunk(chunk)
                    chunks.extend(sub_chunks)
        else:
            # Fall back to line-based chunking
            chunks = self._chunk_by_lines(lines, file_path)

        return chunks

    def _extract_semantic_units(self, content: str, lines: List[str]) -> List[DocumentChunk]:
        """Extract semantic units like functions, classes, etc."""
        chunks = []

        # Python code patterns
        patterns = [
            # Functions and methods
            (r'def\s+(\w+)\s*\([^)]*\)\s*:', 'function'),
            # Classes
            (r'class\s+(\w+)\s*[:\(]', 'class'),
            # Docstrings (triple-quoted strings at start of line)
            (r'^\s*""".*?"""', 'docstring'),
            (r"^\s*'''.*?'''", 'docstring'),
        ]

        current_line = 0
        for i, line in enumerate(lines):
            for pattern, chunk_type in patterns:
                if re.match(pattern, line, re.MULTILINE):
                    # Find the end of this semantic unit
                    start_line = i
                    end_line = self._find_end_of_unit(lines, i, chunk_type)

                    unit_content = '\n'.join(lines[start_line:end_line + 1])
                    token_count = self.count_tokens(unit_content)

                    if token_count > 0:
                        chunks.append(DocumentChunk(
                            content=unit_content,
                            start_line=start_line + 1,  # 1-based line numbers
                            end_line=end_line + 1,
                            token_count=token_count,
                            chunk_type=chunk_type,
                            metadata={'pattern': pattern}
                        ))

                    current_line = end_line + 1
                    break

        return chunks

    def _find_end_of_unit(self, lines: List[str], start_idx: int, unit_type: str) -> int:
        """Find the end line of a semantic unit."""
        indent_level = self._get_indent_level(lines[start_idx])

        for i in range(start_idx + 1, len(lines)):
            line = lines[i]

            # Empty lines don't end units
            if not line.strip():
                continue

            # Check indentation level
            current_indent = self._get_indent_level(line)

            # If we encounter a line with same or less indentation that's not empty
            if current_indent <= indent_level and line.strip():
                # Check if it's another semantic unit at the same level
                if any(re.match(pattern, line) for pattern, _ in [
                    (r'def\s+(\w+)\s*\([^)]*\)\s*:', 'function'),
                    (r'class\s+(\w+)\s*[:\(]', 'class'),
                ]):
                    return i - 1
                # For docstrings, they end at the closing quotes
                elif unit_type == 'docstring':
                    if '"""' in line or "'''" in line:
                        return i

        return len(lines) - 1

    def _get_indent_level(self, line: str) -> int:
        """Get the indentation level of a line."""
        return len(line) - len(line.lstrip())

    def _split_large_chunk(self, chunk: DocumentChunk) -> List[DocumentChunk]:
        """Split a large chunk into smaller pieces."""
        lines = chunk.content.split('\n')
        sub_chunks = []

        current_chunk_lines = []
        current_token_count = 0

        for line in lines:
            line_tokens = self.count_tokens(line + '\n')

            if current_token_count + line_tokens > self.chunk_size_tokens and current_chunk_lines:
                # Create a sub-chunk
                sub_content = '\n'.join(current_chunk_lines)
                sub_chunks.append(DocumentChunk(
                    content=sub_content,
                    start_line=chunk.start_line + len(sub_chunks),
                    end_line=chunk.start_line + len(sub_chunks) + len(current_chunk_lines) - 1,
                    token_count=current_token_count,
                    chunk_type='general',
                    metadata={'parent_type': chunk.chunk_type}
                ))

                current_chunk_lines = [line]
                current_token_count = line_tokens
            else:
                current_chunk_lines.append(line)
                current_token_count += line_tokens

        # Add the last sub-chunk
        if current_chunk_lines:
            sub_content = '\n'.join(current_chunk_lines)
            sub_chunks.append(DocumentChunk(
                content=sub_content,
                start_line=chunk.start_line + len(sub_chunks),
                end_line=chunk.end_line,
                token_count=current_token_count,
                chunk_type='general',
                metadata={'parent_type': chunk.chunk_type}
            ))

        return sub_chunks

    def _chunk_by_lines(self, lines: List[str], file_path: str = None) -> List[DocumentChunk]:
        """Fallback chunking by lines when semantic extraction fails."""
        chunks = []

        current_chunk_lines = []
        current_token_count = 0
        current_start_line = 0

        for i, line in enumerate(lines):
            line_tokens = self.count_tokens(line + '\n')

            if current_token_count + line_tokens > self.chunk_size_tokens and current_chunk_lines:
                # Create chunk
                chunk_content = '\n'.join(current_chunk_lines)
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    start_line=current_start_line + 1,
                    end_line=i,
                    token_count=current_token_count,
                    chunk_type='general',
                    metadata={'file_path': str(file_path) if file_path else None}
                ))

                current_chunk_lines = [line]
                current_token_count = line_tokens
                current_start_line = i
            else:
                current_chunk_lines.append(line)
                current_token_count += line_tokens

        # Add the last chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunks.append(DocumentChunk(
                content=chunk_content,
                start_line=current_start_line + 1,
                end_line=len(lines),
                token_count=current_token_count,
                chunk_type='general',
                metadata={'file_path': str(file_path) if file_path else None}
            ))

        return chunks

    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, any]:
        """Get statistics about the chunking results."""
        if not chunks:
            return {}

        total_tokens = sum(chunk.token_count for chunk in chunks)
        chunk_types = {}
        chunk_sizes = []

        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            chunk_sizes.append(chunk.token_count)

        return {
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'average_chunk_size': total_tokens / len(chunks),
            'chunk_types': chunk_types,
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'target_chunk_size': self.chunk_size_tokens
        }


def chunk_file(file_path: str, chunk_size_tokens: int = None) -> List[DocumentChunk]:
    """Convenience function to chunk a file."""
    chunker = DocumentChunker(chunk_size_tokens)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return chunker.split_into_chunks(content, file_path)

    except Exception as e:
        print(f"Warning: Could not chunk file {file_path}: {e}")
        return []


def chunk_text(content: str, chunk_size_tokens: int = None) -> List[DocumentChunk]:
    """Convenience function to chunk text content."""
    chunker = DocumentChunker(chunk_size_tokens)
    return chunker.split_into_chunks(content)
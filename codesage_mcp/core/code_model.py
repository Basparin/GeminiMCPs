"""
Code Model Generation Module for CodeSage MCP Server.

This module provides a multi-layered graph architecture for representing Python code
structures and relationships. It enables efficient code analysis, search, and
incremental updates.

Classes:
    CodeNode: Represents individual code elements (functions, classes, modules, etc.)
    Relationship: Represents relationships between code elements
    GraphLayer: Represents different layers of abstraction in the code graph
    CodeGraph: Main graph structure managing all layers and operations
    CodeModelGenerator: Generates code models from Python source code using AST
"""

import ast
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import threading

from ..config.config import ENABLE_CACHING
from codesage_mcp.features.caching.cache import get_cache_instance
from codesage_mcp.features.memory_management.memory_manager import get_memory_manager

# Set up logger
logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for performance
NAME_SEARCH_PATTERN = re.compile(r'(?i)')
CONTENT_EXTRACTION_PATTERN = re.compile(r'^(\s*)')
DECORATOR_PATTERN = re.compile(r'@(\w+)')
IMPORT_PATTERN = re.compile(r'^(?:from\s+[\w.]+\s+)?import\s+')


class NodeType(Enum):
    """Types of code elements that can be represented as nodes."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    ATTRIBUTE = "attribute"
    PARAMETER = "parameter"


class RelationshipType(Enum):
    """Types of relationships between code elements."""
    CONTAINS = "contains"  # Parent-child relationship
    CALLS = "calls"  # Function/method call
    INHERITS = "inherits"  # Class inheritance
    IMPLEMENTS = "implements"  # Interface implementation
    USES = "uses"  # Variable/attribute usage
    DEFINES = "defines"  # Definition relationship
    DEPENDS_ON = "depends_on"  # Dependency relationship
    REFERENCES = "references"  # General reference


class LayerType(Enum):
    """Types of graph layers for different abstraction levels."""
    SYNTAX = "syntax"  # Basic AST structure
    SEMANTIC = "semantic"  # Code meaning and relationships
    DEPENDENCY = "dependency"  # Import and dependency relationships
    CONTROL_FLOW = "control_flow"  # Control flow analysis
    DATA_FLOW = "data_flow"  # Data flow analysis


@dataclass
class CodeNode:
    """Represents a code element in the graph."""

    node_type: NodeType
    name: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    content: str = field(default="")
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    id: Optional[str] = None

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            # Create a unique ID based on qualified name and file
            id_content = f"{self.qualified_name}:{self.file_path}:{self.start_line}"
            self.id = hashlib.md5(id_content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeNode':
        """Create node from dictionary."""
        return cls(
            id=data["id"],
            node_type=NodeType(data["node_type"]),
            name=data["name"],
            qualified_name=data["qualified_name"],
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )

    def update_content(self, new_content: str):
        """Update node content and timestamp."""
        self.content = new_content
        self.updated_at = time.time()


@dataclass
class Relationship:
    """Represents a relationship between two code nodes."""

    source_id: str
    target_id: str
    relationship_type: RelationshipType
    layer: LayerType
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type.value,
            "layer": self.layer.value,
            "metadata": self.metadata,
            "weight": self.weight,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Create relationship from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relationship_type=RelationshipType(data["relationship_type"]),
            layer=LayerType(data["layer"]),
            metadata=data.get("metadata", {}),
            weight=data.get("weight", 1.0),
            created_at=data.get("created_at", time.time()),
        )


class GraphLayer:
    """Represents a layer in the multi-layered graph architecture."""

    def __init__(self, layer_type: LayerType):
        self.layer_type = layer_type
        self.nodes: Dict[str, CodeNode] = {}
        self.relationships: List[Relationship] = []
        self.node_relationships: Dict[str, List[str]] = defaultdict(list)  # node_id -> relationship indices
        self._lock = threading.RLock()

    def add_node(self, node: CodeNode) -> None:
        """Add a node to this layer."""
        with self._lock:
            self.nodes[node.id] = node

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to this layer."""
        with self._lock:
            self.relationships.append(relationship)
            # Track relationships for efficient lookup
            self.node_relationships[relationship.source_id].append(str(len(self.relationships) - 1))
            self.node_relationships[relationship.target_id].append(str(len(self.relationships) - 1))

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its relationships from this layer."""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]

            # Remove relationships involving this node
            indices_to_remove = []
            for i, rel in enumerate(self.relationships):
                if rel.source_id == node_id or rel.target_id == node_id:
                    indices_to_remove.append(i)

            # Remove in reverse order to maintain indices
            for i in reversed(indices_to_remove):
                del self.relationships[i]

            # Clean up relationship tracking
            if node_id in self.node_relationships:
                del self.node_relationships[node_id]

    def get_node_relationships(self, node_id: str) -> List[Relationship]:
        """Get all relationships for a given node."""
        with self._lock:
            relationships = []
            if node_id in self.node_relationships:
                for rel_idx in self.node_relationships[node_id]:
                    try:
                        relationships.append(self.relationships[int(rel_idx)])
                    except (IndexError, ValueError):
                        continue
            return relationships

    def get_nodes_by_type(self, node_type: NodeType) -> List[CodeNode]:
        """Get all nodes of a specific type."""
        with self._lock:
            return [node for node in self.nodes.values() if node.node_type == node_type]

    def get_relationships_by_type(self, rel_type: RelationshipType) -> List[Relationship]:
        """Get all relationships of a specific type."""
        with self._lock:
            return [rel for rel in self.relationships if rel.relationship_type == rel_type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert layer to dictionary for serialization."""
        return {
            "layer_type": self.layer_type.value,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "relationships": [rel.to_dict() for rel in self.relationships],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphLayer':
        """Create layer from dictionary."""
        layer = cls(LayerType(data["layer_type"]))

        # Restore nodes
        for node_data in data.get("nodes", {}).values():
            node = CodeNode.from_dict(node_data)
            layer.nodes[node.id] = node

        # Restore relationships
        for rel_data in data.get("relationships", []):
            relationship = Relationship.from_dict(rel_data)
            layer.relationships.append(relationship)
            # Restore relationship tracking
            layer.node_relationships[relationship.source_id].append(str(len(layer.relationships) - 1))
            layer.node_relationships[relationship.target_id].append(str(len(layer.relationships) - 1))

        return layer


class CodeGraph:
    """Main graph structure managing all layers and operations."""

    def __init__(self):
        self.layers: Dict[LayerType, GraphLayer] = {}
        self.file_nodes: Dict[str, Set[str]] = defaultdict(set)  # file_path -> set of node_ids
        self.node_files: Dict[str, str] = {}  # node_id -> file_path
        self.cache = get_cache_instance() if ENABLE_CACHING else None
        self.memory_manager = get_memory_manager()
        self._lock = threading.RLock()

        # Initialize layers
        for layer_type in LayerType:
            self.layers[layer_type] = GraphLayer(layer_type)

    def add_node(self, node: CodeNode, layer_type: LayerType = LayerType.SEMANTIC) -> None:
        """Add a node to the specified layer."""
        with self._lock:
            if layer_type not in self.layers:
                self.layers[layer_type] = GraphLayer(layer_type)

            self.layers[layer_type].add_node(node)
            self.file_nodes[node.file_path].add(node.id)
            self.node_files[node.id] = node.file_path

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the appropriate layer."""
        with self._lock:
            if relationship.layer not in self.layers:
                self.layers[relationship.layer] = GraphLayer(relationship.layer)

            self.layers[relationship.layer].add_relationship(relationship)

    def remove_file_nodes(self, file_path: str) -> None:
        """Remove all nodes and relationships for a file."""
        with self._lock:
            if file_path not in self.file_nodes:
                return

            node_ids = self.file_nodes[file_path].copy()

            # Remove from all layers
            for layer in self.layers.values():
                for node_id in node_ids:
                    layer.remove_node(node_id)

            # Clean up tracking
            del self.file_nodes[file_path]
            for node_id in node_ids:
                if node_id in self.node_files:
                    del self.node_files[node_id]

    def get_node(self, node_id: str, layer_type: LayerType = None) -> Optional[CodeNode]:
        """Get a node by ID from specified layer or any layer."""
        with self._lock:
            if layer_type:
                return self.layers.get(layer_type, GraphLayer(layer_type)).nodes.get(node_id)

            # Search all layers
            for layer in self.layers.values():
                if node_id in layer.nodes:
                    return layer.nodes[node_id]
            return None

    def get_file_nodes(self, file_path: str, layer_type: LayerType = None) -> List[CodeNode]:
        """Get all nodes for a file from specified layer or all layers."""
        with self._lock:
            if file_path not in self.file_nodes:
                return []

            nodes = []
            if layer_type:
                layer = self.layers.get(layer_type, GraphLayer(layer_type))
                for node_id in self.file_nodes[file_path]:
                    if node_id in layer.nodes:
                        nodes.append(layer.nodes[node_id])
            else:
                # Get from all layers
                for layer in self.layers.values():
                    for node_id in self.file_nodes[file_path]:
                        if node_id in layer.nodes:
                            nodes.append(layer.nodes[node_id])

            return nodes

    def get_node_relationships(self, node_id: str, layer_type: LayerType = None) -> List[Relationship]:
        """Get all relationships for a node from specified layer or all layers."""
        with self._lock:
            relationships = []
            if layer_type:
                if layer_type in self.layers:
                    relationships.extend(self.layers[layer_type].get_node_relationships(node_id))
            else:
                for layer in self.layers.values():
                    relationships.extend(layer.get_node_relationships(node_id))

            return relationships

    def find_nodes_by_name(self, name: str, layer_type: LayerType = None) -> List[CodeNode]:
        """Find nodes by name pattern."""
        with self._lock:
            nodes = []
            layers_to_search = [self.layers[layer_type]] if layer_type else self.layers.values()
            name_lower = name.lower()

            for layer in layers_to_search:
                for node in layer.nodes.values():
                    if name_lower in node.name.lower() or name_lower in node.qualified_name.lower():
                        nodes.append(node)

            return nodes

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the graph."""
        with self._lock:
            stats = {
                "total_files": len(self.file_nodes),
                "total_nodes": sum(len(layer.nodes) for layer in self.layers.values()),
                "total_relationships": sum(len(layer.relationships) for layer in self.layers.values()),
                "layers": {},
            }

            for layer_type, layer in self.layers.items():
                stats["layers"][layer_type.value] = {
                    "nodes": len(layer.nodes),
                    "relationships": len(layer.relationships),
                    "node_types": {},
                    "relationship_types": {},
                }

                # Count node types
                for node in layer.nodes.values():
                    node_type = node.node_type.value
                    stats["layers"][layer_type.value]["node_types"][node_type] = \
                        stats["layers"][layer_type.value]["node_types"].get(node_type, 0) + 1

                # Count relationship types
                for rel in layer.relationships:
                    rel_type = rel.relationship_type.value
                    stats["layers"][layer_type.value]["relationship_types"][rel_type] = \
                        stats["layers"][layer_type.value]["relationship_types"].get(rel_type, 0) + 1

            return stats

    def save_to_file(self, file_path: str) -> None:
        """Save the graph to a JSON file."""
        with self._lock:
            data = {
                "layers": {layer_type.value: layer.to_dict() for layer_type, layer in self.layers.items()},
                "file_nodes": {fp: list(node_ids) for fp, node_ids in self.file_nodes.items()},
                "node_files": self.node_files.copy(),
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

    def load_from_file(self, file_path: str) -> None:
        """Load the graph from a JSON file."""
        with self._lock:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Clear existing data
                self.layers.clear()
                self.file_nodes.clear()
                self.node_files.clear()

                # Restore layers
                for layer_name, layer_data in data.get("layers", {}).items():
                    layer_type = LayerType(layer_name)
                    layer = GraphLayer.from_dict(layer_data)
                    self.layers[layer_type] = layer

                # Restore file tracking
                for fp, node_ids in data.get("file_nodes", {}).items():
                    self.file_nodes[fp] = set(node_ids)

                self.node_files.update(data.get("node_files", {}))

            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load graph from {file_path}: {e}")
                # Initialize empty graph
                for layer_type in LayerType:
                    self.layers[layer_type] = GraphLayer(layer_type)

    def cleanup(self) -> None:
        """Clean up resources."""
        with self._lock:
            if self.cache:
                self.cache.save_persistent_cache()

    def optimize_for_memory(self, target_memory_mb: float = 100.0) -> Dict[str, Any]:
        """Optimize graph for memory usage by removing old/unused nodes.

        Args:
            target_memory_mb: Target memory usage in MB

        Returns:
            Dictionary with optimization results
        """
        with self._lock:
            # Calculate current memory usage (rough estimate)
            total_nodes = sum(len(layer.nodes) for layer in self.layers.values())
            total_relationships = sum(len(layer.relationships) for layer in self.layers.values())

            # Rough memory estimate: 1KB per node, 0.5KB per relationship
            current_memory_mb = (total_nodes * 1024 + total_relationships * 512) / (1024 * 1024)

            if current_memory_mb <= target_memory_mb:
                return {
                    "optimized": False,
                    "reason": "Already within memory target",
                    "current_memory_mb": current_memory_mb,
                    "target_memory_mb": target_memory_mb
                }

            # Remove old nodes (older than 30 days)
            import time
            cutoff_time = time.time() - (30 * 24 * 60 * 60)  # 30 days ago
            removed_nodes = 0
            removed_relationships = 0

            for layer in self.layers.values():
                nodes_to_remove = []
                for node_id, node in layer.nodes.items():
                    if node.updated_at < cutoff_time:
                        nodes_to_remove.append(node_id)

                for node_id in nodes_to_remove:
                    layer.remove_node(node_id)
                    removed_nodes += 1

                # Also clean up relationships
                relationships_to_remove = []
                for i, rel in enumerate(layer.relationships):
                    if (rel.source_id not in layer.nodes or
                        rel.target_id not in layer.nodes):
                        relationships_to_remove.append(i)

                for i in reversed(relationships_to_remove):
                    del layer.relationships[i]
                    removed_relationships += 1

            return {
                "optimized": True,
                "removed_nodes": removed_nodes,
                "removed_relationships": removed_relationships,
                "current_memory_mb": current_memory_mb,
                "target_memory_mb": target_memory_mb
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the graph."""
        with self._lock:
            stats = self.get_statistics()
            stats.update({
                "memory_stats": self.memory_manager.get_memory_stats() if self.memory_manager else {},
                "cache_stats": self.cache.get_stats() if self.cache else {},
                "layer_performance": {}
            })

            # Add layer-specific performance metrics
            for layer_type, layer in self.layers.items():
                stats["layer_performance"][layer_type.value] = {
                    "nodes_count": len(layer.nodes),
                    "relationships_count": len(layer.relationships),
                    "avg_relationships_per_node": len(layer.relationships) / max(1, len(layer.nodes)),
                    "node_types_distribution": {},
                    "relationship_types_distribution": {}
                }

                # Count node types
                for node in layer.nodes.values():
                    node_type = node.node_type.value
                    stats["layer_performance"][layer_type.value]["node_types_distribution"][node_type] = \
                        stats["layer_performance"][layer_type.value]["node_types_distribution"].get(node_type, 0) + 1

                # Count relationship types
                for rel in layer.relationships:
                    rel_type = rel.relationship_type.value
                    stats["layer_performance"][layer_type.value]["relationship_types_distribution"][rel_type] = \
                        stats["layer_performance"][layer_type.value]["relationship_types_distribution"].get(rel_type, 0) + 1

            return stats


class CodeModelGenerator:
    """Generates code models from Python source code using AST parsing."""

    def __init__(self, graph: CodeGraph = None):
        self.graph = graph or CodeGraph()
        self.cache = get_cache_instance() if ENABLE_CACHING else None
        self.memory_manager = get_memory_manager()

    def generate_from_file(self, file_path: str, content: str = None) -> List[CodeNode]:
        """Generate code model from a Python file."""
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
                return []

        # Check cache first (using file cache for code models)
        cache_key = f"code_model:{file_path}:{hashlib.md5(content.encode()).hexdigest()}"
        if self.cache:
            cached_content, cache_hit = self.cache.get_file_content(cache_key)
            if cache_hit and cached_content:
                try:
                    cached_nodes = json.loads(cached_content)
                    # Convert back to CodeNode objects
                    nodes = []
                    for node_data in cached_nodes:
                        nodes.append(CodeNode.from_dict(node_data))
                    logger.debug(f"Loaded code model from cache for {file_path}")
                    return nodes
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Failed to deserialize cached code model: {e}")

        try:
            # Parse AST
            tree = ast.parse(content, filename=file_path)

            # Remove existing nodes for this file
            self.graph.remove_file_nodes(file_path)

            # Generate nodes and relationships
            nodes = self._generate_nodes_from_ast(tree, file_path, content)

            # Cache the result
            if self.cache:
                # Serialize nodes to JSON for caching
                nodes_data = [node.to_dict() for node in nodes]
                cache_content = json.dumps(nodes_data)
                self.cache.store_file_content(cache_key, cache_content)

            return nodes

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error generating code model for {file_path}: {e}")
            return []

    def _generate_nodes_from_ast(self, tree: ast.AST, file_path: str, content: str) -> List[CodeNode]:
        """Generate nodes from AST tree."""
        nodes = []
        lines = content.split('\n')

        # Create module node
        module_name = Path(file_path).stem
        module_node = CodeNode(
            id="",
            node_type=NodeType.MODULE,
            name=module_name,
            qualified_name=module_name,
            file_path=file_path,
            start_line=1,
            end_line=len(lines),
            content=content,
            metadata={"type": "module"}
        )
        self.graph.add_node(module_node, LayerType.SEMANTIC)
        nodes.append(module_node)

        # Walk the AST and create nodes
        for node in ast.walk(tree):
            code_node = self._create_node_from_ast_node(node, file_path, lines)
            if code_node:
                self.graph.add_node(code_node, LayerType.SEMANTIC)
                nodes.append(code_node)

                # Create containment relationship with module
                if code_node.node_type in [NodeType.CLASS, NodeType.FUNCTION]:
                    relationship = Relationship(
                        source_id=module_node.id,
                        target_id=code_node.id,
                        relationship_type=RelationshipType.CONTAINS,
                        layer=LayerType.SEMANTIC,
                        metadata={"containment_type": "module"}
                    )
                    self.graph.add_relationship(relationship)

        # Generate relationships
        self._generate_relationships_from_ast(tree, file_path)

        return nodes

    def _create_node_from_ast_node(self, node: ast.AST, file_path: str, lines: List[str]) -> Optional[CodeNode]:
        """Create a CodeNode from an AST node."""
        try:
            if isinstance(node, ast.ClassDef):
                return self._create_class_node(node, file_path, lines)
            elif isinstance(node, ast.FunctionDef):
                return self._create_function_node(node, file_path, lines)
            elif isinstance(node, ast.Import):
                return self._create_import_node(node, file_path, lines)
            elif isinstance(node, ast.ImportFrom):
                return self._create_import_from_node(node, file_path, lines)
        except Exception as e:
            logger.debug(f"Error creating node from AST node {type(node).__name__}: {e}")

        return None

    def _create_class_node(self, node: ast.ClassDef, file_path: str, lines: List[str]) -> CodeNode:
        """Create a CodeNode for a class definition."""
        start_line = node.lineno
        end_line = self._get_node_end_line(node, lines)

        # Get class content - optimized for memory
        if start_line <= len(lines):
            content_lines = lines[start_line-1:end_line]
            content = '\n'.join(content_lines)
        else:
            content = ""

        # Get base classes - optimized
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_attribute_name(base))

        # Get decorators - optimized
        decorators = []
        for d in node.decorator_list:
            decorator_name = self._get_decorator_name(d)
            if decorator_name:
                decorators.append(decorator_name)

        return CodeNode(
            id="",
            node_type=NodeType.CLASS,
            name=node.name,
            qualified_name=node.name,  # TODO: Add module prefix
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            metadata={
                "type": "class",
                "bases": bases,
                "decorators": decorators
            }
        )

    def _create_function_node(self, node: ast.FunctionDef, file_path: str, lines: List[str]) -> CodeNode:
        """Create a CodeNode for a function definition."""
        start_line = node.lineno
        end_line = self._get_node_end_line(node, lines)

        # Get function content - optimized for memory
        if start_line <= len(lines):
            content_lines = lines[start_line-1:end_line]
            content = '\n'.join(content_lines)
        else:
            content = ""

        # Get parameters - optimized
        parameters = []
        for arg in node.args.args:
            param_info = {
                "name": arg.arg,
                "type": self._get_annotation_name(arg.annotation) if arg.annotation else None
            }
            parameters.append(param_info)

        # Get decorators - optimized
        decorators = []
        for d in node.decorator_list:
            decorator_name = self._get_decorator_name(d)
            if decorator_name:
                decorators.append(decorator_name)

        return CodeNode(
            id="",
            node_type=NodeType.FUNCTION,
            name=node.name,
            qualified_name=node.name,  # TODO: Add class/module prefix
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            metadata={
                "type": "function",
                "parameters": parameters,
                "return_type": self._get_annotation_name(node.returns) if node.returns else None,
                "decorators": decorators,
                "is_async": isinstance(node, ast.AsyncFunctionDef)
            }
        )

    def _create_import_node(self, node: ast.Import, file_path: str, lines: List[str]) -> CodeNode:
        """Create a CodeNode for an import statement."""
        names = [alias.name for alias in node.names]

        return CodeNode(
            id="",
            node_type=NodeType.IMPORT,
            name=f"import {', '.join(names)}",
            qualified_name=f"import_{hashlib.md5(str(names).encode()).hexdigest()[:8]}",
            file_path=file_path,
            start_line=node.lineno,
            end_line=node.lineno,
            content=lines[node.lineno-1] if node.lineno <= len(lines) else "",
            metadata={
                "type": "import",
                "names": names,
                "aliases": {alias.name: alias.asname for alias in node.names if alias.asname}
            }
        )

    def _create_import_from_node(self, node: ast.ImportFrom, file_path: str, lines: List[str]) -> CodeNode:
        """Create a CodeNode for a from import statement."""
        module = node.module or ""
        names = [alias.name for alias in node.names]

        return CodeNode(
            id="",
            node_type=NodeType.IMPORT,
            name=f"from {module} import {', '.join(names)}",
            qualified_name=f"import_from_{hashlib.md5(f'{module}:{names}'.encode()).hexdigest()[:8]}",
            file_path=file_path,
            start_line=node.lineno,
            end_line=node.lineno,
            content=lines[node.lineno-1] if node.lineno <= len(lines) else "",
            metadata={
                "type": "import_from",
                "module": module,
                "names": names,
                "aliases": {alias.name: alias.asname for alias in node.names if alias.asname}
            }
        )

    def _generate_relationships_from_ast(self, tree: ast.AST, file_path: str) -> None:
        """Generate relationships from AST analysis."""
        # This is a simplified implementation - in practice, you'd do more sophisticated analysis
        # For now, we'll focus on basic containment and inheritance relationships

        class RelationshipVisitor(ast.NodeVisitor):
            def __init__(self, generator, file_path):
                self.generator = generator
                self.file_path = file_path
                self.current_class = None
                self.current_function = None

            def visit_ClassDef(self, node):
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = None

            def visit_FunctionDef(self, node):
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = None

            def visit_Call(self, node):
                # Create call relationships
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    # Find the called function node
                    called_node = self._find_node_by_name(func_name)
                    if called_node:
                        caller_node = self._find_current_node()
                        if caller_node:
                            relationship = Relationship(
                                source_id=caller_node.id,
                                target_id=called_node.id,
                                relationship_type=RelationshipType.CALLS,
                                layer=LayerType.SEMANTIC,
                                metadata={"call_type": "direct"}
                            )
                            self.generator.graph.add_relationship(relationship)

            def _find_node_by_name(self, name):
                # Simplified node lookup - in practice, you'd need symbol resolution
                for layer in self.generator.graph.layers.values():
                    for node in layer.nodes.values():
                        if node.name == name and node.file_path == self.file_path:
                            return node
                return None

            def _find_current_node(self):
                # Find the current function or class node
                current_name = self.current_function or self.current_class
                if current_name:
                    return self._find_node_by_name(current_name)
                return None

        visitor = RelationshipVisitor(self, file_path)
        visitor.visit(tree)

    def _get_node_end_line(self, node: ast.AST, lines: List[str]) -> int:
        """Get the end line of an AST node."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno

        # Fallback: estimate based on indentation
        start_line = node.lineno
        if start_line > len(lines):
            return start_line

        # Find the end by looking for lines with less indentation
        start_indent = len(lines[start_line-1]) - len(lines[start_line-1].lstrip())

        for i in range(start_line, len(lines)):
            line = lines[i]
            if line.strip():  # Skip empty lines
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= start_indent:
                    return i

        return len(lines)

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get the full name of an attribute node."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        return node.attr

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Get the name of a decorator."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return str(node)

    def batch_generate_from_files(self, file_paths: List[str], contents: Optional[List[str]] = None) -> Dict[str, List[CodeNode]]:
        """Generate code models for multiple files in batch for better performance.

        Args:
            file_paths: List of file paths to process
            contents: Optional list of file contents (if not provided, files will be read)

        Returns:
            Dictionary mapping file paths to lists of code nodes
        """
        results = {}
        contents = contents or [None] * len(file_paths)

        # Pre-allocate results dictionary for better performance
        for file_path in file_paths:
            results[file_path] = []

        # Process files in batches to optimize memory usage
        batch_size = min(50, len(file_paths))  # Process in batches of 50

        for i in range(0, len(file_paths), batch_size):
            batch_end = min(i + batch_size, len(file_paths))
            batch_files = file_paths[i:batch_end]
            batch_contents = contents[i:batch_end]

            for file_path, content in zip(batch_files, batch_contents):
                try:
                    nodes = self.generate_from_file(file_path, content)
                    results[file_path] = nodes
                    logger.debug(f"Generated {len(nodes)} nodes for {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate code model for {file_path}: {e}")
                    results[file_path] = []

        return results

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about code model generation performance."""
        return {
            "cache_enabled": self.cache is not None,
            "memory_manager_enabled": self.memory_manager is not None,
            "supported_languages": ["python"],  # Can be extended for other languages
            "features": {
                "ast_parsing": True,
                "relationship_inference": True,
                "caching": self.cache is not None,
                "incremental_updates": True,
                "batch_processing": True
            }
        }

    def _get_annotation_name(self, node: ast.expr) -> str:
        """Get the name of a type annotation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Str):  # For Python < 3.9
            return node.s
        elif isinstance(node, ast.Constant):  # For Python >= 3.9
            return str(node.value) if isinstance(node.value, str) else str(node)
        return str(node)
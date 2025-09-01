"""
Integration Tests for Code Model and Analysis Components Interaction.

This module contains comprehensive integration tests that verify the interaction
between code model generation and advanced analysis components, including
cross-file dependencies, cache integration, and end-to-end workflows.
"""

import pytest
import tempfile
import os
import time
import threading
import psutil

from codesage_mcp.core.code_model import (
    CodeGraph,
    CodeModelGenerator,
    CodeNode,
    NodeType,
    LayerType
)
from codesage_mcp.features.codebase_manager import (
    AdvancedAnalysisManager
)
from codesage_mcp.features.caching.intelligent_cache import IntelligentCache
from codesage_mcp.features.memory_management.memory_manager import MemoryManager


class TestCodeModelAnalysisIntegration:
    """Integration tests for code model and analysis interaction."""

    @pytest.fixture
    def integrated_setup(self):
        """Create integrated setup with graph, generator, and analyzer."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_end_to_end_codebase_analysis(self, integrated_setup):
        """Test complete end-to-end analysis of a multi-file codebase."""
        setup = integrated_setup
        graph = setup['graph']
        generator = setup['generator']
        analyzer = setup['analyzer']

        # Create a multi-file codebase
        codebase_files = {}

        # Main module
        main_content = '''
from utils import helper_function
from data_processor import process_data
import config

def main():
    """Main entry point."""
    config_data = config.load_config()
    data = process_data(config_data)
    result = helper_function(data)
    return result

if __name__ == "__main__":
    main()
'''
        codebase_files['main.py'] = main_content

        # Utils module
        utils_content = '''
import os
from typing import List

def helper_function(data: List[dict]) -> dict:
    """Helper function with dependencies."""
    result = {}
    for item in data:
        key = os.path.basename(item.get('path', ''))
        result[key] = item.get('value', 0) * 2
    return result

class Utils:
    @staticmethod
    def validate_data(data):
        return len(data) > 0
'''
        codebase_files['utils.py'] = utils_content

        # Data processor module
        processor_content = '''
import json
from collections import Counter
from utils import Utils

def process_data(config: dict) -> List[dict]:
    """Process data with complex operations."""
    # Nested loops (potential bottleneck)
    result = []
    for i in range(10):
        for j in range(10):
            item = {
                'id': i * 10 + j,
                'value': i + j,
                'path': f'/data/item_{i}_{j}.json'
            }
            result.append(item)

    # Use external libraries
    json_str = json.dumps(result)
    counter = Counter(item['value'] for item in result)

    # Validate using utils
    if Utils.validate_data(result):
        return result
    return []
'''
        codebase_files['data_processor.py'] = processor_content

        # Config module
        config_content = '''
import os

def load_config() -> dict:
    """Load configuration."""
    return {
        'data_path': os.getenv('DATA_PATH', '/default/path'),
        'max_items': 100,
        'debug': True
    }

# Global configuration
DEFAULT_CONFIG = load_config()
'''
        codebase_files['config.py'] = config_content

        # Create temporary files
        temp_files = []
        try:
            for filename, content in codebase_files.items():
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(content)
                    temp_files.append(f.name)

            # Generate code models for all files
            start_time = time.time()
            for file_path in temp_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                generator.generate_from_file(file_path, content)
            model_time = time.time() - start_time

            # Run comprehensive analysis
            analysis_start = time.time()
            results = {}
            for file_path in temp_files:
                result = analyzer.run_comprehensive_analysis(file_path)
                results[file_path] = result
            analysis_time = time.time() - analysis_start

            # Verify integration results
            assert model_time < 5  # Model generation should be fast
            assert analysis_time < 10  # Analysis should be reasonable

            # Check that all files were analyzed
            assert len(results) == len(temp_files)

            # Verify cross-file dependencies were detected
            total_dependencies = 0
            total_bottlenecks = 0

            for file_path, result in results.items():
                dep_analysis = result.get('dependency_analysis', {})
                perf_analysis = result.get('performance_analysis', {})

                # Should have dependency information
                assert 'dependencies' in dep_analysis
                assert 'summary' in dep_analysis

                # Should have performance analysis
                assert 'bottlenecks' in perf_analysis
                assert 'summary' in perf_analysis

                # Accumulate totals
                total_dependencies += dep_analysis['summary'].get('total_functions_analyzed', 0)
                total_bottlenecks += len(perf_analysis.get('bottlenecks', []))

            # Should have found dependencies across files
            assert total_dependencies >= 8  # At least 8 functions across all files

            # Should have found performance bottlenecks (nested loops in processor)
            assert total_bottlenecks >= 3  # At least some bottlenecks detected

        finally:
            # Cleanup
            for file_path in temp_files:
                os.unlink(file_path)

    def test_cross_file_dependency_tracking(self, integrated_setup):
        """Test tracking of dependencies across multiple files."""
        setup = integrated_setup
        graph = setup['graph']
        generator = setup['generator']
        analyzer = setup['analyzer']

        # Create files with cross-dependencies
        files_content = {
            'api.py': '''
from models import User
from serializers import UserSerializer

def get_user(user_id: int) -> User:
    """Get user by ID."""
    user = User.query.get(user_id)
    return UserSerializer.serialize(user)
''',
            'models.py': '''
from database import db

class User:
    """User model."""
    def __init__(self, id, name):
        self.id = id
        self.name = name

    @classmethod
    def query(cls):
        return UserQuery()

class UserQuery:
    def get(self, user_id):
        # Simulate database query
        return User(user_id, f"User {user_id}")
''',
            'serializers.py': '''
import json

class UserSerializer:
    @staticmethod
    def serialize(user) -> str:
        """Serialize user to JSON."""
        return json.dumps({
            'id': user.id,
            'name': user.name
        })
''',
            'database.py': '''
# Database connection module
class DBConnection:
    def __init__(self):
        self.connected = True

db = DBConnection()
'''
        }

        # Create temporary files
        temp_files = []
        try:
            for filename, content in files_content.items():
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(content)
                    temp_files.append(f.name)

            # Generate models
            for file_path in temp_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                generator.generate_from_file(file_path, content)

            # Analyze dependencies for the main API file
            api_file = [f for f in temp_files if 'api.py' in f][0]
            result = analyzer.dependency_analyzer.analyze_function_dependencies(api_file, 'get_user')

            # Verify cross-file dependencies
            assert 'dependencies' in result
            deps = result['dependencies']['get_user']

            # Should detect imports from other modules
            assert len(deps['imports_used']) >= 2  # User and UserSerializer

            # Should detect external libraries
            assert 'json' in deps['external_libraries']  # From serializers

            # Check overall analysis
            full_result = analyzer.run_comprehensive_analysis(api_file)
            assert 'dependency_analysis' in full_result
            assert 'performance_analysis' in full_result

        finally:
            for file_path in temp_files:
                os.unlink(file_path)

    def test_incremental_analysis_workflow(self, integrated_setup):
        """Test incremental analysis when files are modified."""
        setup = integrated_setup
        graph = setup['graph']
        generator = setup['generator']
        analyzer = setup['analyzer']

        # Create initial file
        initial_content = '''
def func1():
    return "initial"

def func2():
    return func1()
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(initial_content)
            temp_file = f.name

        try:
            # Initial analysis
            generator.generate_from_file(temp_file, initial_content)
            initial_result = analyzer.run_comprehensive_analysis(temp_file)

            initial_deps = initial_result['dependency_analysis']['summary']['total_functions_analyzed']

            # Modify file by adding more functions
            modified_content = initial_content + '''

def func3():
    import os
    return os.getcwd()

def func4():
    # Nested loops (bottleneck)
    result = []
    for i in range(10):
        for j in range(10):
            result.append(i * j)
    return result
'''

            with open(temp_file, 'w') as f:
                f.write(modified_content)

            # Incremental analysis
            generator.generate_from_file(temp_file, modified_content)
            modified_result = analyzer.run_comprehensive_analysis(temp_file)

            modified_deps = modified_result['dependency_analysis']['summary']['total_functions_analyzed']
            bottlenecks = modified_result['performance_analysis']['bottlenecks']

            # Should have more functions after modification
            assert modified_deps > initial_deps

            # Should detect new dependencies (os import)
            new_deps = modified_result['dependency_analysis']['dependencies']['func3']
            assert 'os' in new_deps['external_libraries']

            # Should detect performance bottlenecks in func4
            nested_bottlenecks = [b for b in bottlenecks if b.get('type') == 'nested_loops']
            assert len(nested_bottlenecks) > 0

        finally:
            os.unlink(temp_file)


class TestCacheIntegration:
    """Test integration with caching system."""

    @pytest.fixture
    def cache_integrated_setup(self):
        """Create setup with cache integration."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        # Mock cache for testing
        cache = IntelligentCache()
        generator.cache = cache
        analyzer.cache = cache

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer,
            'cache': cache
        }

    def test_cache_hit_miss_scenarios(self, cache_integrated_setup):
        """Test cache hit/miss scenarios in integrated workflow."""
        setup = cache_integrated_setup
        generator = setup['generator']
        analyzer = setup['analyzer']
        cache = setup['cache']

        # Create test file
        content = '''
import os
from typing import List

def cached_function(data: List[int]) -> int:
    """Function that should be cached."""
    return sum(data)
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # First analysis (cache miss)
            start_time = time.time()
            generator.generate_from_file(temp_file, content)
            result1 = analyzer.run_comprehensive_analysis(temp_file)
            first_time = time.time() - start_time

            # Second analysis (potential cache hit)
            start_time = time.time()
            result2 = analyzer.run_comprehensive_analysis(temp_file)
            second_time = time.time() - start_time

            # Results should be consistent
            assert result1['dependency_analysis']['summary'] == result2['dependency_analysis']['summary']

            # Second run might be faster due to caching
            # (though this depends on cache implementation)

        finally:
            os.unlink(temp_file)

    def test_cache_invalidation_scenarios(self, cache_integrated_setup):
        """Test cache invalidation in integrated scenarios."""
        setup = cache_integrated_setup
        generator = setup['generator']
        analyzer = setup['analyzer']
        cache = setup['cache']

        # Create test file
        content = '''
def func1():
    return "original"
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Initial analysis
            generator.generate_from_file(temp_file, content)
            result1 = analyzer.run_comprehensive_analysis(temp_file)

            # Modify file
            modified_content = '''
def func1():
    return "modified"

def func2():
    return "new"
'''

            with open(temp_file, 'w') as f:
                f.write(modified_content)

            # Re-analyze (should detect changes)
            generator.generate_from_file(temp_file, modified_content)
            result2 = analyzer.run_comprehensive_analysis(temp_file)

            # Should have different results
            funcs1 = result1['dependency_analysis']['summary']['total_functions_analyzed']
            funcs2 = result2['dependency_analysis']['summary']['total_functions_analyzed']
            assert funcs2 > funcs1

        finally:
            os.unlink(temp_file)


class TestMemoryIntegration:
    """Test integration with memory management."""

    @pytest.fixture
    def memory_integrated_setup(self):
        """Create setup with memory management integration."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        # Initialize memory manager
        memory_manager = MemoryManager()
        generator.memory_manager = memory_manager
        analyzer.memory_manager = memory_manager

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer,
            'memory_manager': memory_manager
        }

    def test_memory_optimization_during_large_analysis(self, memory_integrated_setup):
        """Test memory optimization during large-scale analysis."""
        setup = memory_integrated_setup
        generator = setup['generator']
        analyzer = setup['analyzer']
        memory_manager = setup['memory_manager']

        # Create large codebase
        large_content = ""
        for i in range(200):
            large_content += f"""
def func_{i}():
    data = list(range(100))
    result = []
    for item in data:
        for j in range(10):  # Nested operations
            result.append(item * j)
    return result

class Class_{i}:
    def __init__(self):
        self.data = [0] * 500  # Large instance data

    def process(self):
        return sum(self.data)
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            temp_file = f.name

        try:
            # Monitor memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            # Generate model
            generator.generate_from_file(temp_file, large_content)

            # Run analysis
            result = analyzer.run_comprehensive_analysis(temp_file)

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # Should not cause excessive memory usage
            assert memory_increase < 500  # Less than 500MB increase
            assert result is not None

            # Should have analyzed many functions
            deps = result['dependency_analysis']['summary']['total_functions_analyzed']
            assert deps >= 200  # At least 200 functions

        finally:
            os.unlink(temp_file)

    def test_memory_cleanup_integration(self, memory_integrated_setup):
        """Test memory cleanup integration."""
        setup = memory_integrated_setup
        generator = setup['generator']
        analyzer = setup['analyzer']
        memory_manager = setup['memory_manager']

        # Create test content
        content = '''
def test_func():
    return "test"
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Generate and analyze
            generator.generate_from_file(temp_file, content)
            result = analyzer.run_comprehensive_analysis(temp_file)

            # Check that analysis worked
            assert result is not None

            # Cleanup
            memory_manager.cleanup()

            # Should still be able to analyze (cleanup shouldn't break functionality)
            result2 = analyzer.run_comprehensive_analysis(temp_file)
            assert result2 is not None

        finally:
            os.unlink(temp_file)


class TestConcurrencyIntegration:
    """Test concurrent operations across components."""

    @pytest.fixture
    def concurrent_setup(self):
        """Create setup for concurrent testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_concurrent_file_processing_and_analysis(self, concurrent_setup):
        """Test concurrent file processing and analysis."""
        setup = concurrent_setup
        generator = setup['generator']
        analyzer = setup['analyzer']

        num_files = 20
        files_data = []

        # Create test files
        for i in range(num_files):
            content = f"""
import os

def func_{i}():
    return os.getcwd()

class Class_{i}:
    def method(self):
        return {i}
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                files_data.append((f.name, content))

        try:
            results = {}
            errors = []

            def process_and_analyze_file(file_path, content, index):
                """Process and analyze a single file."""
                try:
                    # Generate model
                    generator.generate_from_file(file_path, content)

                    # Run analysis
                    result = analyzer.run_comprehensive_analysis(file_path)
                    results[index] = result

                except Exception as e:
                    errors.append(f"File {index}: {e}")

            # Execute concurrently
            threads = []
            start_time = time.time()

            for i, (file_path, content) in enumerate(files_data):
                thread = threading.Thread(target=process_and_analyze_file,
                                        args=(file_path, content, i))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            concurrent_time = time.time() - start_time

            # Verify results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == num_files
            assert concurrent_time < 30  # Should complete within 30 seconds

            # Check that all analyses produced valid results
            for result in results.values():
                assert 'dependency_analysis' in result
                assert 'performance_analysis' in result

        finally:
            for file_path, _ in files_data:
                os.unlink(file_path)

    def test_shared_resource_thread_safety(self, concurrent_setup):
        """Test thread safety with shared resources."""
        setup = concurrent_setup
        graph = setup['graph']
        analyzer = setup['analyzer']

        num_threads = 15
        operations_per_thread = 25

        def concurrent_operations(thread_id):
            """Perform concurrent operations on shared resources."""
            for i in range(operations_per_thread):
                try:
                    # Add node to shared graph
                    node = CodeNode(
                        node_type=NodeType.FUNCTION,
                        name=f"shared_func_{thread_id}_{i}",
                        qualified_name=f"shared_func_{thread_id}_{i}",
                        file_path=f"/shared_{thread_id}.py",
                        start_line=i,
                        end_line=i+1
                    )
                    graph.add_node(node, LayerType.SEMANTIC)

                    # Occasionally run analysis on shared file
                    if i % 5 == 0:
                        analyzer.dependency_analyzer.analyze_function_dependencies(
                            f"/shared_{thread_id}.py"
                        )

                except Exception as e:
                    pytest.fail(f"Thread {thread_id} failed: {e}")

        # Execute concurrent operations
        threads = []
        start_time = time.time()

        for i in range(num_threads):
            thread = threading.Thread(target=concurrent_operations, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        concurrent_time = time.time() - start_time

        # Verify graph integrity
        stats = graph.get_statistics()
        expected_nodes = num_threads * operations_per_thread
        assert stats["total_nodes"] == expected_nodes
        assert concurrent_time < 15  # Should complete quickly


class TestPerformanceBenchmarksIntegration:
    """Performance benchmarks for integrated components."""

    @pytest.fixture
    def benchmark_setup(self):
        """Create setup for performance benchmarking."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_large_codebase_performance(self, benchmark_setup):
        """Benchmark performance with large codebases."""
        setup = benchmark_setup
        generator = setup['generator']
        analyzer = setup['analyzer']

        # Create large codebase
        num_files = 50
        files_data = []

        for i in range(num_files):
            content = f"""
import os, sys, json
from typing import List, Dict
from collections import Counter

def complex_func_{i}(data: List[Dict]) -> Dict:
    result = {{}}
    counter = Counter()

    # Nested operations
    for item in data:
        for j in range(5):
            key = f"{{item.get('id', 0)}}_{j}"
            result[key] = item.get('value', 0) * j
            counter[key] += 1

    return dict(counter)

class Processor_{i}:
    def __init__(self):
        self.data = [0] * 100

    def process(self):
        return sum(self.data)
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                files_data.append((f.name, content))

        try:
            # Benchmark model generation
            model_start = time.time()
            for file_path, content in files_data:
                generator.generate_from_file(file_path, content)
            model_time = time.time() - model_start

            # Benchmark analysis
            analysis_start = time.time()
            results = []
            for file_path, _ in files_data[:10]:  # Analyze first 10 files
                result = analyzer.run_comprehensive_analysis(file_path)
                results.append(result)
            analysis_time = time.time() - analysis_start

            # Performance assertions
            assert model_time < 20  # Model generation should be reasonable
            assert analysis_time < 30  # Analysis should be reasonable

            # Verify results quality
            assert len(results) == 10
            for result in results:
                assert 'dependency_analysis' in result
                assert 'performance_analysis' in result

                # Should have found dependencies
                deps = result['dependency_analysis']['summary']['total_functions_analyzed']
                assert deps >= 2  # At least 2 functions per file

        finally:
            for file_path, _ in files_data:
                os.unlink(file_path)

    def test_memory_efficiency_under_load(self, benchmark_setup):
        """Test memory efficiency under load."""
        setup = benchmark_setup
        generator = setup['generator']
        analyzer = setup['analyzer']

        # Create moderately complex content
        content = ""
        for i in range(100):
            content += f"""
def func_{i}():
    data = list(range(50))
    result = []
    for item in data:
        for j in range(5):  # Nested operations
            result.append(item * j)
    return result
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Monitor memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            # Generate model
            generator.generate_from_file(temp_file, content)

            # Run analysis
            result = analyzer.run_comprehensive_analysis(temp_file)

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # Should be memory efficient
            assert memory_increase < 200  # Less than 200MB increase
            assert result is not None

            # Should have analyzed all functions
            deps = result['dependency_analysis']['summary']['total_functions_analyzed']
            assert deps >= 100

        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])
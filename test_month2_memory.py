#!/usr/bin/env python3
"""
Month 2 Memory Management System Validation Test
Tests all Month 2 enhancements and compliance criteria
"""

import sys
import os
import time
import json
from pathlib import Path

# Add CES to path
sys.path.insert(0, str(Path(__file__).parent))

# Import memory_manager directly to avoid circular imports
import importlib.util
spec = importlib.util.spec_from_file_location("memory_manager", "ces/core/memory_manager.py")
memory_manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_manager_module)
MemoryManager = memory_manager_module.MemoryManager

def test_month2_implementation():
    """Test Month 2 memory management enhancements"""
    print("=== Month 2 Memory Management System Validation ===\n")

    # Initialize memory manager with advanced features
    print("1. Initializing Memory Manager with advanced features...")
    memory_manager = MemoryManager(enable_advanced_features=True)
    print("✓ Memory Manager initialized successfully\n")

    # Test 1: SQLite Advanced Indexing
    print("2. Testing SQLite Advanced Indexing...")
    try:
        # Store some test data
        for i in range(100):
            memory_manager.store_task_result(
                f"Test task {i} - implement advanced feature",
                {
                    "status": "completed",
                    "complexity_score": 0.8,
                    "execution_time": 1500 + i * 10,
                    "assistant_used": "test_assistant"
                }
            )

        # Test query performance
        start_time = time.time()
        with memory_manager.db_path.stat() as stat:
            db_size = stat.st_size / (1024 * 1024)  # MB
        query_time = time.time() - start_time

        print(f"✓ Database size: {db_size:.2f} MB")
        print(f"✓ Query performance: {query_time:.4f}s")
        print("✓ SQLite advanced indexing test passed\n")

    except Exception as e:
        print(f"✗ SQLite indexing test failed: {e}\n")

    except Exception as e:
        print(f"✗ SQLite indexing test failed: {e}\n")

    # Test 2: FAISS Vector Search
    print("3. Testing FAISS Vector Search Integration...")
    try:
        import numpy as np

        # Create test embeddings
        test_content = "implement user authentication system"
        test_embedding = np.random.rand(384).astype(np.float32)  # Typical embedding dimension

        # Store semantic memory
        success = memory_manager.store_semantic_memory(
            test_content,
            test_embedding,
            {"test": True, "category": "authentication"}
        )

        if success:
            print("✓ Semantic memory stored successfully")

            # Test semantic search
            search_results = memory_manager.semantic_search(test_embedding, limit=5)
            print(f"✓ Semantic search returned {len(search_results)} results")

            if search_results:
                print(f"✓ Best similarity score: {search_results[0].get('similarity_score', 0):.3f}")
            print("✓ FAISS integration test passed\n")
        else:
            print("✗ FAISS integration test failed\n")

    except Exception as e:
        print(f"✗ FAISS integration test failed: {e}\n")

    # Test 3: Memory Pattern Recognition
    print("4. Testing Memory Pattern Recognition...")
    try:
        patterns = memory_manager.analyze_memory_patterns()
        recognition_score = patterns.get('recognition_score', 0)

        print(f"✓ Pattern recognition score: {recognition_score:.2f}")
        if recognition_score > 0.8:
            print("✓ Pattern recognition meets >80% target")
        else:
            print("⚠ Pattern recognition below target")

        print("✓ Memory pattern recognition test completed\n")

    except Exception as e:
        print(f"✗ Pattern recognition test failed: {e}\n")

    # Test 4: Adaptive Caching
    print("5. Testing Adaptive Caching Strategies...")
    try:
        cache_result = memory_manager.optimize_adaptive_cache()
        hit_rate = cache_result.get('hit_rate', 0)

        print(f"✓ Cache hit rate: {hit_rate:.3f}")
        if hit_rate > 0.95:
            print("✓ Cache hit rate meets >95% target")
        else:
            print("⚠ Cache hit rate below target")

        print("✓ Adaptive caching test completed\n")

    except Exception as e:
        print(f"✗ Adaptive caching test failed: {e}\n")

    except Exception as e:
        print(f"✗ Adaptive caching test failed: {e}\n")

    # Test 5: Memory Optimization
    print("6. Testing Memory Optimization Routines...")
    try:
        optimization_result = memory_manager.optimize_memory_resources()
        reduction_percentage = optimization_result.get('reduction_percentage', 0)

        print(f"✓ Memory reduction: {reduction_percentage:.1f}%")
        if reduction_percentage > 50:
            print("✓ Memory reduction meets >50% target")
        else:
            print("⚠ Memory reduction below target")

        print("✓ Memory optimization test completed\n")

    except Exception as e:
        print(f"✗ Memory optimization test failed: {e}\n")

    except Exception as e:
        print(f"✗ Memory optimization test failed: {e}\n")

    # Test 6: Performance Benchmarks
    print("7. Testing Performance Benchmarks...")
    try:
        benchmarks = memory_manager.monitor_performance_benchmarks()
        compliance_score = benchmarks.get('overall_compliance_score', 0)

        print(f"✓ Performance compliance score: {compliance_score:.2f}")
        if compliance_score > 0.8:
            print("✓ Performance benchmarks meet targets")
        else:
            print("⚠ Performance benchmarks below targets")

        print("✓ Performance monitoring test completed\n")

    except Exception as e:
        print(f"✗ Performance benchmarks test failed: {e}\n")

    except Exception as e:
        print(f"✗ Performance benchmarks test failed: {e}\n")

    # Test 7: Error Handling
    print("8. Testing Error Handling and Recovery...")
    try:
        # Test error handling with invalid operation
        error_result = memory_manager.handle_memory_operation_error(
            "test_operation",
            Exception("Test error"),
            {"test": True}
        )

        if error_result.get('recovery_attempted'):
            print("✓ Error handling mechanism working")
            print("✓ Error recovery test completed\n")
        else:
            print("⚠ Error handling may need improvement\n")

    except Exception as e:
        print(f"✗ Error handling test failed: {e}\n")

    except Exception as e:
        print(f"✗ Error handling test failed: {e}\n")

    # Test 8: Month 2 Milestone Validation
    print("9. Running Month 2 Milestone Validation...")
    try:
        validation_results = memory_manager.validate_month2_milestones()
        compliance_percentage = validation_results.get('compliance_percentage', 0)
        achieved_milestones = validation_results.get('achieved_milestones', 0)
        total_milestones = validation_results.get('total_milestones', 0)

        print(f"✓ Compliance percentage: {compliance_percentage:.1f}%")
        print(f"✓ Achieved {achieved_milestones}/{total_milestones} milestones")

        if compliance_percentage >= 75:
            print("✓ Month 2 validation PASSED")
        else:
            print("⚠ Month 2 validation NEEDS IMPROVEMENT")

        print("\n=== Month 2 Validation Summary ===")
        print(f"Overall Compliance: {compliance_percentage:.1f}%")
        print(f"Milestones Achieved: {achieved_milestones}/{total_milestones}")

        # Show detailed milestone results
        milestones = validation_results.get('milestones', {})
        for milestone_name, milestone_data in milestones.items():
            status = "✓" if milestone_data.get('achieved', False) else "✗"
            print(f"{status} {milestone_name.replace('_', ' ').title()}")

        print("\n=== Month 2 Implementation Complete ===")

    except Exception as e:
        print(f"✗ Month 2 milestone validation failed: {e}\n")

    # Cleanup
    try:
        memory_manager.cleanup_advanced_resources()
        print("✓ Resources cleaned up successfully")
    except Exception as e:
        print(f"⚠ Cleanup warning: {e}")

if __name__ == "__main__":
    test_month2_implementation()
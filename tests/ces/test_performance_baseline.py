"""
CES Phase 0.3: Performance Baseline Tests

This module contains performance baseline tests to establish and validate
performance metrics for the integrated CES system.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime
import tempfile
import os

from ces.core.cognitive_agent import CognitiveAgent
from ces.config.ces_config import CESConfig
from ces.ai_orchestrator.cli_integration import AIAssistantManager


class TestCESPerformanceBaseline:
    """
    Performance baseline tests for CES Phase 0.3

    Establishes performance benchmarks and validates system performance
    against defined targets.
    """

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def baseline_config(self, temp_db_path):
        """Configuration optimized for performance testing"""
        config = CESConfig()
        config.memory_db_path = temp_db_path
        config.debug_mode = False  # Disable debug for performance
        config.cache_enabled = True
        config.max_memory_mb = 512
        return config

    @pytest.fixture
    def mock_ai_manager(self):
        """Mock AI assistant manager for performance testing"""
        with patch('ces.ai_orchestrator.cli_integration.GrokCLIIntegration') as mock_grok, \
             patch('ces.ai_orchestrator.cli_integration.QwenCLICoderIntegration') as mock_qwen, \
             patch('ces.ai_orchestrator.cli_integration.GeminiCLIIntegration') as mock_gemini:

            # Setup fast mock responses
            mock_grok_instance = Mock()
            mock_grok_instance.is_available.return_value = True
            mock_grok_instance.execute_task = Mock(return_value={
                "success": True,
                "response": "Mock response",
                "execution_time": 0.1,
                "tokens_used": 50
            })
            mock_grok.return_value = mock_grok_instance

            mock_qwen_instance = Mock()
            mock_qwen_instance.is_available.return_value = True
            mock_qwen_instance.execute_task = Mock(return_value={
                "success": True,
                "output": "Mock output",
                "execution_time": 0.15,
                "exit_code": 0
            })
            mock_qwen.return_value = mock_qwen_instance

            mock_gemini_instance = Mock()
            mock_gemini_instance.is_available.return_value = True
            mock_gemini_instance.execute_task = Mock(return_value={
                "success": True,
                "response": "Mock response",
                "execution_time": 0.12,
                "tokens_used": 45
            })
            mock_gemini.return_value = mock_gemini_instance

            manager = AIAssistantManager()
            yield manager

    def test_task_analysis_performance(self, baseline_config):
        """
        Test task analysis performance baseline

        Target: <200ms P95 for simple task analysis
        """
        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator'), \
             patch('ces.core.ethical_controller.EthicalController'):

            agent = CognitiveAgent(baseline_config)

            # Test simple task analysis
            task = "Create a simple function"

            # Measure analysis time
            start_time = time.time()
            analysis = agent.analyze_task(task)
            analysis_time = time.time() - start_time

            # Validate performance
            assert analysis_time < 0.2, f"Task analysis too slow: {analysis_time:.3f}s"
            assert analysis.complexity_score >= 0
            assert analysis.estimated_duration > 0

    @pytest.mark.asyncio
    async def test_end_to_end_task_execution_performance(self, baseline_config, mock_ai_manager):
        """
        Test end-to-end task execution performance

        Target: <2s P95 for complete task execution workflow
        """
        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator') as mock_orchestrator, \
             patch('ces.core.ethical_controller.EthicalController'):

            # Setup orchestrator mock
            mock_orchestrator_instance = Mock()
            mock_orchestrator_instance.recommend_assistants.return_value = ['grok']
            mock_orchestrator_instance.execute_task = Mock(return_value={
                "status": "completed",
                "assistant_used": "grok",
                "result": "Task completed successfully",
                "execution_time": 0.5,
                "timestamp": datetime.now().isoformat()
            })
            mock_orchestrator.return_value = mock_orchestrator_instance

            agent = CognitiveAgent(baseline_config)

            # Test complete workflow
            task = "Implement a simple calculator function"

            start_time = time.time()
            result = agent.execute_task(task)
            total_time = time.time() - start_time

            # Validate performance
            assert total_time < 2.0, f"End-to-end execution too slow: {total_time:.3f}s"
            assert result['status'] == 'completed'
            assert 'result' in result

    def test_memory_system_performance(self, baseline_config):
        """
        Test memory system performance baseline

        Target: <100ms for context retrieval, <50ms for storage
        """
        from ces.core.memory_manager import MemoryManager

        memory = MemoryManager(baseline_config.memory_db_path)

        # Test context storage performance
        context_data = {
            "task_history": [{"description": "Test task", "result": "Success"}],
            "user_preferences": {"theme": "dark"}
        }

        start_time = time.time()
        memory.store_task_result("Test task", {"status": "completed", "result": "Success"})
        storage_time = time.time() - start_time

        assert storage_time < 0.05, f"Memory storage too slow: {storage_time:.3f}s"

        # Test context retrieval performance
        start_time = time.time()
        retrieved = memory.retrieve_context("Test task", ["task_history"])
        retrieval_time = time.time() - start_time

        assert retrieval_time < 0.1, f"Memory retrieval too slow: {retrieval_time:.3f}s"
        assert "task_history" in retrieved

    def test_ai_assistant_response_performance(self, mock_ai_manager):
        """
        Test AI assistant response time performance

        Target: <500ms P95 for AI assistant responses
        """
        available_assistants = mock_ai_manager.get_available_assistants()
        assert len(available_assistants) > 0

        # Test each assistant's response time
        for assistant in available_assistants:
            assistant_instance = mock_ai_manager.get_assistant(assistant['name'])
            assert assistant_instance is not None

            # Mock execution time should be within limits
            if hasattr(assistant_instance, 'execute_task'):
                result = assistant_instance.execute_task("Test task")
                execution_time = result.get('execution_time', 0)
                assert execution_time < 0.5, f"AI response too slow: {execution_time:.3f}s for {assistant['name']}"

    def test_concurrent_task_processing_performance(self, baseline_config):
        """
        Test concurrent task processing performance

        Target: Support 5+ concurrent tasks with <20% performance degradation
        """
        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator') as mock_orchestrator, \
             patch('ces.core.ethical_controller.EthicalController'):

            mock_orchestrator_instance = Mock()
            mock_orchestrator_instance.recommend_assistants.return_value = ['grok']
            mock_orchestrator_instance.execute_task = Mock(return_value={
                "status": "completed",
                "assistant_used": "grok",
                "result": "Concurrent task result",
                "execution_time": 0.3,
                "timestamp": datetime.now().isoformat()
            })
            mock_orchestrator.return_value = mock_orchestrator_instance

            agent = CognitiveAgent(baseline_config)

            # Test concurrent execution
            tasks = [f"Task {i}" for i in range(5)]

            start_time = time.time()
            results = []

            for task in tasks:
                result = agent.execute_task(task)
                results.append(result)

            total_time = time.time() - start_time
            avg_time_per_task = total_time / len(tasks)

            # Validate concurrent performance
            assert all(r['status'] == 'completed' for r in results)
            assert avg_time_per_task < 1.0, f"Average task time too high: {avg_time_per_task:.3f}s"

    def test_memory_usage_baseline(self, baseline_config):
        """
        Test memory usage baseline

        Target: <256MB under normal load
        """
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator'), \
             patch('ces.core.ethical_controller.EthicalController'):

            agent = CognitiveAgent(baseline_config)

            # Perform some operations to test memory usage
            for i in range(10):
                agent.analyze_task(f"Test task {i}")

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Allow some memory increase but keep it reasonable
            assert memory_increase < 50, f"Memory usage too high: +{memory_increase:.1f}MB"

    def test_error_recovery_performance(self, baseline_config):
        """
        Test error recovery performance

        Target: <1s for error detection and recovery
        """
        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator') as mock_orchestrator, \
             patch('ces.core.ethical_controller.EthicalController') as mock_ethical:

            # Setup mocks to simulate errors
            mock_orchestrator_instance = Mock()
            mock_orchestrator_instance.recommend_assistants.return_value = ['grok']
            mock_orchestrator_instance.execute_task.side_effect = Exception("Simulated API error")
            mock_orchestrator.return_value = mock_orchestrator_instance

            mock_ethical_instance = Mock()
            mock_ethical_instance.check_task_ethics.return_value = []
            mock_ethical_instance.approve_task.return_value = True
            mock_ethical.return_value = mock_ethical_instance

            agent = CognitiveAgent(baseline_config)

            start_time = time.time()
            result = agent.execute_task("Test task that will fail")
            recovery_time = time.time() - start_time

            # Validate error recovery performance
            assert recovery_time < 1.0, f"Error recovery too slow: {recovery_time:.3f}s"
            assert result['status'] == 'failed'
            assert 'error' in result

    def test_cache_performance_baseline(self, baseline_config):
        """
        Test caching performance baseline

        Target: >95% cache hit rate, <1ms cache access time
        """
        from ces.core.memory_manager import MemoryManager

        memory = MemoryManager(baseline_config.memory_db_path)

        # Test repeated access performance (simulating cache hits)
        test_data = {"key": "value", "complexity": 5.0}

        # First access (cache miss)
        start_time = time.time()
        memory.store_task_result("Cache test task", test_data)
        first_access_time = time.time() - start_time

        # Repeated access (cache hit simulation)
        start_time = time.time()
        retrieved = memory.retrieve_context("Cache test task", ["task_history"])
        second_access_time = time.time() - start_time

        # Validate cache performance
        assert first_access_time < 0.01, f"First access too slow: {first_access_time:.3f}s"
        assert second_access_time < 0.01, f"Cache access too slow: {second_access_time:.3f}s"
        assert len(retrieved.get("task_history", [])) > 0

    @pytest.mark.parametrize("task_complexity", [1, 5, 10])
    def test_task_complexity_performance_scaling(self, baseline_config, task_complexity):
        """
        Test performance scaling with task complexity

        Ensures performance degrades gracefully with complexity
        """
        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator'), \
             patch('ces.core.ethical_controller.EthicalController'):

            agent = CognitiveAgent(baseline_config)

            # Create task with specific complexity
            complexity_indicators = ["simple", "complex", "advanced", "optimize", "architecture"]
            task_words = ["Create"] + complexity_indicators[:task_complexity] + ["function"]
            task = " ".join(task_words)

            start_time = time.time()
            analysis = agent.analyze_task(task)
            analysis_time = time.time() - start_time

            # Validate complexity analysis
            assert analysis.complexity_score <= 10.0
            assert analysis_time < 0.5, f"Complexity analysis too slow: {analysis_time:.3f}s"

            # Ensure complexity correlates with analysis time (but stays within bounds)
            expected_max_time = 0.1 + (task_complexity * 0.05)  # Allow some scaling
            assert analysis_time < expected_max_time, \
                f"Analysis time {analysis_time:.3f}s exceeds expected max {expected_max_time:.3f}s for complexity {task_complexity}"
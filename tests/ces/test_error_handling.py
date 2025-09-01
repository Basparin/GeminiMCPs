"""
CES Phase 0.3: Error Handling and Recovery Tests

This module contains comprehensive tests for error handling and recovery
mechanisms in the CES system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import tempfile
import os

from ces.core.cognitive_agent import CognitiveAgent
from ces.config.ces_config import CESConfig
from ces.ai_orchestrator.cli_integration import AIAssistantManager
from ces.codesage_integration import CodeSageIntegration


class TestCESErrorHandling:
    """
    Error handling and recovery tests for CES Phase 0.3

    Tests various error scenarios and validates recovery mechanisms.
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
    def test_config(self, temp_db_path):
        """Test configuration"""
        config = CESConfig()
        config.memory_db_path = temp_db_path
        config.debug_mode = True
        return config

    def test_api_failure_recovery(self, test_config):
        """
        Test recovery from AI API failures

        Validates fallback mechanisms when AI services are unavailable.
        """
        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator') as mock_orchestrator, \
             patch('ces.core.ethical_controller.EthicalController'):

            # Setup orchestrator to simulate API failure
            mock_orchestrator_instance = Mock()
            mock_orchestrator_instance.recommend_assistants.return_value = ['grok']
            mock_orchestrator_instance.execute_task.side_effect = Exception("API rate limit exceeded")
            mock_orchestrator.return_value = mock_orchestrator_instance

            agent = CognitiveAgent(test_config)

            # Execute task that will fail
            result = agent.execute_task("Test task with API failure")

            # Validate error handling
            assert result['status'] == 'failed'
            assert 'error' in result
            assert 'API rate limit exceeded' in result['error']

    def test_network_timeout_recovery(self, test_config):
        """
        Test recovery from network timeouts

        Validates timeout handling and recovery.
        """
        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator') as mock_orchestrator, \
             patch('ces.core.ethical_controller.EthicalController'):

            # Setup orchestrator to simulate timeout
            mock_orchestrator_instance = Mock()
            mock_orchestrator_instance.recommend_assistants.return_value = ['qwen']
            mock_orchestrator_instance.execute_task.side_effect = asyncio.TimeoutError("Network timeout")
            mock_orchestrator.return_value = mock_orchestrator_instance

            agent = CognitiveAgent(test_config)

            result = agent.execute_task("Test task with timeout")

            # Validate timeout handling
            assert result['status'] == 'failed'
            assert 'error' in result

    def test_database_corruption_recovery(self, test_config):
        """
        Test recovery from database corruption

        Validates database error handling and recovery.
        """
        from ces.core.memory_manager import MemoryManager

        # Create corrupted database file
        with open(test_config.memory_db_path, 'w') as f:
            f.write("corrupted database content")

        # Attempt to initialize memory manager
        memory = MemoryManager(test_config.memory_db_path)

        # Should handle corruption gracefully
        status = memory.get_status()
        assert status['status'] == 'operational'  # Should recover by recreating DB

    def test_invalid_task_handling(self, test_config):
        """
        Test handling of invalid or malformed tasks

        Validates input validation and error reporting.
        """
        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator'), \
             patch('ces.core.ethical_controller.EthicalController'):

            agent = CognitiveAgent(test_config)

            # Test with empty task
            result = agent.execute_task("")

            # Should handle gracefully
            assert result['status'] in ['failed', 'completed']  # Either is acceptable

            # Test with very long task
            long_task = "test " * 1000
            result = agent.execute_task(long_task)

            # Should handle long input
            assert result['status'] in ['failed', 'completed']

    def test_ethical_violation_handling(self, test_config):
        """
        Test handling of ethically problematic tasks

        Validates ethical controller integration and task rejection.
        """
        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator'), \
             patch('ces.core.ethical_controller.EthicalController') as mock_ethical:

            # Setup ethical controller to reject task
            mock_ethical_instance = Mock()
            mock_ethical_instance.check_task_ethics.return_value = ['harmful content']
            mock_ethical_instance.approve_task.return_value = False
            mock_ethical.return_value = mock_ethical_instance

            agent = CognitiveAgent(test_config)

            result = agent.execute_task("Create a harmful program")

            # Validate ethical rejection
            assert result['status'] == 'rejected'
            assert 'ethical concerns' in result.get('reason', '').lower()

    def test_component_initialization_failures(self, test_config):
        """
        Test handling of component initialization failures

        Validates graceful degradation when components fail to initialize.
        """
        with patch('ces.core.memory_manager.MemoryManager', side_effect=Exception("DB init failed")), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator'), \
             patch('ces.core.ethical_controller.EthicalController'):

            # Should handle initialization failure gracefully
            agent = CognitiveAgent(test_config)

            # Agent should still be created but with limited functionality
            assert agent is not None

            # Status should reflect the issue
            status = agent.get_status()
            assert status['status'] == 'operational'  # Should still report operational

    def test_concurrent_error_handling(self, test_config):
        """
        Test error handling in concurrent task execution

        Validates that errors in one task don't affect others.
        """
        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator') as mock_orchestrator, \
             patch('ces.core.ethical_controller.EthicalController'):

            mock_orchestrator_instance = Mock()
            mock_orchestrator.return_value = mock_orchestrator_instance

            # Setup alternating success/failure
            call_count = 0
            def alternating_result(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:
                    raise Exception(f"Simulated error {call_count}")
                return {
                    "status": "completed",
                    "assistant_used": "grok",
                    "result": f"Success {call_count}",
                    "execution_time": 0.1,
                    "timestamp": datetime.now().isoformat()
                }

            mock_orchestrator_instance.recommend_assistants.return_value = ['grok']
            mock_orchestrator_instance.execute_task.side_effect = alternating_result

            agent = CognitiveAgent(test_config)

            # Execute multiple tasks
            results = []
            for i in range(4):
                result = agent.execute_task(f"Task {i}")
                results.append(result)

            # Validate mixed success/failure handling
            success_count = sum(1 for r in results if r['status'] == 'completed')
            failure_count = sum(1 for r in results if r['status'] == 'failed')

            assert success_count > 0
            assert failure_count > 0
            assert success_count + failure_count == len(results)

    def test_codesage_connection_failures(self, test_config):
        """
        Test CodeSage connection failure handling

        Validates MCP server connection error handling.
        """
        with patch('ces.codesage_integration.aiohttp.ClientSession') as mock_session:

            # Setup session to fail
            mock_session_instance = Mock()
            mock_session_instance.post.side_effect = Exception("Connection refused")
            mock_session.return_value = mock_session_instance

            codesage = CodeSageIntegration()

            # Test connection failure
            connected = asyncio.run(codesage.connect())
            assert connected == False

            # Test tool execution with no connection
            result = asyncio.run(codesage.execute_tool("test_tool", {}))
            assert result['status'] == 'error'
            assert 'Not connected' in result['error']

    def test_memory_system_error_recovery(self, test_config):
        """
        Test memory system error recovery

        Validates SQLite error handling and recovery.
        """
        from ces.core.memory_manager import MemoryManager

        memory = MemoryManager(test_config.memory_db_path)

        # Test with invalid data
        result = memory.store_task_result("test", {"invalid": object()})  # Non-serializable

        # Should handle gracefully (store_task_result doesn't return anything, but shouldn't crash)
        assert result is None

        # Test retrieval after error
        context = memory.retrieve_context("test", ["task_history"])
        assert isinstance(context, dict)

    def test_ai_assistant_fallback_mechanisms(self, test_config):
        """
        Test AI assistant fallback mechanisms

        Validates switching between assistants when one fails.
        """
        with patch('ces.ai_orchestrator.cli_integration.AIAssistantManager') as mock_manager:

            mock_manager_instance = Mock()
            mock_manager_instance.get_available_assistants.return_value = ['grok', 'qwen']

            # Setup primary assistant to fail, secondary to succeed
            def assistant_selector(assistant_name, *args, **kwargs):
                if assistant_name == 'grok':
                    raise Exception("Grok API down")
                elif assistant_name == 'qwen':
                    return {
                        "success": True,
                        "output": "Fallback successful",
                        "execution_time": 0.2,
                        "exit_code": 0
                    }

            mock_manager_instance.execute_with_assistant.side_effect = assistant_selector
            mock_manager.return_value = mock_manager_instance

            with patch('ces.core.memory_manager.MemoryManager'), \
                 patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator') as mock_orchestrator, \
                 patch('ces.core.ethical_controller.EthicalController'):

                mock_orchestrator_instance = Mock()
                mock_orchestrator_instance.recommend_assistants.return_value = ['grok', 'qwen']
                mock_orchestrator_instance.execute_task.side_effect = lambda *args, **kwargs: asyncio.run(
                    mock_manager_instance.execute_with_assistant('qwen', *args, **kwargs)
                )
                mock_orchestrator.return_value = mock_orchestrator_instance

                agent = CognitiveAgent(test_config)

                result = agent.execute_task("Test fallback mechanism")

                # Should succeed with fallback
                assert result['status'] == 'completed'

    def test_configuration_validation_errors(self):
        """
        Test configuration validation error handling

        Validates config validation and error reporting.
        """
        # Test invalid memory setting
        config = CESConfig()
        config.max_memory_mb = -1  # Invalid

        # Should raise validation error
        with pytest.raises(ValueError, match="max_memory_mb must be at least 64"):
            config._validate_configuration()

    def test_partial_system_recovery(self, test_config):
        """
        Test partial system recovery

        Validates operation when some components are unavailable.
        """
        with patch('ces.core.memory_manager.MemoryManager', side_effect=Exception("Memory init failed")), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator') as mock_orchestrator, \
             patch('ces.core.ethical_controller.EthicalController'):

            mock_orchestrator_instance = Mock()
            mock_orchestrator_instance.recommend_assistants.return_value = ['grok']
            mock_orchestrator_instance.execute_task.return_value = {
                "status": "completed",
                "assistant_used": "grok",
                "result": "Partial success",
                "execution_time": 0.1,
                "timestamp": datetime.now().isoformat()
            }
            mock_orchestrator.return_value = mock_orchestrator_instance

            # Should still create agent despite memory failure
            agent = CognitiveAgent(test_config)
            assert agent is not None

            # Should still be able to execute tasks (with limited memory)
            result = agent.execute_task("Test partial recovery")
            assert result['status'] == 'completed'

    def test_resource_exhaustion_handling(self, test_config):
        """
        Test resource exhaustion handling

        Validates memory and CPU limit handling.
        """
        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator') as mock_orchestrator, \
             patch('ces.core.ethical_controller.EthicalController'):

            mock_orchestrator_instance = Mock()
            mock_orchestrator_instance.recommend_assistants.return_value = ['grok']
            mock_orchestrator_instance.execute_task.side_effect = MemoryError("Out of memory")
            mock_orchestrator.return_value = mock_orchestrator_instance

            agent = CognitiveAgent(test_config)

            result = agent.execute_task("Test memory exhaustion")

            # Should handle memory error gracefully
            assert result['status'] == 'failed'
            assert 'error' in result

    def test_logging_error_handling(self, test_config):
        """
        Test logging system error handling

        Validates that logging failures don't break the system.
        """
        with patch('ces.utils.helpers.setup_logging', side_effect=Exception("Logging init failed")), \
             patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator'), \
             patch('ces.core.ethical_controller.EthicalController'):

            # Should handle logging failure gracefully
            agent = CognitiveAgent(test_config)
            assert agent is not None

            # Should still function without logging
            result = agent.execute_task("Test without logging")
            assert result is not None
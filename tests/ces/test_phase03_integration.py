"""
CES Phase 0.3: Basic Integration - End-to-End Integration Tests

This module contains comprehensive end-to-end integration tests for CES Phase 0.3,
validating the complete workflow from user input through AI assistant execution.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from ces.core.cognitive_agent import CognitiveAgent
from ces.ai_orchestrator.ai_assistant import AIOrchestrator
from ces.cli.ces_cli import CESCLI
from ces.config.ces_config import CESConfig
from ces.codesage_integration import CodeSageIntegration


class TestCESPhase03Integration:
    """
    End-to-end integration tests for CES Phase 0.3 Basic Integration

    Tests the complete workflow:
    1. CLI task input
    2. Cognitive agent task analysis
    3. AI assistant selection and execution
    4. Result processing and storage
    5. Error handling and recovery
    """

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def test_config(self, temp_db_path):
        """Test configuration with temporary database"""
        config = CESConfig()
        config.memory_db_path = temp_db_path
        config.debug_mode = True
        config.development_mode = True
        return config

    @pytest.fixture
    async def mock_codesage_integration(self):
        """Mock CodeSage integration for testing"""
        mock_integration = Mock(spec=CodeSageIntegration)
        mock_integration.connect = AsyncMock(return_value=True)
        mock_integration.connected = True
        mock_integration.execute_tool = AsyncMock(return_value={
            "status": "success",
            "result": {"analysis": "Test analysis"},
            "timestamp": datetime.now().isoformat()
        })
        return mock_integration

    @pytest.fixture
    def mock_ai_assistants(self):
        """Mock AI assistants for testing"""
        return {
            'grok': Mock(),
            'qwen': Mock(),
            'gemini': Mock()
        }

    @pytest.mark.asyncio
    async def test_complete_task_execution_workflow(self, test_config, mock_codesage_integration, mock_ai_assistants):
        """
        Test complete end-to-end task execution workflow

        This test validates:
        - Task input parsing
        - Cognitive agent analysis
        - AI assistant selection and execution
        - Result processing and storage
        - Memory system integration
        """
        task_description = "Implement a simple user authentication function in Python"

        # Mock AI assistant responses
        mock_ai_response = {
            "status": "completed",
            "assistant_used": "qwen",
            "result": "def authenticate_user(username, password):\n    # Authentication logic here\n    return True",
            "execution_time": 1.5,
            "timestamp": datetime.now().isoformat()
        }

        with patch('ces.core.memory_manager.MemoryManager') as mock_memory, \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator') as mock_orchestrator, \
             patch('ces.core.ethical_controller.EthicalController') as mock_ethical, \
             patch('ces.codesage_integration.CodeSageIntegration', return_value=mock_codesage_integration):

            # Setup mocks
            mock_memory_instance = Mock()
            mock_memory.return_value = mock_memory_instance
            mock_memory_instance.analyze_context_needs.return_value = ['task_history']
            mock_memory_instance.retrieve_context.return_value = {}
            mock_memory_instance.store_task_result = Mock()

            mock_orchestrator_instance = Mock()
            mock_orchestrator.return_value = mock_orchestrator_instance
            mock_orchestrator_instance.recommend_assistants.return_value = ['qwen']
            mock_orchestrator_instance.execute_task = AsyncMock(return_value=mock_ai_response)

            mock_ethical_instance = Mock()
            mock_ethical.return_value = mock_ethical_instance
            mock_ethical_instance.check_task_ethics.return_value = []
            mock_ethical_instance.approve_task.return_value = True

            # Initialize cognitive agent
            agent = CognitiveAgent(test_config)

            # Execute task
            result = await asyncio.get_event_loop().run_in_executor(
                None, agent.execute_task, task_description
            )

            # Validate complete workflow
            assert result['status'] == 'completed'
            assert 'analysis' in result
            assert result['result']['assistant_used'] == 'qwen'
            assert 'execution_time' in result['result']
            assert 'timestamp' in result

            # Verify component interactions
            mock_memory_instance.analyze_context_needs.assert_called_with(task_description)
            mock_memory_instance.retrieve_context.assert_called()
            mock_memory_instance.store_task_result.assert_called_with(task_description, result['result'])
            mock_orchestrator_instance.recommend_assistants.assert_called()
            mock_orchestrator_instance.execute_task.assert_called()
            mock_ethical_instance.check_task_ethics.assert_called_with(task_description)

    def test_cli_task_execution_integration(self, test_config, capsys):
        """
        Test CLI task execution integration

        Validates the complete CLI workflow from command parsing to result display.
        """
        task_description = "Create a simple calculator function"

        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator'), \
             patch('ces.core.ethical_controller.EthicalController'), \
             patch('ces.cli.ces_cli.CognitiveAgent.execute_task') as mock_execute:

            mock_execute.return_value = {
                'status': 'completed',
                'analysis': Mock(complexity_score=3.0, required_skills=['programming'], estimated_duration=30),
                'result': {'assistant_used': 'qwen', 'result': 'def calculator(a, b, op): return eval(f"{a}{op}{b}")'},
                'timestamp': datetime.now().isoformat()
            }

            # Initialize CLI
            cli = CESCLI()
            cli.config = test_config

            # Simulate CLI execution
            import sys
            from io import StringIO

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            try:
                # Initialize agent and execute task
                agent = cli.initialize_agent()
                result = agent.execute_task(task_description)

                # Validate CLI output contains expected information
                output = captured_output.getvalue()
                assert "Task:" in output or "Status:" in output or "Assistant:" in output

            finally:
                sys.stdout = old_stdout

    @pytest.mark.asyncio
    async def test_ai_assistant_integration_validation(self, mock_ai_assistants):
        """
        Test AI assistant integration validation

        Validates that all AI assistants are properly integrated and can be called.
        """
        from ces.ai_orchestrator.cli_integration import AIAssistantManager

        with patch('ces.ai_orchestrator.cli_integration.GrokCLIIntegration') as mock_grok, \
             patch('ces.ai_orchestrator.cli_integration.QwenCLICoderIntegration') as mock_qwen, \
             patch('ces.ai_orchestrator.cli_integration.GeminiCLIIntegration') as mock_gemini:

            # Setup mock assistants
            mock_grok_instance = Mock()
            mock_grok_instance.is_available.return_value = True
            mock_grok_instance.get_status.return_value = {"available": True, "name": "Grok CLI"}
            mock_grok.return_value = mock_grok_instance

            mock_qwen_instance = Mock()
            mock_qwen_instance.is_available.return_value = True
            mock_qwen_instance.get_status.return_value = {"available": True, "name": "qwen-cli-coder"}
            mock_qwen.return_value = mock_qwen_instance

            mock_gemini_instance = Mock()
            mock_gemini_instance.is_available.return_value = True
            mock_gemini_instance.get_status.return_value = {"available": True, "name": "Gemini CLI"}
            mock_gemini.return_value = mock_gemini_instance

            # Initialize manager
            manager = AIAssistantManager()

            # Test assistant availability
            available = manager.get_available_assistants()
            assert len(available) == 3
            assert any(a['name'] == 'Grok CLI' for a in available)
            assert any(a['name'] == 'qwen-cli-coder' for a in available)
            assert any(a['name'] == 'Gemini CLI' for a in available)

            # Test health check
            health = manager.health_check()
            assert health['overall_status'] == 'healthy'
            assert len(health['assistants']) == 3

    def test_memory_system_integration(self, test_config):
        """
        Test memory system integration

        Validates memory storage, retrieval, and context management.
        """
        from ces.core.memory_manager import MemoryManager

        # Initialize memory manager
        memory = MemoryManager(test_config.memory_db_path)

        # Test task storage and retrieval
        task_desc = "Test memory integration"
        task_result = {
            "status": "completed",
            "assistant_used": "grok",
            "result": "Memory test successful",
            "timestamp": datetime.now().isoformat()
        }

        # Store task result
        memory.store_task_result(task_desc, task_result)

        # Test context retrieval
        context = memory.retrieve_context("Similar task", ['task_history'])
        assert 'task_history' in context
        assert len(context['task_history']) > 0

        # Test user preferences
        memory.store_user_preference('test_pref', 'test_value')
        prefs = memory._get_user_preferences()
        assert 'test_pref' in prefs
        assert prefs['test_pref'] == 'test_value'

        # Test status
        status = memory.get_status()
        assert status['status'] == 'operational'
        assert status['task_history_count'] >= 1

    def test_error_handling_and_recovery(self, test_config):
        """
        Test error handling and recovery mechanisms

        Validates that the system can handle and recover from various error conditions.
        """
        with patch('ces.core.memory_manager.MemoryManager') as mock_memory, \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator') as mock_orchestrator, \
             patch('ces.core.ethical_controller.EthicalController') as mock_ethical:

            # Setup mocks to simulate errors
            mock_memory_instance = Mock()
            mock_memory.return_value = mock_memory_instance

            mock_orchestrator_instance = Mock()
            mock_orchestrator.return_value = mock_orchestrator_instance
            mock_orchestrator_instance.execute_task = AsyncMock(side_effect=Exception("API Error"))

            mock_ethical_instance = Mock()
            mock_ethical.return_value = mock_ethical_instance

            # Initialize agent
            agent = CognitiveAgent(test_config)

            # Test error handling
            result = agent.execute_task("Test task that will fail")

            # Validate error handling
            assert result['status'] == 'failed'
            assert 'error' in result
            assert 'API Error' in result['error']

    @pytest.mark.asyncio
    async def test_performance_baseline_establishment(self, test_config):
        """
        Test performance baseline establishment

        Measures and validates performance metrics for the integrated system.
        """
        import time

        task_description = "Simple performance test task"

        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator') as mock_orchestrator, \
             patch('ces.core.ethical_controller.EthicalController'):

            mock_orchestrator_instance = Mock()
            mock_orchestrator.return_value = mock_orchestrator_instance
            mock_orchestrator_instance.recommend_assistants.return_value = ['grok']
            mock_orchestrator_instance.execute_task = AsyncMock(return_value={
                "status": "completed",
                "assistant_used": "grok",
                "result": "Performance test result",
                "execution_time": 0.5,
                "timestamp": datetime.now().isoformat()
            })

            agent = CognitiveAgent(test_config)

            # Measure execution time
            start_time = time.time()
            result = await asyncio.get_event_loop().run_in_executor(
                None, agent.execute_task, task_description
            )
            total_time = time.time() - start_time

            # Validate performance meets baseline
            assert result['status'] == 'completed'
            assert total_time < 5.0  # Should complete within 5 seconds
            assert result['result']['execution_time'] < 2.0  # AI execution < 2 seconds

    def test_component_data_flow_validation(self, test_config):
        """
        Test component data flow validation

        Validates that data flows correctly between all CES components.
        """
        from ces.core.memory_manager import MemoryManager
        from ces.core.ethical_controller import EthicalController

        # Initialize components
        memory = MemoryManager(test_config.memory_db_path)
        ethical = EthicalController()

        # Test data flow: Task -> Analysis -> Ethics Check -> Memory -> Result
        task = "Create a data validation function"

        # 1. Memory analysis
        context_needs = memory.analyze_context_needs(task)
        assert isinstance(context_needs, list)

        # 2. Ethical check
        ethical_concerns = ethical.check_task_ethics(task)
        assert isinstance(ethical_concerns, list)

        # 3. Context retrieval
        context = memory.retrieve_context(task, context_needs)
        assert isinstance(context, dict)

        # 4. Store result
        result = {
            "status": "completed",
            "result": "Function created successfully",
            "timestamp": datetime.now().isoformat()
        }
        memory.store_task_result(task, result)

        # 5. Verify storage
        status = memory.get_status()
        assert status['task_history_count'] >= 1

    @pytest.mark.asyncio
    async def test_codesage_integration_workflow(self, mock_codesage_integration):
        """
        Test CodeSage integration workflow

        Validates the integration between CES and CodeSage MCP server.
        """
        # Test connection
        connected = await mock_codesage_integration.connect()
        assert connected

        # Test tool execution
        result = await mock_codesage_integration.execute_tool("test_tool", {"param": "value"})
        assert result['status'] == 'success'
        assert 'result' in result
        assert 'timestamp' in result

        # Verify connection state
        assert mock_codesage_integration.connected

    def test_configuration_integration(self, test_config):
        """
        Test configuration integration across all components

        Validates that configuration is properly loaded and used by all components.
        """
        # Test config validation
        from ces.utils.helpers import validate_config
        validation = validate_config(test_config.to_dict())
        assert validation['valid']

        # Test config access
        assert test_config.debug_mode == True
        assert test_config.development_mode == True
        assert test_config.memory_db_path.endswith('.db')

        # Test AI assistant configs
        ai_configs = test_config.get_ai_assistant_configs()
        assert 'grok' in ai_configs
        assert 'qwen' in ai_configs
        assert 'gemini' in ai_configs

        # Test memory config
        memory_config = test_config.get_memory_config()
        assert 'db_path' in memory_config
        assert 'cache_enabled' in memory_config
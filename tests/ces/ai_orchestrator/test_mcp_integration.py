"""
Test MCP Protocol Integration and AI Assistant Functionality

Tests for Phase 0.2: MCP Protocol Integration
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from ces.ai_orchestrator.cli_integration import (
    AIAssistantManager,
    GrokCLIIntegration,
    QwenCLICoderIntegration,
    GeminiCLIIntegration,
    APIResult,
    CLIResult
)
from ces.ai_orchestrator.ai_assistant import AIOrchestrator
from ces.ai_orchestrator.task_delegation import TaskDelegator


class TestGrokCLIIntegration:
    """Test Grok CLI integration"""

    def test_initialization_without_api_key(self):
        """Test initialization without API key"""
        integration = GrokCLIIntegration(api_key=None)
        assert not integration.is_available()
        assert integration.api_key is None

    def test_initialization_with_api_key(self):
        """Test initialization with API key"""
        api_key = "test_key"
        integration = GrokCLIIntegration(api_key=api_key)
        assert integration.api_key == api_key
        # Note: is_available() would be False without real client setup

    @patch('ces.ai_orchestrator.cli_integration.Groq')
    def test_successful_api_call(self, mock_groq):
        """Test successful API call"""
        # Mock the Groq client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.total_tokens = 100
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_client

        integration = GrokCLIIntegration(api_key="test_key")
        integration.client = mock_client

        async def run_test():
            result = await integration.execute_task("Test task")

            assert isinstance(result, APIResult)
            assert result.success
            assert result.response == "Test response"
            assert result.tokens_used == 100

        asyncio.run(run_test())

    def test_get_status(self):
        """Test get_status method"""
        integration = GrokCLIIntegration(api_key="test_key")
        status = integration.get_status()

        assert status["name"] == "Grok CLI"
        assert status["model"] == "mixtral-8x7b-32768"
        assert "has_api_key" in status
        assert "last_check" in status


class TestQwenCLICoderIntegration:
    """Test Qwen CLI Coder integration"""

    @patch('subprocess.run')
    def test_available_when_command_exists(self, mock_run):
        """Test availability when command exists"""
        mock_run.return_value = Mock(returncode=0)
        integration = QwenCLICoderIntegration()
        assert integration.is_available()

    @patch('subprocess.run')
    def test_not_available_when_command_missing(self, mock_run):
        """Test unavailability when command is missing"""
        mock_run.side_effect = FileNotFoundError()
        integration = QwenCLICoderIntegration()
        assert not integration.is_available()

    @patch('asyncio.create_subprocess_exec')
    async def test_successful_cli_execution(self, mock_subprocess):
        """Test successful CLI execution"""
        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Test output", b"")
        mock_subprocess.return_value = mock_process

        integration = QwenCLICoderIntegration()

        result = await integration.execute_task("Test task")

        assert isinstance(result, CLIResult)
        assert result.success
        assert result.output == "Test output"
        assert result.exit_code == 0

    def test_get_status(self):
        """Test get_status method"""
        integration = QwenCLICoderIntegration()
        status = integration.get_status()

        assert status["name"] == "qwen-cli-coder"
        assert status["command"] == "qwen-cli-coder"
        assert "available" in status
        assert "last_check" in status


class TestGeminiCLIIntegration:
    """Test Gemini CLI integration"""

    def test_initialization_without_api_key(self):
        """Test initialization without API key"""
        integration = GeminiCLIIntegration(api_key=None)
        assert not integration.is_available()

    @patch('ces.ai_orchestrator.cli_integration.genai.GenerativeModel')
    def test_successful_api_call(self, mock_model_class):
        """Test successful API call"""
        # Mock the Gemini model
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Test Gemini response"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        integration = GeminiCLIIntegration(api_key="test_key")

        async def run_test():
            result = await integration.execute_task("Test task")

            assert isinstance(result, APIResult)
            assert result.success
            assert result.response == "Test Gemini response"

        asyncio.run(run_test())


class TestAIAssistantManager:
    """Test AI Assistant Manager"""

    def test_initialization(self):
        """Test manager initialization"""
        manager = AIAssistantManager()
        assert len(manager.assistants) == 3
        assert "grok" in manager.assistants
        assert "qwen" in manager.assistants
        assert "gemini" in manager.assistants

    def test_get_available_assistants(self):
        """Test getting available assistants"""
        manager = AIAssistantManager()
        # Note: Actual availability depends on API keys and CLI tools
        available = manager.get_available_assistants()
        assert isinstance(available, list)

    def test_health_check(self):
        """Test health check functionality"""
        manager = AIAssistantManager()
        health = manager.health_check()

        assert health["component"] == "AI Assistant Manager"
        assert "overall_status" in health
        assert "assistants" in health
        assert "timestamp" in health


class TestAIOrchestrator:
    """Test AI Orchestrator"""

    def test_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = AIOrchestrator()
        assert hasattr(orchestrator, 'cli_manager')
        assert hasattr(orchestrator, 'assistants')

    def test_recommend_assistants(self):
        """Test assistant recommendation"""
        orchestrator = AIOrchestrator()
        recommendations = orchestrator.recommend_assistants(
            "Write a Python function",
            ["coding", "programming"]
        )
        assert isinstance(recommendations, list)

    def test_get_available_assistants(self):
        """Test getting available assistants"""
        orchestrator = AIOrchestrator()
        available = orchestrator.get_available_assistants()
        assert isinstance(available, list)
        # Each assistant should have name, display_name, capabilities, strengths
        for assistant in available:
            assert "name" in assistant
            assert "display_name" in assistant
            assert "capabilities" in assistant
            assert "strengths" in assistant

    def test_get_status(self):
        """Test get_status method"""
        orchestrator = AIOrchestrator()
        status = orchestrator.get_status()

        assert status["status"] == "operational"
        assert "total_assistants" in status
        assert "available_assistants" in status
        assert "assistants" in status
        assert "cli_integration_status" in status

    def test_health_check(self):
        """Test health check"""
        orchestrator = AIOrchestrator()
        health = orchestrator.health_check()

        assert health["component"] == "AI Orchestrator"
        assert "overall_status" in health
        assert "checks" in health


class TestTaskDelegator:
    """Test Task Delegator"""

    def test_initialization(self):
        """Test delegator initialization"""
        delegator = TaskDelegator()
        assert hasattr(delegator, 'ai_orchestrator')
        assert hasattr(delegator, 'delegation_rules')
        assert hasattr(delegator, 'delegation_history')

    def test_determine_task_type(self):
        """Test task type determination"""
        delegator = TaskDelegator()

        # Test coding task
        assert delegator._determine_task_type(["programming", "coding"]) == "coding"

        # Test analysis task
        assert delegator._determine_task_type(["analysis", "testing"]) == "analysis"

        # Test documentation task
        assert delegator._determine_task_type(["documentation"]) == "documentation"

        # Test complex task
        assert delegator._determine_task_type(["programming", "analysis", "testing"]) == "complex"

        # Test simple task
        assert delegator._determine_task_type(["general"]) == "simple"

    def test_get_delegation_stats(self):
        """Test delegation statistics"""
        delegator = TaskDelegator()
        stats = delegator.get_delegation_stats()

        assert "total_delegations" in stats
        assert "successful_delegations" in stats
        assert "success_rate" in stats
        assert "rules_defined" in stats
        assert "task_types_supported" in stats
        assert "assistant_usage" in stats
        assert "task_type_usage" in stats

    def test_get_status(self):
        """Test get_status method"""
        delegator = TaskDelegator()
        status = delegator.get_status()

        assert status["status"] == "operational"
        assert "delegation_rules" in status
        assert "supported_task_types" in status
        assert "ai_orchestrator_status" in status

    def test_health_check(self):
        """Test health check"""
        delegator = TaskDelegator()
        health = delegator.health_check()

        assert health["component"] == "Task Delegator"
        assert "overall_status" in health
        assert "checks" in health


class TestMCPIntegration:
    """Test MCP Protocol Integration"""

    @pytest.mark.asyncio
    async def test_full_integration_flow(self):
        """Test full integration flow from task to execution"""
        # This is a high-level integration test
        delegator = TaskDelegator()

        task_analysis = {
            "description": "Write a simple Python function to calculate factorial",
            "complexity_score": 3,
            "required_skills": ["programming", "coding"]
        }

        # Note: This test may fail if no real AI assistants are configured
        # In a real environment, this would test the full flow
        result = await delegator.delegate_task(task_analysis)

        assert "decision" in result
        assert "assistant" in result
        assert "timestamp" in result

        # The execution_result may be None if no assistants are available
        if result.get("execution_result"):
            assert "status" in result["execution_result"]


# Integration test fixtures
@pytest.fixture
def mock_grok_integration():
    """Mock Grok integration for testing"""
    integration = GrokCLIIntegration(api_key="test_key")
    integration.client = Mock()
    return integration


@pytest.fixture
def mock_ai_orchestrator():
    """Mock AI orchestrator for testing"""
    orchestrator = AIOrchestrator()
    # Mock the CLI manager to avoid real API calls
    orchestrator.cli_manager = Mock()
    return orchestrator


@pytest.fixture
def mock_task_delegator():
    """Mock task delegator for testing"""
    delegator = TaskDelegator()
    # Mock the AI orchestrator to avoid real API calls
    delegator.ai_orchestrator = Mock()
    return delegator


if __name__ == "__main__":
    pytest.main([__file__])
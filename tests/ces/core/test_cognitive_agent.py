"""
Tests for CES Cognitive Agent
"""

import pytest
from unittest.mock import Mock, patch
from ces.core.cognitive_agent import CognitiveAgent, TaskAnalysis
from ces.config.ces_config import CESConfig


class TestCognitiveAgent:
    """Test cases for CognitiveAgent"""

    @pytest.fixture
    def config(self):
        """Test configuration"""
        return CESConfig()

    @pytest.fixture
    def agent(self, config):
        """Test cognitive agent instance"""
        with patch('ces.core.memory_manager.MemoryManager'), \
             patch('ces.ai_orchestrator.ai_assistant.AIOrchestrator'), \
             patch('ces.core.ethical_controller.EthicalController'):
            return CognitiveAgent(config)

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent is not None
        assert hasattr(agent, 'memory_manager')
        assert hasattr(agent, 'ai_orchestrator')
        assert hasattr(agent, 'ethical_controller')

    def test_task_analysis(self, agent):
        """Test task analysis functionality"""
        task = "Implement user authentication"

        with patch.object(agent, '_calculate_complexity', return_value=5.0), \
             patch.object(agent, '_identify_required_skills', return_value=['programming']), \
             patch.object(agent, '_estimate_duration', return_value=60), \
             patch.object(agent.ai_orchestrator, 'recommend_assistants', return_value=['qwen']), \
             patch.object(agent.ethical_controller, 'check_task_ethics', return_value=[]), \
             patch.object(agent.memory_manager, 'analyze_context_needs', return_value=['task_history']):

            analysis = agent.analyze_task(task)

            assert isinstance(analysis, TaskAnalysis)
            assert analysis.complexity_score == 5.0
            assert 'programming' in analysis.required_skills
            assert analysis.estimated_duration == 60
            assert 'qwen' in analysis.recommended_assistants

    def test_complexity_calculation(self, agent):
        """Test complexity score calculation"""
        simple_task = "Add a button"
        complex_task = "Implement complex authentication with OAuth, JWT, and database integration"

        simple_score = agent._calculate_complexity(simple_task)
        complex_score = agent._calculate_complexity(complex_task)

        assert simple_score < complex_score
        assert 0 <= simple_score <= 10
        assert 0 <= complex_score <= 10

    def test_skill_identification(self, agent):
        """Test required skills identification"""
        coding_task = "Write a Python function to sort a list"
        design_task = "Design the architecture for a web application"
        test_task = "Write unit tests for the authentication module"

        coding_skills = agent._identify_required_skills(coding_task)
        design_skills = agent._identify_required_skills(design_task)
        test_skills = agent._identify_required_skills(test_task)

        assert 'programming' in coding_skills
        assert 'system_design' in design_skills
        assert 'testing' in test_skills

    def test_duration_estimation(self, agent):
        """Test task duration estimation"""
        duration = agent._estimate_duration(5.0, ['programming', 'testing'])

        assert duration > 0
        assert isinstance(duration, int)

    @patch('ces.core.cognitive_agent.datetime')
    def test_task_execution_success(self, mock_datetime, agent):
        """Test successful task execution"""
        mock_datetime.now.return_value.isoformat.return_value = "2025-01-01T12:00:00"

        task = "Implement login feature"

        with patch.object(agent, 'analyze_task') as mock_analyze, \
             patch.object(agent.ethical_controller, 'approve_task', return_value=True), \
             patch.object(agent.memory_manager, 'retrieve_context', return_value={}), \
             patch.object(agent.ai_orchestrator, 'execute_task') as mock_execute, \
             patch.object(agent.memory_manager, 'store_task_result') as mock_store:

            mock_analyze.return_value = TaskAnalysis(
                complexity_score=5.0,
                required_skills=['programming'],
                estimated_duration=60,
                recommended_assistants=['qwen'],
                ethical_concerns=[],
                context_requirements=[]
            )

            mock_execute.return_value = {"result": "Login feature implemented"}

            result = agent.execute_task(task)

            assert result['status'] == 'completed'
            assert 'result' in result
            assert result['assistant_used'] == 'qwen'
            mock_store.assert_called_once()

    def test_task_execution_ethical_rejection(self, agent):
        """Test task rejection due to ethical concerns"""
        task = "Create harmful software"

        with patch.object(agent, 'analyze_task') as mock_analyze, \
             patch.object(agent.ethical_controller, 'check_task_ethics', return_value=['harmful']), \
             patch.object(agent.ethical_controller, 'approve_task', return_value=False):

            mock_analyze.return_value = TaskAnalysis(
                complexity_score=5.0,
                required_skills=['programming'],
                estimated_duration=60,
                recommended_assistants=['qwen'],
                ethical_concerns=['harmful'],
                context_requirements=[]
            )

            result = agent.execute_task(task)

            assert result['status'] == 'rejected'
            assert 'ethical concerns' in result['reason'].lower()

    def test_get_status(self, agent):
        """Test status retrieval"""
        with patch.object(agent.memory_manager, 'get_status', return_value={'status': 'operational'}), \
             patch.object(agent.ai_orchestrator, 'get_status', return_value={'status': 'operational'}), \
             patch.object(agent.ethical_controller, 'get_status', return_value={'status': 'operational'}):

            status = agent.get_status()

            assert 'status' in status
            assert 'components' in status
            assert 'timestamp' in status
            assert status['components']['memory_manager']['status'] == 'operational'
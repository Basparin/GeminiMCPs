"""
Tests for CES Memory Manager
"""

import pytest
import tempfile
import os
from pathlib import Path
from ces.core.memory_manager import MemoryManager


class TestMemoryManager:
    """Test cases for MemoryManager"""

    @pytest.fixture
    def temp_db(self):
        """Temporary database file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def memory_manager(self, temp_db):
        """Test memory manager instance"""
        return MemoryManager(db_path=temp_db)

    def test_memory_manager_initialization(self, memory_manager):
        """Test memory manager initializes correctly"""
        assert memory_manager is not None
        assert memory_manager.db_path.exists()

    def test_store_and_retrieve_task_result(self, memory_manager):
        """Test storing and retrieving task results"""
        task_desc = "Test task"
        result = {
            "status": "completed",
            "result": "Task completed successfully",
            "analysis": {"complexity_score": 5.0}
        }

        # Store task result
        memory_manager.store_task_result(task_desc, result)

        # Retrieve recent tasks
        recent_tasks = memory_manager._get_recent_tasks(limit=5)

        assert len(recent_tasks) == 1
        assert recent_tasks[0]['description'] == task_desc
        assert recent_tasks[0]['result']['status'] == 'completed'

    def test_context_analysis(self, memory_manager):
        """Test context requirements analysis"""
        task_with_history = "Similar to previous task"
        task_with_preferences = "I usually prefer this approach"
        simple_task = "Implement a simple feature"

        history_needs = memory_manager.analyze_context_needs(task_with_history)
        pref_needs = memory_manager.analyze_context_needs(task_with_preferences)
        simple_needs = memory_manager.analyze_context_needs(simple_task)

        assert 'similar_tasks' in history_needs
        assert 'user_preferences' in pref_needs
        assert 'task_history' in simple_needs  # Always included

    def test_user_preferences(self, memory_manager):
        """Test user preferences storage and retrieval"""
        # Store preference
        memory_manager.store_user_preference('preferred_assistant', 'qwen')
        memory_manager.store_user_preference('complexity_threshold', 7)

        # Retrieve preferences
        prefs = memory_manager._get_user_preferences()

        assert prefs['preferred_assistant'] == 'qwen'
        assert prefs['complexity_threshold'] == 7

    def test_similar_tasks_finding(self, memory_manager):
        """Test finding similar tasks"""
        # Store some tasks
        tasks = [
            "Implement user authentication system",
            "Create login functionality",
            "Build user registration feature",
            "Optimize database queries"
        ]

        for task in tasks:
            memory_manager.store_task_result(task, {"status": "completed"})

        # Find similar tasks
        similar = memory_manager._find_similar_tasks("Implement user login system")

        assert len(similar) > 0
        # Should find the authentication and login tasks as similar
        similar_descriptions = [s['description'] for s in similar]
        assert any('authentication' in desc.lower() for desc in similar_descriptions)
        assert any('login' in desc.lower() for desc in similar_descriptions)

    def test_context_retrieval(self, memory_manager):
        """Test context retrieval for tasks"""
        # Store some task history
        memory_manager.store_task_result("Previous task", {"status": "completed"})

        # Store user preferences
        memory_manager.store_user_preference('preferred_language', 'python')

        # Retrieve context
        context = memory_manager.retrieve_context(
            "New task requiring context",
            ['task_history', 'user_preferences']
        )

        assert 'task_history' in context
        assert 'user_preferences' in context
        assert len(context['task_history']) == 1
        assert context['user_preferences']['preferred_language'] == 'python'

    def test_database_cleanup(self, memory_manager):
        """Test database cleanup functionality"""
        # This is a basic test - in real implementation would test actual cleanup
        initial_status = memory_manager.get_status()

        # Cleanup (with very old date to ensure cleanup)
        memory_manager.cleanup_old_data(days_to_keep=0)

        final_status = memory_manager.get_status()

        # Status should still be valid
        assert 'status' in final_status
        assert 'database_path' in final_status

    def test_status_reporting(self, memory_manager):
        """Test status reporting"""
        status = memory_manager.get_status()

        assert 'status' in status
        assert 'database_path' in status
        assert 'task_history_count' in status
        assert 'preferences_count' in status

        # Should be operational
        assert status['status'] == 'operational'

        # Counts should be integers
        assert isinstance(status['task_history_count'], int)
        assert isinstance(status['preferences_count'], int)
"""
Memory Manager - CES Context and Knowledge Management

Handles storage, retrieval, and management of context data, task history,
and learned patterns for the Cognitive Enhancement System.
"""

import logging
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json


class MemoryManager:
    """
    Manages different types of memory for CES:
    - Working Memory: Current session context
    - Task History: Completed tasks and outcomes
    - User Preferences: Personalized settings and patterns
    - Semantic Memory: Vector-based knowledge storage
    """

    def __init__(self, db_path: str = "ces_memory.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
        self.logger.info("Memory Manager initialized")

    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Task history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_description TEXT NOT NULL,
                    result TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    complexity_score REAL,
                    execution_time INTEGER,
                    assistant_used TEXT
                )
            ''')

            # User preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Context storage table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS context_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME,
                    UNIQUE(context_type, key)
                )
            ''')

            conn.commit()

    def store_task_result(self, task_description: str, result: Dict[str, Any]):
        """Store the result of a completed task"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO task_history
                (task_description, result, complexity_score, execution_time, assistant_used)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                task_description,
                json.dumps(result),
                result.get('analysis', {}).get('complexity_score', 0),
                result.get('analysis', {}).get('estimated_duration', 0),
                result.get('assistant_used', 'unknown')
            ))
            conn.commit()

        self.logger.info(f"Stored task result for: {task_description[:50]}...")

    def retrieve_context(self, task_description: str, requirements: List[str]) -> Dict[str, Any]:
        """
        Retrieve relevant context for a task based on requirements

        Args:
            task_description: Current task description
            requirements: List of context requirements

        Returns:
            Dict containing relevant context data
        """
        context = {}

        # Retrieve recent task history
        if 'task_history' in requirements:
            context['task_history'] = self._get_recent_tasks(limit=5)

        # Retrieve user preferences
        if 'user_preferences' in requirements:
            context['user_preferences'] = self._get_user_preferences()

        # Retrieve similar past tasks
        if 'similar_tasks' in requirements:
            context['similar_tasks'] = self._find_similar_tasks(task_description, limit=3)

        return context

    def analyze_context_needs(self, task_description: str) -> List[str]:
        """Analyze what context is needed for a given task"""
        needs = ['task_history']  # Always include recent history

        task_lower = task_description.lower()

        if any(word in task_lower for word in ['similar', 'like', 'previous', 'before']):
            needs.append('similar_tasks')

        if any(word in task_lower for word in ['prefer', 'usually', 'always', 'never']):
            needs.append('user_preferences')

        return needs

    def _get_recent_tasks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent task history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT task_description, result, timestamp, complexity_score
                FROM task_history
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            tasks = []
            for row in cursor.fetchall():
                tasks.append({
                    'description': row[0],
                    'result': json.loads(row[1]) if row[1] else None,
                    'timestamp': row[2],
                    'complexity': row[3]
                })

            return tasks

    def _get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM user_preferences')

            preferences = {}
            for row in cursor.fetchall():
                try:
                    preferences[row[0]] = json.loads(row[1])
                except json.JSONDecodeError:
                    preferences[row[0]] = row[1]

            return preferences

    def _find_similar_tasks(self, task_description: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find tasks similar to the current one (basic keyword matching)"""
        # Placeholder for semantic similarity - would use embeddings in production
        keywords = set(task_description.lower().split())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT task_description, result FROM task_history')

            similar_tasks = []
            for row in cursor.fetchall():
                task_text = row[0].lower()
                task_keywords = set(task_text.split())
                similarity = len(keywords.intersection(task_keywords)) / len(keywords.union(task_keywords))

                if similarity > 0.3:  # Basic similarity threshold
                    similar_tasks.append({
                        'description': row[0],
                        'result': json.loads(row[1]) if row[1] else None,
                        'similarity': similarity
                    })

            # Sort by similarity and return top matches
            similar_tasks.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_tasks[:limit]

    def store_user_preference(self, key: str, value: Any):
        """Store a user preference"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, json.dumps(value)))
            conn.commit()

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to manage database size"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Clean old task history
            cursor.execute('DELETE FROM task_history WHERE timestamp < ?',
                         (cutoff_date.isoformat(),))

            # Clean expired context data
            cursor.execute('DELETE FROM context_data WHERE expires_at < ?',
                         (datetime.now().isoformat(),))

            conn.commit()

        self.logger.info(f"Cleaned up data older than {days_to_keep} days")

    def get_status(self) -> Dict[str, Any]:
        """Get memory manager status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM task_history')
                task_count = cursor.fetchone()[0]

                cursor.execute('SELECT COUNT(*) FROM user_preferences')
                pref_count = cursor.fetchone()[0]

            return {
                "status": "operational",
                "database_path": str(self.db_path),
                "task_history_count": task_count,
                "preferences_count": pref_count
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE = os.getenv('DATABASE_URL', 'tasks.db')

def init_db():
    """Initialize the database with tasks table"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'pending',
            priority TEXT DEFAULT 'medium',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def get_all_tasks(status=None, priority=None):
    """Get all tasks with optional filtering"""
    conn = get_db_connection()
    
    # Build query based on filters
    query = 'SELECT * FROM tasks'
    params = []
    
    if status or priority:
        query += ' WHERE '
        conditions = []
        
        if status:
            conditions.append('status = ?')
            params.append(status)
        
        if priority:
            conditions.append('priority = ?')
            params.append(priority)
        
        query += ' AND '.join(conditions)
    
    query += ' ORDER BY created_at DESC'
    
    tasks = conn.execute(query, params).fetchall()
    conn.close()
    
    return [dict(task) for task in tasks]

def get_task_by_id(task_id):
    """Get a specific task by ID"""
    conn = get_db_connection()
    task = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
    conn.close()
    
    return dict(task) if task else None

def create_task(title, description='', status='pending', priority='medium'):
    """Create a new task"""
    conn = get_db_connection()
    cursor = conn.execute(
        'INSERT INTO tasks (title, description, status, priority) VALUES (?, ?, ?, ?)',
        (title, description, status, priority)
    )
    task_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        'id': task_id,
        'title': title,
        'description': description,
        'status': status,
        'priority': priority,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }

def update_task(task_id, title=None, description=None, status=None, priority=None):
    """Update an existing task"""
    # Get current task data
    current_task = get_task_by_id(task_id)
    if not current_task:
        return None
    
    # Use current values if new values not provided
    updated_title = title if title is not None else current_task['title']
    updated_description = description if description is not None else current_task['description']
    updated_status = status if status is not None else current_task['status']
    updated_priority = priority if priority is not None else current_task['priority']
    
    # Update in database
    conn = get_db_connection()
    conn.execute(
        'UPDATE tasks SET title = ?, description = ?, status = ?, priority = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
        (updated_title, updated_description, updated_status, updated_priority, task_id)
    )
    conn.commit()
    conn.close()
    
    # Return updated task
    updated_task = get_task_by_id(task_id)
    return updated_task

def delete_task(task_id):
    """Delete a task"""
    conn = get_db_connection()
    result = conn.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
    conn.commit()
    conn.close()
    
    return result.rowcount > 0
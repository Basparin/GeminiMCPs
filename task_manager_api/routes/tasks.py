from flask import Blueprint, request, jsonify
from datetime import datetime
from models.task_model import (
    get_all_tasks, 
    get_task_by_id, 
    create_task, 
    update_task, 
    delete_task
)
from utils.validation import validate_task_data, validate_task_id

# Create blueprint
tasks_bp = Blueprint('tasks', __name__)

@tasks_bp.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200

@tasks_bp.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get all tasks with optional filtering"""
    try:
        # Get query parameters for filtering
        status = request.args.get('status')
        priority = request.args.get('priority')
        
        tasks = get_all_tasks(status, priority)
        return jsonify(tasks), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tasks_bp.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    """Get a specific task by ID"""
    try:
        # Validate task ID
        is_valid, error = validate_task_id(task_id)
        if not is_valid:
            return jsonify({'error': error}), 400
        
        task = get_task_by_id(task_id)
        if task is None:
            return jsonify({'error': 'Task not found'}), 404
        
        return jsonify(task), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tasks_bp.route('/api/tasks', methods=['POST'])
def create_new_task():
    """Create a new task"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        is_valid, error = validate_task_data(data, ['title'])
        if not is_valid:
            return jsonify({'error': error}), 400
        
        # Extract fields with defaults
        title = data['title']
        description = data.get('description', '')
        status = data.get('status', 'pending')
        priority = data.get('priority', 'medium')
        
        # Create task
        task = create_task(title, description, status, priority)
        return jsonify(task), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tasks_bp.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_existing_task(task_id):
    """Update an existing task"""
    try:
        # Validate task ID
        is_valid, error = validate_task_id(task_id)
        if not is_valid:
            return jsonify({'error': error}), 400
        
        # Check if task exists
        task = get_task_by_id(task_id)
        if task is None:
            return jsonify({'error': 'Task not found'}), 404
        
        # Get data from request
        data = request.get_json()
        
        # Validate provided data
        if data:
            is_valid, error = validate_task_data(data)
            if not is_valid:
                return jsonify({'error': error}), 400
        
        # Extract fields (only update provided fields)
        title = data.get('title')
        description = data.get('description')
        status = data.get('status')
        priority = data.get('priority')
        
        # Update task
        updated_task = update_task(task_id, title, description, status, priority)
        if updated_task is None:
            return jsonify({'error': 'Task not found'}), 404
        
        return jsonify(updated_task), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tasks_bp.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_existing_task(task_id):
    """Delete a task"""
    try:
        # Validate task ID
        is_valid, error = validate_task_id(task_id)
        if not is_valid:
            return jsonify({'error': error}), 400
        
        # Check if task exists
        task = get_task_by_id(task_id)
        if task is None:
            return jsonify({'error': 'Task not found'}), 404
        
        # Delete task
        success = delete_task(task_id)
        if success:
            return jsonify({'message': 'Task deleted successfully'}), 200
        else:
            return jsonify({'error': 'Failed to delete task'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
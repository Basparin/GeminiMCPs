import requests
import json

BASE_URL = 'http://localhost:5000'

def test_health_check():
    """Test health check endpoint"""
    response = requests.get(f'{BASE_URL}/api/health')
    print(f'Health Check: {response.status_code}')
    print(json.dumps(response.json(), indent=2))

def test_create_task():
    """Test creating a task"""
    task_data = {
        'title': 'Test Task',
        'description': 'This is a test task',
        'status': 'pending',
        'priority': 'medium'
    }
    
    response = requests.post(f'{BASE_URL}/api/tasks', json=task_data)
    print(f'Create Task: {response.status_code}')
    print(json.dumps(response.json(), indent=2))
    return response.json().get('id')

def test_get_tasks():
    """Test getting all tasks"""
    response = requests.get(f'{BASE_URL}/api/tasks')
    print(f'Get Tasks: {response.status_code}')
    print(json.dumps(response.json(), indent=2))

def test_get_task(task_id):
    """Test getting a specific task"""
    response = requests.get(f'{BASE_URL}/api/tasks/{task_id}')
    print(f'Get Task: {response.status_code}')
    print(json.dumps(response.json(), indent=2))

def test_update_task(task_id):
    """Test updating a task"""
    update_data = {
        'status': 'completed',
        'priority': 'high'
    }
    
    response = requests.put(f'{BASE_URL}/api/tasks/{task_id}', json=update_data)
    print(f'Update Task: {response.status_code}')
    print(json.dumps(response.json(), indent=2))

def test_delete_task(task_id):
    """Test deleting a task"""
    response = requests.delete(f'{BASE_URL}/api/tasks/{task_id}')
    print(f'Delete Task: {response.status_code}')
    print(json.dumps(response.json(), indent=2))

if __name__ == '__main__':
    print('Testing Task Manager API')
    print('=' * 30)
    
    # Test health check
    test_health_check()
    print()
    
    # Test creating a task
    task_id = test_create_task()
    print()
    
    # Test getting all tasks
    test_get_tasks()
    print()
    
    # Test getting specific task
    if task_id:
        test_get_task(task_id)
        print()
        
        # Test updating task
        test_update_task(task_id)
        print()
        
        # Test getting updated task
        test_get_task(task_id)
        print()
        
        # Test deleting task
        test_delete_task(task_id)
        print()
        
        # Verify task is deleted
        test_get_tasks()
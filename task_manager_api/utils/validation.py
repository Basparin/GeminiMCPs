def validate_task_data(data, required_fields=None):
    """Validate task data"""
    if required_fields is None:
        required_fields = []
    
    # Check if data is provided
    if not data:
        return False, 'No data provided'
    
    # Check required fields
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f'{field} is required'
    
    # Validate status if provided
    if 'status' in data:
        valid_statuses = ['pending', 'in_progress', 'completed']
        if data['status'] not in valid_statuses:
            return False, f'Status must be one of {valid_statuses}'
    
    # Validate priority if provided
    if 'priority' in data:
        valid_priorities = ['low', 'medium', 'high']
        if data['priority'] not in valid_priorities:
            return False, f'Priority must be one of {valid_priorities}'
    
    return True, None

def validate_task_id(task_id):
    """Validate task ID"""
    try:
        tid = int(task_id)
        return tid > 0, 'Invalid task ID' if tid <= 0 else None
    except ValueError:
        return False, 'Task ID must be a number'
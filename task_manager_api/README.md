# Task Manager API

A Flask-based REST API for managing tasks with CRUD operations, SQLite database integration, and proper error handling.

## Features

- Create, Read, Update, and Delete tasks
- Filter tasks by status or priority
- SQLite database storage
- Comprehensive error handling
- Health check endpoint
- Environment-based configuration
- Modular code structure

## Project Structure

```
task_manager_api/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
├── README.md          # This file
├── test_api.py        # API testing script
├── models/            # Database models
│   ├── __init__.py
│   └── task_model.py
├── routes/            # API routes
│   ├── __init__.py
│   └── tasks.py
└── utils/             # Utility functions
    ├── __init__.py
    └── validation.py
```

## Setup

1. Clone the repository (if applicable) or navigate to the project directory:
   ```
   cd task_manager_api
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

The API will be available at `http://localhost:5000`

## Environment Variables

Create a `.env` file with the following variables:
```
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
DATABASE_URL=tasks.db
```

## API Endpoints

### Health Check
- `GET /api/health` - Check if the API is running

### Tasks
- `GET /api/tasks` - Get all tasks (optional filtering by status or priority)
- `GET /api/tasks/<id>` - Get a specific task by ID
- `POST /api/tasks` - Create a new task
- `PUT /api/tasks/<id>` - Update an existing task
- `DELETE /api/tasks/<id>` - Delete a task

## Task Object

```json
{
  "id": 1,
  "title": "Task Title",
  "description": "Task Description",
  "status": "pending|in_progress|completed",
  "priority": "low|medium|high",
  "created_at": "2023-01-01T00:00:00",
  "updated_at": "2023-01-01T00:00:00"
}
```

## Validation Rules

- **title**: Required string
- **description**: Optional string
- **status**: Optional, must be one of: `pending`, `in_progress`, `completed` (default: `pending`)
- **priority**: Optional, must be one of: `low`, `medium`, `high` (default: `medium`)

## Examples

### Create a new task
```bash
curl -X POST http://localhost:5000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"title": "New Task", "description": "Task description", "status": "pending", "priority": "medium"}'
```

### Get all tasks
```bash
curl http://localhost:5000/api/tasks
```

### Get tasks with status filter
```bash
curl http://localhost:5000/api/tasks?status=completed
```

### Get tasks with priority filter
```bash
curl http://localhost:5000/api/tasks?priority=high
```

### Get a specific task
```bash
curl http://localhost:5000/api/tasks/1
```

### Update a task
```bash
curl -X PUT http://localhost:5000/api/tasks/1 \
  -H "Content-Type: application/json" \
  -d '{"status": "completed"}'
```

### Delete a task
```bash
curl -X DELETE http://localhost:5000/api/tasks/1
```

## Testing

Run the provided test script to verify the API functionality:
```
python test_api.py
```

Note: The test script requires the API to be running and the `requests` library to be installed.

## Error Handling

The API returns appropriate HTTP status codes and JSON error messages:
- `400`: Bad Request (invalid input)
- `404`: Not Found (resource not found)
- `500`: Internal Server Error (unexpected server error)
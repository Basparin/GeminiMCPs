# CES API Reference

## Core Classes

### CognitiveAgent

The main orchestration class for CES operations.

#### Constructor

```python
CognitiveAgent(config: Optional[CESConfig] = None)
```

**Parameters:**
- `config`: Optional CES configuration object

#### Methods

##### analyze_task(task_description: str) -> TaskAnalysis

Analyzes a task to determine complexity, requirements, and optimal approach.

**Parameters:**
- `task_description`: Natural language description of the task

**Returns:**
- `TaskAnalysis`: Detailed analysis including complexity score, required skills, estimated duration, recommended assistants, ethical concerns, and context requirements

##### execute_task(task_description: str) -> Dict[str, Any]

Executes a task using the optimal AI assistant configuration.

**Parameters:**
- `task_description`: Task to execute

**Returns:**
- Dictionary containing execution results with status, analysis, result, and timestamp

##### get_status() -> Dict[str, Any]

Gets the current status of the cognitive agent and all components.

**Returns:**
- Dictionary with overall status and component statuses

---

### MemoryManager

Handles storage, retrieval, and management of context data.

#### Constructor

```python
MemoryManager(db_path: str = "ces_memory.db")
```

**Parameters:**
- `db_path`: Path to SQLite database file

#### Methods

##### store_task_result(task_description: str, result: Dict[str, Any]) -> None

Stores the result of a completed task in memory.

**Parameters:**
- `task_description`: Description of the completed task
- `result`: Execution result and metadata

##### retrieve_context(task_description: str, requirements: List[str]) -> Dict[str, Any]

Retrieves relevant context for a task based on requirements.

**Parameters:**
- `task_description`: Current task description
- `requirements`: List of context requirements (e.g., ['task_history', 'user_preferences'])

**Returns:**
- Dictionary containing relevant context data

##### analyze_context_needs(task_description: str) -> List[str]

Analyzes what context is needed for a given task.

**Parameters:**
- `task_description`: Task description to analyze

**Returns:**
- List of context requirements

##### store_user_preference(key: str, value: Any) -> None

Stores a user preference for future use.

**Parameters:**
- `key`: Preference key
- `value`: Preference value

##### get_status() -> Dict[str, Any]

Gets memory manager status and statistics.

**Returns:**
- Dictionary with status, database info, and usage statistics

---

### EthicalController

Ensures all CES operations comply with ethical standards.

#### Constructor

```python
EthicalController()
```

#### Methods

##### check_task_ethics(task_description: str) -> List[str]

Checks if a task raises ethical concerns.

**Parameters:**
- `task_description`: Description of the task to evaluate

**Returns:**
- List of identified ethical concerns

##### approve_task(concerns: List[str]) -> bool

Determines if a task can proceed despite identified concerns.

**Parameters:**
- `concerns`: List of ethical concerns

**Returns:**
- True if task can proceed, False otherwise

##### validate_output(task_description: str, output: str) -> Dict[str, Any]

Validates the output of a task for ethical compliance.

**Parameters:**
- `task_description`: Original task description
- `output`: Generated output to validate

**Returns:**
- Validation result with approval status and issues

##### get_status() -> Dict[str, Any]

Gets ethical controller status.

**Returns:**
- Dictionary with operational status and configuration info

---

### AIOrchestrator

Manages AI assistant integration and task delegation.

#### Constructor

```python
AIOrchestrator()
```

#### Methods

##### recommend_assistants(task_description: str, required_skills: List[str]) -> List[str]

Recommends AI assistants for a task based on description and skills.

**Parameters:**
- `task_description`: Description of the task
- `required_skills`: List of required skills

**Returns:**
- List of recommended assistant names

##### execute_task(task_description: str, context: Optional[Dict[str, Any]] = None, assistant_preferences: Optional[List[str]] = None) -> Dict[str, Any]

Executes a task using the most appropriate AI assistant.

**Parameters:**
- `task_description`: Task to execute
- `context`: Additional context for the task
- `assistant_preferences`: Preferred assistants to use

**Returns:**
- Task execution result

##### get_available_assistants() -> List[Dict[str, Any]]

Gets list of available AI assistants.

**Returns:**
- List of assistant information dictionaries

##### test_assistant_connection(assistant_name: str) -> Dict[str, Any]

Tests connection to a specific AI assistant.

**Parameters:**
- `assistant_name`: Name of the assistant to test

**Returns:**
- Test result with status and details

---

### CESConfig

Configuration management for CES.

#### Constructor

```python
CESConfig()
```

#### Key Attributes

- `debug_mode: bool` - Enable debug mode
- `log_level: str` - Logging level
- `max_memory_mb: int` - Maximum memory usage
- `memory_db_path: str` - Memory database path
- `ethical_checks_enabled: bool` - Enable ethical checks
- `cache_enabled: bool` - Enable caching

#### Methods

##### save_to_file(config_path: Optional[Path] = None) -> None

Saves current configuration to file.

**Parameters:**
- `config_path`: Optional path to save configuration

##### get_ai_assistant_configs() -> Dict[str, Dict[str, Any]]

Gets configuration for AI assistants.

**Returns:**
- Dictionary with AI assistant configurations

##### get_memory_config() -> Dict[str, Any]

Gets memory-related configuration.

**Returns:**
- Dictionary with memory configuration

---

### CodeSageIntegration

Integration layer between CES and CodeSage MCP server.

#### Constructor

```python
CodeSageIntegration(server_url: str = "http://localhost:8000", timeout: int = 30)
```

**Parameters:**
- `server_url`: URL of CodeSage MCP server
- `timeout`: Request timeout in seconds

#### Methods

##### connect() -> bool

Establishes connection to CodeSage MCP server.

**Returns:**
- True if connection successful, False otherwise

##### disconnect() -> None

Closes connection to CodeSage MCP server.

##### execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]

Executes a tool on the CodeSage server.

**Parameters:**
- `tool_name`: Name of the tool to execute
- `arguments`: Tool arguments

**Returns:**
- Tool execution result

##### get_available_tools() -> Dict[str, Dict[str, Any]]

Gets list of available tools from CodeSage.

**Returns:**
- Dictionary of available tools

##### get_server_status() -> Dict[str, Any]

Gets CodeSage server status.

**Returns:**
- Server status information

##### health_check() -> Dict[str, Any]

Performs health check on CodeSage integration.

**Returns:**
- Health check results

---

## Data Classes

### TaskAnalysis

Analysis result for a task.

**Attributes:**
- `complexity_score: float` - Task complexity (0-10)
- `required_skills: List[str]` - Required skills for the task
- `estimated_duration: int` - Estimated duration in minutes
- `recommended_assistants: List[str]` - Recommended AI assistants
- `ethical_concerns: List[str]` - Identified ethical concerns
- `context_requirements: List[str]` - Required context types

---

## CLI Commands

### ces task

Execute a task with CES.

```bash
ces task "Implement user authentication system" [--assistant ASSISTANT] [--verbose]
```

**Options:**
- `--assistant`: Preferred AI assistant
- `--verbose, -v`: Verbose output

### ces status

Show CES system status.

```bash
ces status [--detailed]
```

**Options:**
- `--detailed, -d`: Show detailed component status

### ces config

Configuration management.

```bash
ces config show                    # Show current configuration
ces config set KEY VALUE          # Set configuration value
```

### ces memory

Memory operations.

```bash
ces memory stats                  # Show memory statistics
ces memory clear                  # Clear memory data (not implemented)
```

---

## Error Handling

CES uses consistent error handling patterns:

### Error Response Format

```python
{
    "status": "error",
    "error": "Error description",
    "details": {},  # Optional additional details
    "timestamp": "ISO timestamp"
}
```

### Common Error Types

- `INVALID_PARAMS`: Invalid parameters provided
- `TOOL_NOT_FOUND`: Requested tool not available
- `CONNECTION_ERROR`: Connection issues with external services
- `ETHICAL_VIOLATION`: Task violates ethical guidelines
- `RESOURCE_LIMIT`: Resource limits exceeded

---

## Configuration Files

### Environment Variables

CES can be configured using environment variables prefixed with `CES_`:

```bash
# Core settings
CES_DEBUG=false
CES_LOG_LEVEL=INFO
CES_MAX_MEMORY_MB=256

# AI assistants
GROQ_API_KEY=your_grok_key
GEMINI_API_KEY=your_gemini_key
QWEN_API_KEY=your_qwen_key

# Memory settings
CES_MEMORY_DB_PATH=ces_memory.db
CES_MAX_CONTEXT_AGE_DAYS=90

# Ethical settings
CES_ETHICAL_CHECKS_ENABLED=true
CES_BIAS_DETECTION_ENABLED=true
```

### Configuration File

CES also supports JSON configuration file at `~/.ces/config.json`:

```json
{
    "debug_mode": false,
    "log_level": "INFO",
    "max_memory_mb": 256,
    "ethical_checks_enabled": true,
    "cache_enabled": true
}
```

---

## Examples

### Basic Task Execution

```python
from ces import CognitiveAgent

agent = CognitiveAgent()

# Analyze a task
analysis = agent.analyze_task("Create a REST API for user management")
print(f"Complexity: {analysis.complexity_score}")
print(f"Skills needed: {', '.join(analysis.required_skills)}")

# Execute the task
result = agent.execute_task("Create a REST API for user management")
print(f"Status: {result['status']}")
if result['status'] == 'completed':
    print(f"Result: {result['result']}")
```

### CodeSage Integration

```python
import asyncio
from ces import CodeSageIntegration

async def main():
    integration = CodeSageIntegration("http://localhost:8000")

    # Connect to CodeSage
    connected = await integration.connect()
    if not connected:
        print("Failed to connect to CodeSage")
        return

    # Execute a tool
    result = await integration.execute_tool(
        "read_code_file",
        {"file_path": "example.py"}
    )

    print(f"Tool result: {result}")

asyncio.run(main())
```

### Custom Configuration

```python
from ces.config import CESConfig

# Load custom configuration
config = CESConfig()
config.debug_mode = True
config.max_memory_mb = 512

# Save configuration
config.save_to_file()

# Use with agent
from ces import CognitiveAgent
agent = CognitiveAgent(config)
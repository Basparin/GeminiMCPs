// CES Dashboard JavaScript

// Global variables
let websocket = null;
let currentUserId = 'user_' + Math.random().toString(36).substr(2, 9);
let charts = {};
let reconnectAttempts = 0;
let maxReconnectAttempts = 5;

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupWebSocket();
    setupEventListeners();
    loadInitialData();
});

// Initialize dashboard components
function initializeDashboard() {
    console.log('Initializing CES Dashboard...');

    // Initialize charts
    initializeCharts();

    // Set up periodic updates
    setInterval(updateSystemStatus, 30000); // Update every 30 seconds
    setInterval(updateActiveUsers, 10000); // Update every 10 seconds
}

// Setup WebSocket connection
function setupWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${currentUserId}`;

    websocket = new WebSocket(wsUrl);

    websocket.onopen = function(event) {
        console.log('WebSocket connected');
        reconnectAttempts = 0;
        showNotification('Connected to CES server', 'success');
        updateConnectionStatus(true);
    };

    websocket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };

    websocket.onclose = function(event) {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
        attemptReconnect();
    };

    websocket.onerror = function(error) {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
    console.log('Received WebSocket message:', data);

    switch (data.type) {
        case 'task_created':
            handleTaskCreated(data.data);
            break;
        case 'task_updated':
            handleTaskUpdated(data.data);
            break;
        case 'session_created':
            handleSessionCreated(data.data);
            break;
        case 'session_updated':
            handleSessionUpdated(data.data);
            break;
        case 'user_joined':
            handleUserJoined(data.data);
            break;
        case 'user_disconnected':
            handleUserDisconnected(data.data);
            break;
        case 'system_status':
            updateSystemStatusDisplay(data.data);
            break;
        case 'ai_response':
            handleAIResponse(data.data);
            break;
        default:
            console.log('Unknown message type:', data.type);
    }
}

// Setup event listeners
function setupEventListeners() {
    // AI query input
    const aiQueryInput = document.getElementById('ai-query-input');
    if (aiQueryInput) {
        aiQueryInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendAIQuery();
            }
        });
    }

    // Window focus/blur for activity tracking
    window.addEventListener('focus', function() {
        document.title = 'CES Dashboard';
    });

    window.addEventListener('blur', function() {
        document.title = '(Inactive) CES Dashboard';
    });
}

// Load initial data
async function loadInitialData() {
    try {
        await Promise.all([
            updateSystemStatus(),
            updateActiveUsers(),
            loadRecentActivity(),
            loadActiveSessions()
        ]);
    } catch (error) {
        console.error('Error loading initial data:', error);
        showNotification('Error loading dashboard data', 'error');
    }
}

// Update system status
async function updateSystemStatus() {
    try {
        const response = await fetch('/api/system/status');
        const data = await response.json();

        if (response.ok) {
            updateSystemStatusDisplay(data);
        } else {
            console.error('Error fetching system status:', data);
        }
    } catch (error) {
        console.error('Error updating system status:', error);
    }
}

// Update system status display
function updateSystemStatusDisplay(data) {
    // Update health indicators
    updateHealthIndicator('ces-health', data.performance ? 95 : 50);
    updateHealthIndicator('ai-health', data.ai_assistants ? 90 : 30);
    updateHealthIndicator('memory-health', 85); // Placeholder
    updateHealthIndicator('response-time', data.performance?.response_time_ms ? Math.min(data.performance.response_time_ms / 10, 100) : 80);

    // Update response time text
    const responseTimeText = document.getElementById('response-time-text');
    if (responseTimeText && data.performance?.response_time_ms) {
        responseTimeText.textContent = `${data.performance.response_time_ms}ms`;
    }

    // Update summary cards
    document.getElementById('total-tasks').textContent = data.analytics?.total_tasks || 0;
    document.getElementById('completed-tasks').textContent = Math.floor((data.analytics?.total_tasks || 0) * 0.7); // Placeholder
    document.getElementById('active-sessions-count').textContent = data.active_sessions || 0;
    document.getElementById('avg-response-time').textContent = `${data.analytics?.avg_response_time_ms || 0}ms`;

    // Update charts with new data
    updateCharts(data);
}

// Update health indicator
function updateHealthIndicator(elementId, percentage) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.width = `${percentage}%`;

        // Update color based on percentage
        element.className = 'progress-bar';
        if (percentage >= 90) {
            element.classList.add('bg-success');
        } else if (percentage >= 70) {
            element.classList.add('bg-warning');
        } else {
            element.classList.add('bg-danger');
        }
    }
}

// Update active users count
async function updateActiveUsers() {
    try {
        const response = await fetch('/api/users/active');
        const users = await response.json();

        const activeUsersElement = document.getElementById('active-users');
        if (activeUsersElement) {
            activeUsersElement.textContent = users.length;
        }
    } catch (error) {
        console.error('Error updating active users:', error);
    }
}

// Load recent activity
async function loadRecentActivity() {
    try {
        const response = await fetch('/api/analytics/overview');
        const data = await response.json();

        const activityList = document.getElementById('recent-activity-list');
        if (activityList && data.user_engagement?.activity_distribution) {
            activityList.innerHTML = '';

            // Create activity items from analytics data
            Object.entries(data.user_engagement.activity_distribution).forEach(([type, count]) => {
                const activityItem = createActivityItem(type, count);
                activityList.appendChild(activityItem);
            });
        }
    } catch (error) {
        console.error('Error loading recent activity:', error);
        document.getElementById('recent-activity-list').innerHTML =
            '<p class="text-muted">Unable to load recent activity</p>';
    }
}

// Create activity item
function createActivityItem(type, count) {
    const div = document.createElement('div');
    div.className = 'activity-item d-flex align-items-center';

    const iconDiv = document.createElement('div');
    iconDiv.className = `activity-icon ${getActivityIconClass(type)}`;
    iconDiv.innerHTML = getActivityIcon(type);

    const contentDiv = document.createElement('div');
    contentDiv.className = 'ms-3 flex-grow-1';
    contentDiv.innerHTML = `
        <div class="fw-bold">${type.replace('_', ' ').toUpperCase()}</div>
        <small class="text-muted">${count} events</small>
    `;

    div.appendChild(iconDiv);
    div.appendChild(contentDiv);

    return div;
}

// Get activity icon class
function getActivityIconClass(type) {
    const iconMap = {
        'task_created': 'task',
        'ai_interaction': 'ai',
        'session_created': 'session',
        'default': 'task'
    };
    return iconMap[type] || iconMap.default;
}

// Get activity icon
function getActivityIcon(type) {
    const iconMap = {
        'task_created': '<i class="fas fa-tasks"></i>',
        'ai_interaction': '<i class="fas fa-robot"></i>',
        'session_created': '<i class="fas fa-users"></i>',
        'default': '<i class="fas fa-info-circle"></i>'
    };
    return iconMap[type] || iconMap.default;
}

// Load active sessions
async function loadActiveSessions() {
    try {
        const response = await fetch('/api/sessions');
        const sessions = await response.json();

        const sessionsList = document.getElementById('active-sessions-list');
        if (sessionsList) {
            sessionsList.innerHTML = '';

            if (sessions.length === 0) {
                sessionsList.innerHTML = '<p class="text-muted">No active sessions</p>';
            } else {
                sessions.forEach(session => {
                    const sessionItem = createSessionItem(session);
                    sessionsList.appendChild(sessionItem);
                });
            }
        }
    } catch (error) {
        console.error('Error loading active sessions:', error);
    }
}

// Create session item
function createSessionItem(session) {
    const div = document.createElement('div');
    div.className = 'session-item';
    div.innerHTML = `
        <div class="d-flex justify-content-between align-items-start">
            <div>
                <h6 class="mb-1">${session.name}</h6>
                <p class="mb-1 text-muted small">${session.description || 'No description'}</p>
                <small class="text-muted">
                    <i class="fas fa-users"></i> ${session.participant_count} participants
                    <i class="fas fa-tasks ms-2"></i> ${session.task_count} tasks
                </small>
            </div>
            <div class="text-end">
                <small class="text-muted d-block">${formatTimeAgo(session.last_activity)}</small>
                <span class="badge bg-success">Active</span>
            </div>
        </div>
    `;

    return div;
}

// Initialize charts
function initializeCharts() {
    // Response Time Chart
    const responseTimeCtx = document.getElementById('responseTimeChart');
    if (responseTimeCtx) {
        charts.responseTime = new Chart(responseTimeCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Response Time (ms)',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Task Completion Chart
    const taskCompletionCtx = document.getElementById('taskCompletionChart');
    if (taskCompletionCtx) {
        charts.taskCompletion = new Chart(taskCompletionCtx, {
            type: 'bar',
            data: {
                labels: ['Completed', 'Pending', 'In Progress'],
                datasets: [{
                    label: 'Tasks',
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.8)',
                        'rgba(255, 193, 7, 0.8)',
                        'rgba(23, 162, 184, 0.8)'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
}

// Update charts with new data
function updateCharts(data) {
    if (charts.responseTime && data.performance?.response_time_trend) {
        const trend = data.performance.response_time_trend;
        charts.responseTime.data.labels = trend.map(item => formatTime(item.timestamp));
        charts.responseTime.data.datasets[0].data = trend.map(item => item.value);
        charts.responseTime.update();
    }

    if (charts.taskCompletion && data.tasks?.completion_trend) {
        const trend = data.tasks.completion_trend;
        const latest = trend[trend.length - 1] || {};
        charts.taskCompletion.data.datasets[0].data = [
            latest.completions || 0,
            Math.floor((latest.total_tasks || 0) * 0.3), // Placeholder for pending
            Math.floor((latest.total_tasks || 0) * 0.2)  // Placeholder for in progress
        ];
        charts.taskCompletion.update();
    }
}

// Send AI query
async function sendAIQuery() {
    const queryInput = document.getElementById('ai-query-input');
    const assistantSelect = document.getElementById('ai-assistant-select');
    const responseArea = document.getElementById('ai-response-area');

    if (!queryInput || !queryInput.value.trim()) {
        showNotification('Please enter a query', 'warning');
        return;
    }

    const query = queryInput.value.trim();
    const assistant = assistantSelect.value;

    // Add user message to response area
    addMessageToResponseArea('user', query);

    // Clear input
    queryInput.value = '';

    // Show loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'ai-response assistant';
    loadingDiv.innerHTML = '<div class="loading-spinner"></div> Thinking...';
    responseArea.appendChild(loadingDiv);

    try {
        // Here you would integrate with the actual AI assistant
        // For now, we'll simulate a response
        // TODO
        setTimeout(() => {
            loadingDiv.remove();
            addMessageToResponseArea('assistant', `Response from ${assistant}: This is a simulated response to "${query}"`);
        }, 2000);

    } catch (error) {
        console.error('Error sending AI query:', error);
        loadingDiv.remove();
        addMessageToResponseArea('assistant', 'Error: Unable to process query');
        showNotification('Error sending query', 'error');
    }
}

// Add message to response area
function addMessageToResponseArea(type, content) {
    const responseArea = document.getElementById('ai-response-area');
    if (!responseArea) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `ai-response ${type} fade-in`;

    // Simple markdown-like formatting for code blocks
    if (content.includes('```')) {
        content = content.replace(/```(\w+)?\n?([\s\S]*?)```/g, '<div class="code-block">$2</div>');
    }

    messageDiv.innerHTML = content;
    responseArea.appendChild(messageDiv);

    // Scroll to bottom
    responseArea.scrollTop = responseArea.scrollHeight;
}

// Create new task
function createNewTask() {
    const modal = new bootstrap.Modal(document.getElementById('taskModal'));
    modal.show();
}

// Submit task
async function submitTask() {
    const description = document.getElementById('taskDescription').value;
    const priority = document.getElementById('taskPriority').value;
    const tags = document.getElementById('taskTags').value;

    if (!description.trim()) {
        showNotification('Please enter a task description', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/tasks', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                description: description,
                priority: priority,
                tags: tags.split(',').map(tag => tag.trim()).filter(tag => tag),
                user_id: currentUserId
            })
        });

        const result = await response.json();

        if (response.ok) {
            showNotification('Task created successfully', 'success');
            bootstrap.Modal.getInstance(document.getElementById('taskModal')).hide();

            // Clear form
            document.getElementById('taskDescription').value = '';
            document.getElementById('taskTags').value = '';

            // Refresh data
            loadRecentActivity();
        } else {
            showNotification('Error creating task', 'error');
        }
    } catch (error) {
        console.error('Error submitting task:', error);
        showNotification('Error creating task', 'error');
    }
}

// Create new session
function createNewSession() {
    const modal = new bootstrap.Modal(document.getElementById('sessionModal'));
    modal.show();
}

// Submit session
async function submitSession() {
    const name = document.getElementById('sessionName').value;
    const description = document.getElementById('sessionDescription').value;
    const collaborators = document.getElementById('sessionCollaborators').value;

    if (!name.trim()) {
        showNotification('Please enter a session name', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/sessions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                description: description,
                user_id: currentUserId,
                collaborators: collaborators.split(',').map(email => email.trim()).filter(email => email)
            })
        });

        const result = await response.json();

        if (response.ok) {
            showNotification('Session created successfully', 'success');
            bootstrap.Modal.getInstance(document.getElementById('sessionModal')).hide();

            // Clear form
            document.getElementById('sessionName').value = '';
            document.getElementById('sessionDescription').value = '';
            document.getElementById('sessionCollaborators').value = '';

            // Refresh data
            loadActiveSessions();
        } else {
            showNotification('Error creating session', 'error');
        }
    } catch (error) {
        console.error('Error submitting session:', error);
        showNotification('Error creating session', 'error');
    }
}

// Handle task created
function handleTaskCreated(task) {
    showNotification(`New task created: ${task.description}`, 'info');
    loadRecentActivity();
}

// Handle session created
function handleSessionCreated(session) {
    showNotification(`New session created: ${session.name}`, 'info');
    loadActiveSessions();
}

// Handle user joined
function handleUserJoined(data) {
    showNotification(`User joined session: ${data.user_id}`, 'info');
    updateActiveUsers();
}

// Handle user disconnected
function handleUserDisconnected(data) {
    showNotification(`User disconnected: ${data.user_id}`, 'info');
    updateActiveUsers();
}

// Handle AI response
function handleAIResponse(data) {
    addMessageToResponseArea('assistant', data.response);
}

// Update connection status
function updateConnectionStatus(connected) {
    const indicator = document.querySelector('.navbar-brand');
    if (indicator) {
        if (connected) {
            indicator.style.color = '#28a745';
        } else {
            indicator.style.color = '#dc3545';
        }
    }
}

// Attempt to reconnect WebSocket
function attemptReconnect() {
    if (reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++;
        console.log(`Attempting to reconnect WebSocket (${reconnectAttempts}/${maxReconnectAttempts})`);

        setTimeout(() => {
            setupWebSocket();
        }, 2000 * reconnectAttempts); // Exponential backoff
    } else {
        showNotification('Unable to reconnect to server', 'error');
    }
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.innerHTML = `
        <div class="toast ${type} show" role="alert">
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;

    // Add to page
    document.body.appendChild(notification);

    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Format time ago
function formatTimeAgo(timestamp) {
    const now = new Date();
    const time = new Date(timestamp);
    const diff = now - time;

    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'Just now';
}

// Format time for charts
function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Placeholder functions for future implementation
function showAnalytics() {
    showNotification('Analytics view coming soon', 'info');
}

function showSettings() {
    showNotification('Settings panel coming soon', 'info');
}

// Handle task updated (placeholder)
function handleTaskUpdated(task) {
    console.log('Task updated:', task);
}

// Handle session updated (placeholder)
function handleSessionUpdated(session) {
    console.log('Session updated:', session);
}
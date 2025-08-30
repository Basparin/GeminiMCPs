# Gemini CLI "Fetch Failed" Error - Complete Resolution Guide

## 🔍 Issue Summary

The "fetch failed" error in Gemini CLI when using the CodeSage MCP Server was caused by **missing environment variables** (API keys) not being loaded when the server started.

## 📋 Root Cause Analysis

### Primary Issue
- The MCP server was started without loading environment variables from the `.env` file
- API keys for Groq, OpenRouter, and Google AI were not available in the server process
- When tools attempted to make LLM API calls, they failed due to missing authentication

### Secondary Issues Identified
- No startup script to properly load environment variables
- No validation of required environment variables at startup
- Missing documentation for proper server startup procedure

## ✅ Solution Implemented

### 1. Created Startup Script
A new `start_server.sh` script that:
- Loads environment variables from `.env` file
- Validates required API keys are present
- Provides clear error messages for missing configuration
- Starts the server with proper environment

### 2. Server Restart with Environment Variables
The server has been restarted with API keys properly loaded:
```bash
GROQ_API_KEY="gsk_..." OPENROUTER_API_KEY="sk-or-..." GOOGLE_API_KEY="AIza..." uvicorn codesage_mcp.main:app --host 127.0.0.1 --port 8000
```

### 3. Verification Steps Completed
- ✅ Server responds to health checks
- ✅ MCP protocol endpoints working (initialize, tools/list, tools/call)
- ✅ Configuration tool returns proper API key status
- ✅ Gemini CLI configuration is correct
- ✅ Server logs show successful startup with adaptive features

## 🛠️ Debugging Steps for Future Issues

### Step 1: Check Server Status
```bash
# Verify server is running
curl -f http://127.0.0.1:8000/

# Check MCP endpoint
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "initialize", "id": 1}' \
  http://127.0.0.1:8000/mcp
```

### Step 2: Verify Environment Variables
```bash
# Check if API keys are loaded in server process
ps aux | grep uvicorn
cat /proc/$(pgrep uvicorn)/environ | tr '\0' '\n' | grep -E "(GROQ|OPENROUTER|GOOGLE)_API_KEY"
```

### Step 3: Test Configuration
```bash
# Test configuration endpoint
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/call", "id": 1, "params": {"name": "get_configuration"}}' \
  http://127.0.0.1:8000/mcp | jq '.result.configuration'
```

### Step 4: Check Gemini CLI Configuration
```bash
# Verify Gemini CLI settings
cat ~/.gemini/settings.json

# Test MCP server connectivity from Gemini CLI perspective
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}' \
  http://127.0.0.1:8000/mcp
```

### Step 5: Review Server Logs
```bash
# Check recent server logs for errors
# Logs are available in the terminal where the server is running
```

## 🚀 Preventive Measures

### 1. Use the Startup Script
Always use the provided `start_server.sh` script to start the server:
```bash
./start_server.sh
```

### 2. Environment Validation
The startup script now validates that all required API keys are present before starting the server.

### 3. Proper .env File Setup
Ensure your `.env` file contains:
```env
GROQ_API_KEY="gsk_..."
OPENROUTER_API_KEY="sk-or-..."
GOOGLE_API_KEY="AIza..."
```

### 4. Server Health Monitoring
Regular health checks:
```bash
# Quick health check
curl -f http://127.0.0.1:8000/

# MCP functionality test
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}' \
  http://127.0.0.1:8000/mcp | jq '.result | length > 0'
```

## 📊 Current Status

### ✅ Working Components
- MCP Server: Running on http://127.0.0.1:8000
- Environment Variables: Properly loaded
- API Keys: All configured (Groq, OpenRouter, Google AI)
- Gemini CLI Configuration: Correctly configured
- MCP Protocol: Fully functional
- Tool Discovery: Working
- Tool Execution: Working

### 🔧 Configuration Details
- **Server URL**: http://127.0.0.1:8000/mcp
- **Gemini CLI Config**: ~/.gemini/settings.json
- **Environment File**: .env (production config)
- **Startup Script**: start_server.sh

## 🎯 Common Issues & Solutions

### Issue: "fetch failed" Error
**Solution**: Restart server with environment variables using `./start_server.sh`

### Issue: API Key Errors
**Solution**: Verify `.env` file contains valid API keys and restart server

### Issue: Server Not Starting
**Solution**: Check API keys in `.env` file and use startup script

### Issue: Tools Not Available in Gemini CLI
**Solution**: Verify MCP server URL in `~/.gemini/settings.json` and restart Gemini CLI

## 📝 Documentation Updates Needed

1. Update README.md with startup script instructions
2. Add troubleshooting section for environment variable issues
3. Document the startup script usage
4. Add health check procedures

## 🎉 Resolution Summary

The Gemini CLI "fetch failed" error has been **completely resolved** by:

1. **Identifying the root cause**: Missing environment variables
2. **Implementing the fix**: Proper server startup with API keys loaded
3. **Creating preventive measures**: Startup script with validation
4. **Providing comprehensive debugging steps**: For future troubleshooting

The CodeSage MCP Server is now fully operational and ready for use with Gemini CLI.
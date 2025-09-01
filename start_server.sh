#!/bin/bash
# CodeSage MCP Server Startup Script
# This script properly loads environment variables and starts the server

set -e

echo "🚀 Starting CodeSage MCP Server..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your API keys. See .env.example for reference."
    exit 1
fi

# Load environment variables from .env file
echo "📋 Loading environment variables..."
export $(grep -v '^#' .env | xargs)

# Verify required API keys are set
if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ Error: GROQ_API_KEY is not set in .env file"
    exit 1
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ Error: OPENROUTER_API_KEY is not set in .env file"
    exit 1
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ Error: GOOGLE_API_KEY is not set in .env file"
    exit 1
fi

echo "✅ Environment variables loaded successfully"
echo "🔧 Starting server on http://127.0.0.1:8002"

# Start the server
uvicorn codesage_mcp.main:app --host 127.0.0.1 --port 8000
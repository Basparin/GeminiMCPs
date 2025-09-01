#!/bin/bash
# CES Web Interface Startup Script
# This script starts the CES web dashboard

set -e

echo "🚀 Starting CES Web Interface..."

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
echo "🌐 Starting web interface on http://127.0.0.1:8001"

# Start the web interface
python -m ces.web.app
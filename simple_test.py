#!/usr/bin/env python3
"""
Simple CES CLI Integration Test

Tests the basic CLI integrations without complex dependencies
"""

import asyncio
import subprocess
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def test_cli_integrations():
    """Test basic CLI integrations"""
    print("🧪 Testing CES CLI Integrations")
    print("=" * 40)

    # Test 1: Check if CLI tools are available
    print("\n🔍 TEST 1: CLI Tool Availability")
    print("-" * 30)

    cli_tools = {
        'grok': 'grok',
        'qwen-cli-coder': 'qwen-cli-coder',
        'gemini-cli': 'gemini-cli'
    }

    available_tools = {}
    for name, command in cli_tools.items():
        try:
            result = subprocess.run(
                [command, '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                available_tools[name] = command
                print(f"  ✅ {name}: Available")
            else:
                print(f"  ❌ {name}: Not available (exit code {result.returncode})")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"  ❌ {name}: Not available ({type(e).__name__})")

    # Test 2: Environment variables
    print("\n🔍 TEST 2: Environment Variables")
    print("-" * 30)

    env_vars = {
        'GROQ_API_KEY': 'Grok API',
        'GOOGLE_API_KEY': 'Gemini API',
        'OPENROUTER_API_KEY': 'OpenRouter API'
    }

    for var, description in env_vars.items():
        if os.getenv(var):
            print(f"  ✅ {description}: Set")
        else:
            print(f"  ❌ {description}: Not set")

    # Test 3: Basic CLI execution test
    print("\n🔍 TEST 3: Basic CLI Execution Test")
    print("-" * 30)

    if 'qwen-cli-coder' in available_tools:
        print("  🧪 Testing qwen-cli-coder with simple task...")
        try:
            # Simple test command
            result = subprocess.run(
                ['qwen-cli-coder', '--help'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print("  ✅ qwen-cli-coder help command successful")
                print(f"  📝 Output length: {len(result.stdout)} characters")
            else:
                print(f"  ❌ qwen-cli-coder help failed: {result.stderr[:100]}...")
        except Exception as e:
            print(f"  ❌ qwen-cli-coder test failed: {e}")
    else:
        print("  ⏭️  Skipping qwen-cli-coder test (not available)")

    # Test 4: API connectivity test (if keys are available)
    print("\n🔍 TEST 4: API Connectivity Test")
    print("-" * 30)

    if os.getenv('GROQ_API_KEY'):
        print("  🧪 Testing Groq API connectivity...")
        try:
            import groq
            client = groq.Groq(api_key=os.getenv('GROQ_API_KEY'))
            # Simple test
            response = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            print("  ✅ Groq API connection successful")
            print(f"  📝 Response: {response.choices[0].message.content[:50]}...")
        except Exception as e:
            print(f"  ❌ Groq API test failed: {e}")
    else:
        print("  ⏭️  Skipping Groq API test (no API key)")

    if os.getenv('GOOGLE_API_KEY'):
        print("  🧪 Testing Gemini API connectivity...")
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content("Hello")
            print("  ✅ Gemini API connection successful")
            print(f"  📝 Response: {response.text[:50]}...")
        except Exception as e:
            print(f"  ❌ Gemini API test failed: {e}")
    else:
        print("  ⏭️  Skipping Gemini API test (no API key)")

    # Summary
    print("\n📊 SUMMARY")
    print("-" * 30)
    print(f"Available CLI tools: {len(available_tools)}/{len(cli_tools)}")
    print(f"Environment variables set: {sum(1 for var in env_vars if os.getenv(var))}/{len(env_vars)}")

    if available_tools:
        print("✅ Basic CLI integration test completed")
        print("💡 CES can use the following assistants:")
        for name in available_tools:
            print(f"   - {name}")
    else:
        print("⚠️  No CLI tools available - check installation")

    print("\n🎯 Next Steps:")
    print("1. Ensure API keys are set in environment or .env file")
    print("2. Install missing CLI tools if needed")
    print("3. Test full CES orchestrator integration")

if __name__ == "__main__":
    asyncio.run(test_cli_integrations())
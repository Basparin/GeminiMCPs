#!/usr/bin/env python3
"""
CES AI Assistant Integration Test Script

Tests the integration between CES and AI assistants (Grok, qwen-cli-coder, gemini-cli)
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add CES modules to path
sys.path.append('ces')
sys.path.append('.')

async def test_ai_integrations():
    """Test AI assistant integrations"""
    print("🚀 Testing CES AI Assistant Integrations")
    print("=" * 50)

    try:
        # Import CES components with absolute imports
        import ces.ai_orchestrator.ai_assistant as ai_assistant
        import ces.ai_orchestrator.cli_integration as cli_integration

        print("✅ CES modules imported successfully")

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure CES modules are properly installed and configured")
        return
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    try:

        # Initialize components
        print("\n📋 Initializing AI Orchestrator...")
        orchestrator = ai_assistant.AIOrchestrator()

        print("📋 Initializing AI Assistant Manager...")
        ai_manager = cli_integration.AIAssistantManager()

        # Test 1: Check available assistants
        print("\n🧪 TEST 1: Available Assistants")
        print("-" * 30)

        available_assistants = ai_manager.get_available_assistants()
        print(f"Available assistants: {len(available_assistants)}")

        for assistant in available_assistants:
            print(f"  - {assistant['display_name']} ({assistant['name']})")
            print(f"    Capabilities: {', '.join(assistant['capabilities'])}")
            print(f"    Strengths: {', '.join(assistant['strengths'])}")

        # Test 2: Test individual assistant connections
        print("\n🧪 TEST 2: Individual Assistant Connection Tests")
        print("-" * 30)

        test_tasks = [
            ("grok", "Hello, can you confirm you're working? Please respond with a simple greeting."),
            ("qwen-cli-coder", "Write a simple Python function to add two numbers."),
            ("gemini-cli", "Analyze this code: def hello(): print('Hello World')")
        ]

        for assistant_name, test_task in test_tasks:
            print(f"\n🔍 Testing {assistant_name}...")
            try:
                result = await orchestrator.test_assistant_connection(assistant_name)
                if result['status'] == 'success':
                    print(f"  ✅ {assistant_name} connection successful")
                    print(f"  📝 Response preview: {result['response'][:100]}...")
                else:
                    print(f"  ❌ {assistant_name} connection failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"  ❌ {assistant_name} test failed: {e}")

        # Test 3: Test task delegation
        print("\n🧪 TEST 3: Task Delegation Test")
        print("-" * 30)

        test_tasks = [
            "Write a Python function to calculate fibonacci numbers",
            "Analyze the performance of a sorting algorithm",
            "Document a REST API endpoint"
        ]

        for task in test_tasks:
            print(f"\n📋 Task: {task}")
            try:
                # Get recommendations
                recommendations = orchestrator.recommend_assistants(task, [])
                print(f"  🤖 Recommended assistants: {', '.join(recommendations)}")

                # Execute with first recommendation
                if recommendations:
                    preferred_assistant = recommendations[0]
                    print(f"  ⚡ Executing with {preferred_assistant}...")

                    result = await orchestrator.execute_task(task, assistant_preferences=[preferred_assistant])
                    if result['status'] == 'completed':
                        print("  ✅ Task completed successfully")
                        print(f"  📊 Execution time: {result.get('execution_time', 'N/A')}s")
                    else:
                        print(f"  ❌ Task failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"  ❌ Task delegation failed: {e}")

        # Test 4: Multi-assistant coordination
        print("\n🧪 TEST 4: Multi-Assistant Coordination Test")
        print("-" * 30)

        complex_task = "Design and implement a REST API for a task management system with user authentication"
        print(f"📋 Complex task: {complex_task}")

        try:
            result = await orchestrator.execute_task(complex_task)
            if result['status'] == 'completed':
                print("  ✅ Multi-assistant task completed")
                print(f"  👥 Assistants used: {result.get('assistants_used', [])}")
                print(f"  📊 Subtasks completed: {result.get('subtasks_count', 0)}")
                print(f"  ⏱️  Total execution time: {result.get('execution_time', 'N/A')}s")
            else:
                print(f"  ❌ Multi-assistant task failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"  ❌ Multi-assistant coordination failed: {e}")

        # Test 5: Performance metrics
        print("\n🧪 TEST 5: Performance Metrics")
        print("-" * 30)

        try:
            status = orchestrator.get_status()
            print(f"Total assistants configured: {status['total_assistants']}")
            print(f"Available assistants: {status['available_assistants']}")
            print(f"Orchestrator status: {status['status']}")

            # Get load balancer stats
            lb_stats = ai_manager.get_load_balancer_stats()
            print(f"Load balancer stats available: {'load_balancer_stats' in lb_stats}")

        except Exception as e:
            print(f"❌ Performance metrics failed: {e}")

        print("\n🎉 AI Integration Testing Complete!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set environment variables if needed
    if not os.getenv('GROQ_API_KEY'):
        print("⚠️  GROQ_API_KEY not set - Grok integration will be unavailable")
    if not os.getenv('GOOGLE_API_KEY'):
        print("⚠️  GOOGLE_API_KEY not set - Gemini integration will be unavailable")

    asyncio.run(test_ai_integrations())
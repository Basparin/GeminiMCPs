"""
AI Orchestrator - CES AI Assistant Management

Manages integration with multiple AI assistants (Grok, qwen-cli-coder, gemini-cli)
and coordinates task delegation based on capabilities and performance.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import subprocess
import json

from .cli_integration import AIAssistantManager, APIResult, CLIResult
from .specialization import AISpecializationManager
from ..core.conflict_resolution import ConflictResolver, AssistantOutput


class AIOrchestrator:
    """
    Orchestrates AI assistant interactions for CES.

    Manages multiple AI providers, handles task delegation,
    and optimizes assistant selection based on task requirements.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize real CLI integration manager
        self.cli_manager = AIAssistantManager()

        # Initialize specialization manager for advanced assistant selection
        self.specialization_manager = AISpecializationManager()

        # Initialize conflict resolution system
        self.conflict_resolver = ConflictResolver()

        # Available AI assistants and their capabilities (for backward compatibility)
        self.assistants = {
            'grok': {
                'name': 'Grok CLI',
                'capabilities': ['general_reasoning', 'coding', 'analysis'],
                'command': 'grok',
                'strengths': ['reasoning', 'general_knowledge']
            },
            'qwen': {
                'name': 'qwen-cli-coder',
                'capabilities': ['coding', 'code_generation', 'debugging'],
                'command': 'qwen-cli-coder',
                'strengths': ['code_generation', 'technical_tasks']
            },
            'gemini': {
                'name': 'Gemini CLI',
                'capabilities': ['analysis', 'documentation', 'review'],
                'command': 'gemini-cli',
                'strengths': ['code_analysis', 'documentation']
            }
        }

        self.logger.info("AI Orchestrator initialized with real CLI integration")

    def recommend_assistants(self, task_description: str, required_skills: List[str]) -> List[str]:
        """
        Recommend AI assistants for a task based on description and skills

        Args:
            task_description: Description of the task
            required_skills: List of required skills

        Returns:
            List of recommended assistant names
        """
        recommendations = []
        task_lower = task_description.lower()

        # Match skills to assistant capabilities
        for assistant_name, assistant_info in self.assistants.items():
            capability_match = any(skill in assistant_info['capabilities']
                                 for skill in required_skills)

            # Additional keyword matching
            keyword_match = False
            for keyword in assistant_info.get('keywords', []):
                if keyword in task_lower:
                    keyword_match = True
                    break

            if capability_match or keyword_match:
                recommendations.append(assistant_name)

        # If no specific matches, return general-purpose assistants
        if not recommendations:
            recommendations = ['grok', 'qwen']

        return recommendations

    async def execute_task(self, task_description: str, context: Optional[Dict[str, Any]] = None,
                          assistant_preferences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute a task using multiple AI assistants for enhanced coordination

        Args:
            task_description: Task to execute
            context: Additional context for the task
            assistant_preferences: Preferred assistants to use

        Returns:
            Task execution result with multi-assistant coordination
        """
        start_time = datetime.now()

        # Determine if task requires multi-assistant coordination
        task_complexity = self._assess_task_complexity(task_description, context)
        requires_multiple_assistants = task_complexity > 6 or len(assistant_preferences or []) > 1

        if requires_multiple_assistants:
            return await self._execute_multi_assistant_task(task_description, context, assistant_preferences, start_time)
        else:
            return await self._execute_single_assistant_task(task_description, context, assistant_preferences, start_time)

    async def _execute_multi_assistant_task(self, task_description: str, context: Optional[Dict[str, Any]],
                                          assistant_preferences: Optional[List[str]], start_time: datetime) -> Dict[str, Any]:
        """
        Execute task with multiple AI assistants working simultaneously
        """
        self.logger.info(f"Executing multi-assistant task: {task_description[:50]}...")

        # Decompose task into subtasks
        subtasks = self._decompose_task(task_description, context)

        # Select assistants for each subtask
        assistant_assignments = await self._assign_assistants_to_subtasks(subtasks, assistant_preferences)

        # Execute subtasks in parallel
        execution_results = await self._execute_parallel_subtasks(subtasks, assistant_assignments, context)

        # Synthesize results
        final_result = await self._synthesize_multi_assistant_results(execution_results, task_description)

        execution_time = (datetime.now() - start_time).total_seconds()

        return {
            "status": "completed",
            "execution_mode": "multi_assistant",
            "subtasks_count": len(subtasks),
            "assistants_used": list(set(assistant_assignments.values())),
            "subtask_results": execution_results,
            "synthesized_result": final_result,
            "execution_time": execution_time,
            "timestamp": start_time.isoformat()
        }

    async def _execute_single_assistant_task(self, task_description: str, context: Optional[Dict[str, Any]],
                                           assistant_preferences: Optional[List[str]], start_time: datetime) -> Dict[str, Any]:
        """
        Execute task with single AI assistant (fallback method)
        """
        # Select assistant
        assistant = self._select_assistant(task_description, assistant_preferences)

        if not assistant:
            return {
                "status": "failed",
                "error": "No suitable AI assistant available",
                "timestamp": start_time.isoformat()
            }

        # Execute with selected assistant
        try:
            result = await self._execute_with_assistant(assistant, task_description, context)
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "status": "completed",
                "execution_mode": "single_assistant",
                "assistant_used": assistant,
                "result": result,
                "execution_time": execution_time,
                "timestamp": start_time.isoformat()
            }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Task execution failed with {assistant}: {e}")
            return {
                "status": "failed",
                "assistant_used": assistant,
                "error": str(e),
                "execution_time": execution_time,
                "timestamp": start_time.isoformat()
            }

    def _assess_task_complexity(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Assess task complexity to determine if multi-assistant execution is needed"""
        complexity = 0

        # Length-based complexity
        if len(task_description) > 200:
            complexity += 2
        elif len(task_description) > 100:
            complexity += 1

        # Keyword-based complexity
        complex_keywords = ['complex', 'multiple', 'integrate', 'optimize', 'architecture', 'system']
        task_lower = task_description.lower()
        complexity += sum(1 for keyword in complex_keywords if keyword in task_lower)

        # Context-based complexity
        if context:
            if context.get('has_dependencies'):
                complexity += 1
            if context.get('requires_multiple_skills'):
                complexity += 2
            if len(context.get('subtasks', [])) > 1:
                complexity += 1

        return complexity

    def _decompose_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Decompose complex task into subtasks for parallel execution"""
        subtasks = []

        # Basic decomposition based on task type
        task_lower = task_description.lower()

        if 'analyze' in task_lower or 'review' in task_lower:
            subtasks.extend([
                {'id': 'analysis', 'description': f'Analyze {task_description}', 'priority': 'high'},
                {'id': 'review', 'description': f'Review analysis results for {task_description}', 'priority': 'medium'}
            ])

        if 'implement' in task_lower or 'develop' in task_lower:
            subtasks.extend([
                {'id': 'design', 'description': f'Design solution for {task_description}', 'priority': 'high'},
                {'id': 'implement', 'description': f'Implement the designed solution', 'priority': 'high'},
                {'id': 'test', 'description': f'Test the implementation', 'priority': 'medium'}
            ])

        if 'optimize' in task_lower or 'performance' in task_lower:
            subtasks.extend([
                {'id': 'profile', 'description': f'Profile current performance for {task_description}', 'priority': 'high'},
                {'id': 'optimize', 'description': f'Optimize performance bottlenecks', 'priority': 'high'},
                {'id': 'validate', 'description': f'Validate performance improvements', 'priority': 'medium'}
            ])

        # If no specific decomposition, create generic subtasks
        if not subtasks:
            subtasks = [
                {'id': 'main', 'description': task_description, 'priority': 'high'},
                {'id': 'verification', 'description': f'Verify results for {task_description}', 'priority': 'medium'}
            ]

        return subtasks

    async def _assign_assistants_to_subtasks(self, subtasks: List[Dict[str, Any]],
                                           preferences: Optional[List[str]] = None) -> Dict[str, str]:
        """Assign AI assistants to subtasks using specialization manager for optimal selection"""
        assignments = {}

        for subtask in subtasks:
            subtask_desc = subtask['description']

            # Use preferences if available
            if preferences:
                for pref in preferences:
                    if pref in self.assistants and self._is_assistant_available(pref):
                        assignments[subtask['id']] = pref
                        break

            # Use specialization manager for intelligent assignment
            if subtask['id'] not in assignments:
                try:
                    # Analyze subtask requirements
                    task_profile = await self.specialization_manager.analyze_task_requirements(subtask_desc)

                    # Get assistant recommendations
                    recommendations = await self.specialization_manager.recommend_assistants(task_profile)

                    # Select best available assistant
                    for assistant_name, confidence, factors in recommendations:
                        if self._is_assistant_available(assistant_name):
                            assignments[subtask['id']] = assistant_name
                            self.logger.info(f"Assigned {assistant_name} to subtask {subtask['id']} (confidence: {confidence:.2f})")
                            break
                    else:
                        # Fallback to basic selection
                        assignments[subtask['id']] = self._fallback_assistant_selection(subtask_desc)

                except Exception as e:
                    self.logger.warning(f"Specialization analysis failed for subtask {subtask['id']}: {e}")
                    assignments[subtask['id']] = self._fallback_assistant_selection(subtask_desc)

        return assignments

    def _fallback_assistant_selection(self, subtask_desc: str) -> str:
        """Fallback assistant selection when specialization analysis fails"""
        if 'analyze' in subtask_desc.lower() or 'review' in subtask_desc.lower():
            return 'gemini-cli' if self._is_assistant_available('gemini-cli') else 'grok'
        elif 'implement' in subtask_desc.lower() or 'code' in subtask_desc.lower():
            return 'qwen-cli-coder' if self._is_assistant_available('qwen-cli-coder') else 'grok'
        elif 'test' in subtask_desc.lower() or 'debug' in subtask_desc.lower():
            return 'qwen-cli-coder' if self._is_assistant_available('qwen-cli-coder') else 'gemini-cli'
        else:
            return 'grok'  # Default to general-purpose

    async def _execute_parallel_subtasks(self, subtasks: List[Dict[str, Any]],
                                       assignments: Dict[str, str],
                                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute subtasks in parallel with assigned assistants"""
        import asyncio

        async def execute_subtask(subtask: Dict[str, Any]) -> Dict[str, Any]:
            assistant = assignments.get(subtask['id'], 'grok')
            try:
                result = await self._execute_with_assistant(assistant, subtask['description'], context)
                return {
                    'subtask_id': subtask['id'],
                    'assistant': assistant,
                    'result': result,
                    'status': 'completed',
                    'priority': subtask['priority']
                }
            except Exception as e:
                self.logger.error(f"Subtask {subtask['id']} failed with {assistant}: {e}")
                return {
                    'subtask_id': subtask['id'],
                    'assistant': assistant,
                    'error': str(e),
                    'status': 'failed',
                    'priority': subtask['priority']
                }

        # Execute all subtasks concurrently
        tasks = [execute_subtask(subtask) for subtask in subtasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Organize results by subtask ID
        organized_results = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Parallel execution error: {result}")
                continue
            organized_results[result['subtask_id']] = result

        return organized_results

    async def _synthesize_multi_assistant_results(self, execution_results: Dict[str, Any],
                                               original_task: str) -> Dict[str, Any]:
        """Synthesize results from multiple assistants using advanced conflict resolution"""
        successful_results = [r for r in execution_results.values() if r['status'] == 'completed']
        failed_results = [r for r in execution_results.values() if r['status'] == 'failed']

        if not successful_results:
            return {
                'status': 'failed',
                'error': 'All subtasks failed',
                'failures': failed_results
            }

        # Convert to AssistantOutput format for conflict resolution
        assistant_outputs = []
        for result in successful_results:
            output = AssistantOutput(
                assistant_name=result['assistant'],
                content=str(result['result']),
                confidence_score=0.8,  # Default confidence, could be enhanced
                quality_metrics={'completeness': 0.8, 'relevance': 0.9},
                metadata={'priority': result['priority'], 'subtask_id': result['subtask_id']}
            )
            assistant_outputs.append(output)

        # Apply conflict resolution
        resolution_result = await self.conflict_resolver.resolve_conflicts(
            assistant_outputs,
            context={'original_task': original_task}
        )

        # Create synthesized response
        synthesized = {
            'status': 'completed',
            'original_task': original_task,
            'subtasks_completed': len(successful_results),
            'subtasks_failed': len(failed_results),
            'assistants_used': list(set(r['assistant'] for r in successful_results)),
            'results': successful_results,
            'synthesized_response': resolution_result.resolved_content,
            'resolution_confidence': resolution_result.confidence_score,
            'resolution_strategy': resolution_result.resolution_strategy.value,
            'conflict_analysis': [c.__dict__ for c in resolution_result.conflict_analysis],
            'processing_time_ms': resolution_result.processing_time_ms
        }

        if failed_results:
            synthesized['warnings'] = f"{len(failed_results)} subtasks failed"

        return synthesized

    def _combine_assistant_responses(self, results: List[Dict[str, Any]]) -> str:
        """Combine responses from multiple assistants into a coherent answer"""
        if len(results) == 1:
            return results[0]['result']

        # Group by assistant type for better synthesis
        responses_by_assistant = {}
        for result in results:
            assistant = result['assistant']
            if assistant not in responses_by_assistant:
                responses_by_assistant[assistant] = []
            responses_by_assistant[assistant].append(result['result'])

        # Create combined response
        combined_parts = []
        for assistant, responses in responses_by_assistant.items():
            combined_parts.append(f"{assistant.upper()} Analysis:")
            combined_parts.extend(responses)

        return "\n\n".join(combined_parts)

    def _select_assistant(self, task_description: str,
                          preferences: Optional[List[str]] = None) -> Optional[str]:
        """Select the most appropriate assistant for a task"""
        if preferences:
            # Try preferred assistants first
            for pref in preferences:
                if pref in self.assistants and self._is_assistant_available(pref):
                    return pref

        # Fallback to recommendation logic
        recommendations = self.recommend_assistants(task_description, [])
        for rec in recommendations:
            if self._is_assistant_available(rec):
                return rec

        return None

    def _is_assistant_available(self, assistant_name: str) -> bool:
        """Check if an AI assistant is available"""
        if assistant_name not in self.assistants:
            return False

        # Check availability using real CLI manager
        assistant = self.cli_manager.get_assistant(assistant_name)
        if assistant:
            return assistant.is_available()

        return False

    def _prepare_prompt(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare a comprehensive prompt with context"""
        prompt_parts = [f"Task: {task_description}"]

        if context:
            if 'task_history' in context and context['task_history']:
                prompt_parts.append("\nRecent similar tasks:")
                for task in context['task_history'][:3]:
                    prompt_parts.append(f"- {task['description'][:100]}...")

            if 'user_preferences' in context and context['user_preferences']:
                prompt_parts.append(f"\nUser preferences: {context['user_preferences']}")

        return "\n".join(prompt_parts)

    async def _execute_with_assistant(self, assistant_name: str, task_description: str, context: Optional[Dict[str, Any]] = None) -> Union[APIResult, CLIResult]:
        """
        Execute a task with a specific AI assistant using real CLI integration

        Args:
            assistant_name: Name of the assistant to use
            task_description: Task description
            context: Additional context

        Returns:
            Result from assistant execution
        """
        self.logger.info(f"Executing task with {assistant_name}")

        try:
            # Use the real CLI manager to execute the task
            result = await self.cli_manager.execute_with_assistant(assistant_name, task_description, context)

            if isinstance(result, APIResult):
                if result.success:
                    return result.response
                else:
                    self.logger.error(f"API call failed for {assistant_name}: {result.error}")
                    return f"Error: {result.error}"
            elif isinstance(result, CLIResult):
                if result.success:
                    return result.output
                else:
                    self.logger.error(f"CLI execution failed for {assistant_name}: {result.error}")
                    return f"Error: {result.error}"
            else:
                return f"Unknown result type from {assistant_name}"

        except Exception as e:
            self.logger.error(f"Failed to execute with {assistant_name}: {e}")
            return f"Execution failed: {str(e)}"

    def get_available_assistants(self) -> List[Dict[str, Any]]:
        """Get list of available AI assistants"""
        available = []
        for name, info in self.assistants.items():
            if self._is_assistant_available(name):
                available.append({
                    "name": name,
                    "display_name": info["name"],
                    "capabilities": info["capabilities"],
                    "strengths": info["strengths"]
                })
        return available

    async def test_assistant_connection(self, assistant_name: str) -> Dict[str, Any]:
        """
        Test connection to an AI assistant

        Args:
            assistant_name: Name of the assistant to test

        Returns:
            Test result
        """
        if assistant_name not in self.assistants:
            return {"status": "error", "message": "Assistant not found"}

        try:
            # Test execution with real CLI manager
            test_result = await self._execute_with_assistant(assistant_name, "Test connection - please respond with 'Connection successful'")
            return {
                "status": "success",
                "assistant": assistant_name,
                "response": test_result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "assistant": assistant_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_status(self) -> Dict[str, Any]:
        """Get AI orchestrator status"""
        available_count = len(self.get_available_assistants())
        cli_status = self.cli_manager.get_all_status()

        return {
            "status": "operational",
            "total_assistants": len(self.assistants),
            "available_assistants": available_count,
            "assistants": list(self.assistants.keys()),
            "cli_integration_status": cli_status,
            "last_check": datetime.now().isoformat()
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "component": "AI Orchestrator",
            "timestamp": datetime.now().isoformat(),
            "orchestrator_status": "healthy",
            "checks": {}
        }

        # Check CLI manager health
        cli_health = self.cli_manager.health_check()
        health_status["checks"]["cli_manager"] = cli_health

        # Check assistant availability
        available_assistants = self.get_available_assistants()
        health_status["checks"]["assistant_availability"] = {
            "status": "healthy" if len(available_assistants) > 0 else "warning",
            "details": f"{len(available_assistants)} assistants available"
        }

        # Overall health
        all_healthy = all(
            check.get("status") in ["healthy", "skipped"]
            for check in health_status["checks"].values()
            if isinstance(check, dict)
        )
        health_status["overall_status"] = "healthy" if all_healthy else "degraded"

        return health_status
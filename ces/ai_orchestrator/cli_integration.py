"""
CES AI Assistant CLI Integration Module

Provides real CLI integration for AI assistants (Grok, qwen-cli-coder, gemini-cli)
with proper error handling, health monitoring, and MCP protocol support.
"""

import logging
import asyncio
import subprocess
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
import os
import sys

from groq import Groq
from google import generativeai as genai
import openai


@dataclass
class CLIResult:
    """Result from CLI execution"""
    success: bool
    output: str
    error: str
    exit_code: int
    execution_time: float
    timestamp: str


@dataclass
class APIResult:
    """Result from API call"""
    success: bool
    response: str
    error: str
    tokens_used: int
    execution_time: float
    timestamp: str


class GrokCLIIntegration:
    """
    Real Grok CLI integration using Groq API
    """

    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = None
        self.model = "mixtral-8x7b-32768"  # Default model

        if self.api_key:
            self.client = Groq(api_key=self.api_key)
            self.logger.info("Grok CLI integration initialized with API key")
        else:
            self.logger.warning("Grok API key not found, integration will be unavailable")

    def is_available(self) -> bool:
        """Check if Grok integration is available"""
        return self.client is not None

    async def execute_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> APIResult:
        """
        Execute a task using Grok API

        Args:
            task_description: Description of the task
            context: Additional context

        Returns:
            APIResult with execution details
        """
        start_time = datetime.now()

        if not self.is_available():
            return APIResult(
                success=False,
                response="",
                error="Grok API key not configured",
                tokens_used=0,
                execution_time=0.0,
                timestamp=start_time.isoformat()
            )

        try:
            # Prepare the prompt
            prompt = self._prepare_grok_prompt(task_description, context)

            # Make API call
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                    temperature=0.7
                )
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            return APIResult(
                success=True,
                response=response.choices[0].message.content,
                error="",
                tokens_used=response.usage.total_tokens,
                execution_time=execution_time,
                timestamp=start_time.isoformat()
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Grok API call failed: {e}")

            return APIResult(
                success=False,
                response="",
                error=str(e),
                tokens_used=0,
                execution_time=execution_time,
                timestamp=start_time.isoformat()
            )

    def _prepare_grok_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare a comprehensive prompt for Grok"""
        prompt_parts = [
            f"You are Grok, a helpful AI assistant integrated into the CES (Cognitive Enhancement System).",
            f"Task: {task}",
            "",
            "Please provide a clear, actionable response. If this involves code, provide complete, working solutions."
        ]

        if context:
            if 'codebase_path' in context:
                prompt_parts.append(f"Codebase: {context['codebase_path']}")

            if 'file_path' in context:
                prompt_parts.append(f"File: {context['file_path']}")

            if 'task_history' in context and context['task_history']:
                prompt_parts.append("")
                prompt_parts.append("Recent context:")
                for history_item in context['task_history'][:3]:
                    prompt_parts.append(f"- {history_item.get('description', 'Unknown task')[:100]}...")

        return "\n".join(prompt_parts)

    def get_status(self) -> Dict[str, Any]:
        """Get Grok integration status"""
        return {
            "name": "Grok CLI",
            "available": self.is_available(),
            "model": self.model,
            "has_api_key": bool(self.api_key),
            "last_check": datetime.now().isoformat()
        }


class QwenCLICoderIntegration:
    """
    qwen-cli-coder integration
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.command = "qwen-cli-coder"
        self.timeout = 300  # 5 minutes

    def is_available(self) -> bool:
        """Check if qwen-cli-coder is available"""
        try:
            result = subprocess.run(
                [self.command, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

    async def execute_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> CLIResult:
        """
        Execute a task using qwen-cli-coder

        Args:
            task_description: Description of the task
            context: Additional context

        Returns:
            CLIResult with execution details
        """
        start_time = datetime.now()

        if not self.is_available():
            return CLIResult(
                success=False,
                output="",
                error="qwen-cli-coder not available",
                exit_code=-1,
                execution_time=0.0,
                timestamp=start_time.isoformat()
            )

        try:
            # Prepare command arguments
            cmd_args = [self.command]

            # Add task description
            cmd_args.extend(["--task", task_description])

            # Add context if available
            if context:
                if 'file_path' in context:
                    cmd_args.extend(["--file", context['file_path']])
                if 'codebase_path' in context:
                    cmd_args.extend(["--codebase", context['codebase_path']])

            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.get('working_directory', '.')
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )

                execution_time = (datetime.now() - start_time).total_seconds()

                return CLIResult(
                    success=process.returncode == 0,
                    output=stdout.decode('utf-8', errors='replace'),
                    error=stderr.decode('utf-8', errors='replace'),
                    exit_code=process.returncode,
                    execution_time=execution_time,
                    timestamp=start_time.isoformat()
                )

            except asyncio.TimeoutError:
                process.kill()
                execution_time = (datetime.now() - start_time).total_seconds()

                return CLIResult(
                    success=False,
                    output="",
                    error=f"Command timed out after {self.timeout} seconds",
                    exit_code=-1,
                    execution_time=execution_time,
                    timestamp=start_time.isoformat()
                )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"qwen-cli-coder execution failed: {e}")

            return CLIResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time=execution_time,
                timestamp=start_time.isoformat()
            )

    def get_status(self) -> Dict[str, Any]:
        """Get qwen-cli-coder integration status"""
        return {
            "name": "qwen-cli-coder",
            "available": self.is_available(),
            "command": self.command,
            "timeout": self.timeout,
            "last_check": datetime.now().isoformat()
        }


class GeminiCLIIntegration:
    """
    Gemini CLI integration using Google AI API
    """

    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = "gemini-pro"

        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.logger.info("Gemini CLI integration initialized with API key")
        else:
            self.logger.warning("Google API key not found, Gemini integration will be unavailable")

    def is_available(self) -> bool:
        """Check if Gemini integration is available"""
        return bool(self.api_key)

    async def execute_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> APIResult:
        """
        Execute a task using Gemini API

        Args:
            task_description: Description of the task
            context: Additional context

        Returns:
            APIResult with execution details
        """
        start_time = datetime.now()

        if not self.is_available():
            return APIResult(
                success=False,
                response="",
                error="Google API key not configured",
                tokens_used=0,
                execution_time=0.0,
                timestamp=start_time.isoformat()
            )

        try:
            # Prepare the prompt
            prompt = self._prepare_gemini_prompt(task_description, context)

            # Initialize model
            model = genai.GenerativeModel(self.model)

            # Make API call
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(prompt)
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            return APIResult(
                success=True,
                response=response.text,
                error="",
                tokens_used=getattr(response, 'usage_metadata', {}).get('total_token_count', 0),
                execution_time=execution_time,
                timestamp=start_time.isoformat()
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Gemini API call failed: {e}")

            return APIResult(
                success=False,
                response="",
                error=str(e),
                tokens_used=0,
                execution_time=execution_time,
                timestamp=start_time.isoformat()
            )

    def _prepare_gemini_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare a comprehensive prompt for Gemini"""
        prompt_parts = [
            "You are Gemini, an AI assistant integrated into the CES (Cognitive Enhancement System).",
            f"Task: {task}",
            "",
            "Please provide a clear, well-structured response. Focus on analysis and documentation tasks."
        ]

        if context:
            if 'codebase_path' in context:
                prompt_parts.append(f"Codebase: {context['codebase_path']}")

            if 'file_path' in context:
                prompt_parts.append(f"File: {context['file_path']}")

            if 'analysis_type' in context:
                prompt_parts.append(f"Analysis Type: {context['analysis_type']}")

        return "\n".join(prompt_parts)

    def get_status(self) -> Dict[str, Any]:
        """Get Gemini integration status"""
        return {
            "name": "Gemini CLI",
            "available": self.is_available(),
            "model": self.model,
            "has_api_key": bool(self.api_key),
            "last_check": datetime.now().isoformat()
        }


class AIAssistantManager:
    """
    Manager for all AI assistant integrations
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize all assistants
        self.assistants = {
            'grok': GrokCLIIntegration(),
            'qwen': QwenCLICoderIntegration(),
            'gemini': GeminiCLIIntegration()
        }

        self.logger.info("AI Assistant Manager initialized")

    def get_assistant(self, name: str) -> Optional[Any]:
        """Get an AI assistant by name"""
        return self.assistants.get(name)

    def get_available_assistants(self) -> List[str]:
        """Get list of available assistants"""
        return [name for name, assistant in self.assistants.items() if assistant.is_available()]

    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all assistants"""
        return {
            name: assistant.get_status()
            for name, assistant in self.assistants.items()
        }

    async def execute_with_assistant(self, assistant_name: str, task: str,
                                   context: Optional[Dict[str, Any]] = None) -> Union[APIResult, CLIResult]:
        """
        Execute a task with a specific assistant

        Args:
            assistant_name: Name of the assistant
            task: Task description
            context: Additional context

        Returns:
            Result from assistant execution
        """
        assistant = self.get_assistant(assistant_name)
        if not assistant:
            # Return a generic error result
            start_time = datetime.now()
            return APIResult(
                success=False,
                response="",
                error=f"Assistant '{assistant_name}' not found",
                tokens_used=0,
                execution_time=0.0,
                timestamp=start_time.isoformat()
            )

        return await assistant.execute_task(task, context)

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all assistants"""
        health_status = {
            "component": "AI Assistant Manager",
            "timestamp": datetime.now().isoformat(),
            "assistants": {}
        }

        for name, assistant in self.assistants.items():
            try:
                status = assistant.get_status()
                health_status["assistants"][name] = {
                    "status": "healthy" if status["available"] else "unhealthy",
                    "details": status
                }
            except Exception as e:
                health_status["assistants"][name] = {
                    "status": "error",
                    "details": str(e)
                }

        # Overall health
        all_healthy = all(
            assistant["status"] == "healthy"
            for assistant in health_status["assistants"].values()
        )
        health_status["overall_status"] = "healthy" if all_healthy else "degraded"

        return health_status
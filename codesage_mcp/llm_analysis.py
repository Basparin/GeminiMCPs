"""
LLM Analysis Module for CodeSage MCP Server.

This module contains the logic for analyzing code using Large Language Models (LLMs).
It provides the LLMAnalysisManager class, which is responsible for:
- Summarizing code sections.
- Profiling code performance.
- Suggesting code improvements.
- Generating unit tests.
- Auto-documenting tools.
- Interacting with various LLM APIs (Groq, OpenRouter, Google AI).

Classes:
    LLMAnalysisManager: Manages code analysis using LLMs.
"""

import textwrap
import tempfile
import os
import cProfile
import pstats
import io
import importlib.util
import ast
import inspect
import re  # New import
import time
import asyncio
import logging
from typing import Optional, Tuple
from codesage_mcp.utils import _count_todo_fixme_comments, safe_read_file  # New import
import json  # New import
from codesage_mcp.config import (
    HTTP_REQUEST_TIMEOUT,
    HTTP_MAX_RETRIES,
    HTTP_RETRY_BACKOFF_FACTOR,
    HTTP_CONNECTION_POOL_SIZE,
)

# Set up logger
logger = logging.getLogger(__name__)


class LLMAnalysisManager:
    """Manages code analysis using Large Language Models (LLMs).

    This class is responsible for interacting with various LLM APIs (Groq,
    OpenRouter, Google AI) to perform tasks like code summarization, performance
    profiling, code improvement suggestions, unit test generation, and
    auto-documentation.

    Attributes:
        groq_client: Client for Groq API.
        openrouter_client: Client for OpenRouter API.
        google_ai_client: Client for Google AI API.
    """

    def __init__(self, groq_client, openrouter_client, google_ai_client):
        """Initializes the LLMAnalysisManager with API clients."""
        self.groq_client = groq_client
        self.openrouter_client = openrouter_client
        self.google_ai_client = google_ai_client

        # Initialize semaphores for concurrent request management
        self._semaphores = {
            "groq": asyncio.Semaphore(HTTP_CONNECTION_POOL_SIZE // 3 or 1),
            "openrouter": asyncio.Semaphore(HTTP_CONNECTION_POOL_SIZE // 3 or 1),
            "google": asyncio.Semaphore(HTTP_CONNECTION_POOL_SIZE // 3 or 1),
        }

        # Rate limiting tracking
        self._last_request_times = {
            "groq": {},
            "openrouter": {},
            "google": {},
        }

        # Request counters for monitoring
        self._request_counts = {
            "groq": 0,
            "openrouter": 0,
            "google": 0,
        }

    def _get_provider_name(self, llm_model: str) -> str:
        """Get the provider name from the LLM model identifier."""
        if llm_model.startswith("openrouter/"):
            return "openrouter"
        elif llm_model.startswith("llama3") or llm_model.startswith("mixtral"):
            return "groq"
        elif llm_model.startswith("google/"):
            return "google"
        else:
            return "unknown"

    def _check_rate_limit(self, provider: str) -> bool:
        """
        Check if the request should be rate limited.

        Args:
            provider: The LLM provider name

        Returns:
            bool: True if request should proceed, False if rate limited
        """
        current_time = time.time()
        provider_times = self._last_request_times[provider]

        # Simple rate limiting: max 10 requests per minute per provider
        max_requests_per_minute = 10
        time_window = 60

        # Clean old entries
        cutoff_time = current_time - time_window
        provider_times = {
            model: req_time
            for model, req_time in provider_times.items()
            if req_time > cutoff_time
        }
        self._last_request_times[provider] = provider_times

        # Check if we're within rate limits
        if len(provider_times) >= max_requests_per_minute:
            return False

        return True

    def _record_request(self, provider: str, model: str):
        """Record a request for rate limiting purposes."""
        current_time = time.time()
        self._last_request_times[provider][model] = current_time
        self._request_counts[provider] += 1

    def get_connection_pool_stats(self) -> dict:
        """
        Get connection pool statistics for monitoring.

        Returns:
            dict: Connection pool statistics
        """
        stats = {
            "connection_pool": {
                "max_connections": HTTP_CONNECTION_POOL_SIZE,
                "timeout_seconds": HTTP_REQUEST_TIMEOUT,
                "max_retries": HTTP_MAX_RETRIES,
                "retry_backoff_factor": HTTP_RETRY_BACKOFF_FACTOR,
            },
            "providers": {},
            "rate_limiting": {},
        }

        # Provider-specific stats
        for provider in ["groq", "openrouter", "google"]:
            semaphore = self._semaphores[provider]
            provider_times = self._last_request_times[provider]

            # Calculate requests per minute
            current_time = time.time()
            recent_requests = sum(
                1 for req_time in provider_times.values()
                if current_time - req_time < 60
            )

            stats["providers"][provider] = {
                "active_connections": semaphore._value if hasattr(semaphore, '_value') else 0,
                "total_requests": self._request_counts[provider],
                "requests_per_minute": recent_requests,
                "models_used": list(provider_times.keys()),
            }

            # Rate limiting stats
            stats["rate_limiting"][provider] = {
                "current_window_requests": len(provider_times),
                "rate_limit_threshold": 10,  # requests per minute
                "is_rate_limited": len(provider_times) >= 10,
            }

        return stats

    def _summarize_with_llm(self, code_snippet: str, llm_model: str) -> str:
        """
        Unified method to summarize a code snippet using any supported LLM provider.

        Args:
            code_snippet (str): The code snippet to summarize.
            llm_model (str): The LLM model to use for summarization.

        Returns:
            str: The summary of the code snippet or an error message if it fails.
        """
        system_message = self._build_system_prompt(
            "You are a helpful assistant that summarizes code."
        )

        template = """
            Please summarize the following code snippet:

            ```python
            {code_snippet}
            ```

            Provide a clear, concise summary that captures the main functionality and purpose of the code.
        """
        prompt = self._build_user_prompt(template, code_snippet=code_snippet)

        response_content, error_message = self._get_llm_response(
            prompt, llm_model, system_message
        )
        if error_message:
            return f"Error during summarization: {error_message}"
        return response_content

    def summarize_code_section(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        llm_model: str,
    ) -> str:
        """
        Resume una sección específica de código usando un LLM elegido.

        Args:
            file_path (str): Ruta al archivo que contiene el código a resumir.
            start_line (int): Número de línea inicial de la sección a resumir.
            end_line (int): Número de línea final de la sección a resumir.
            llm_model (str): Modelo LLM a usar para el resumen (e.g., 'llama3-8b-8192',
                'openrouter/google/gemini-pro', 'google/gemini-pro').

        Returns:
            str: Resumen de la sección de código o un mensaje de error si falla.

        Raises:
            FileNotFoundError: Si el archivo especificado no existe.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        all_lines = safe_read_file(file_path, as_lines=True)
        lines = all_lines[start_line - 1 : end_line]

        if not lines:
            return "No code found in the specified line range."

        code_snippet = "".join(lines)

        # Use the unified LLM summarization method
        return self._summarize_with_llm(code_snippet, llm_model)

    async def _retry_llm_call(self, llm_call_func, provider: str, model: str, *args, **kwargs) -> tuple[str, str]:
        """
        Retry LLM API calls with exponential backoff and rate limiting for connection resilience.

        Args:
            llm_call_func: The LLM API call function to retry
            provider: The LLM provider name
            model: The model name
            *args: Positional arguments for the LLM call
            **kwargs: Keyword arguments for the LLM call

        Returns:
            tuple[str, str]: (response_content, error_message)
        """
        # Check rate limiting first
        if not self._check_rate_limit(provider):
            return None, f"Rate limit exceeded for {provider}. Please try again later."

        last_exception = None

        async with self._semaphores[provider]:
            for attempt in range(HTTP_MAX_RETRIES + 1):
                try:
                    # Record the request
                    self._record_request(provider, model)

                    # Make the API call
                    result = await llm_call_func(*args, **kwargs)
                    return result
                except Exception as e:
                    last_exception = e

                    # Check if this is a retryable error
                    if self._is_retryable_error(e):
                        if attempt < HTTP_MAX_RETRIES:
                            # Calculate backoff delay with jitter
                            delay = HTTP_RETRY_BACKOFF_FACTOR * (2 ** attempt)
                            delay = min(delay, 60)  # Cap at 60 seconds

                            logger.warning(f"LLM API call failed (attempt {attempt + 1}/{HTTP_MAX_RETRIES + 1}): {e}. Retrying in {delay:.2f}s...")
                            await asyncio.sleep(delay)
                            continue

                    # Not retryable or max retries reached
                    break

        # All retries failed
        error_response = self._handle_llm_error(last_exception, "LLM call with retries")
        return None, error_response["error"]["message"]

    def _is_retryable_error(self, exception: Exception) -> bool:
        """
        Determine if an exception is retryable based on error type and message.

        Args:
            exception: The exception to check

        Returns:
            bool: True if the error is retryable
        """
        error_message = str(exception).lower()
        error_type = type(exception).__name__

        # Network and connection errors
        retryable_errors = [
            "connectionreseterror",
            "connectionabortederror",
            "connectionrefusederror",
            "timeouterror",
            "readtimeout",
            "connecttimeout",
            "connectionpool",
            "connection reset by peer",
            "connection timed out",
            "connection refused",
            "connection aborted",
            "network is unreachable",
            "temporary failure in name resolution",
            "badstatusline",
            "remotely closed",
        ]

        # Rate limiting
        if "rate limit" in error_message or "429" in error_message:
            return True

        # Check error type
        if any(error_type.lower().endswith(retryable) for retryable in ["error", "exception"]):
            if any(keyword in error_message for keyword in retryable_errors):
                return True

        return False

    def _get_llm_response(
        self, prompt: str, llm_model: str, system_message: str = None
    ) -> tuple[str, str]:
        """
        Helper to get LLM response from various providers using standardized patterns with retry logic.

        Args:
            prompt: The user prompt to send to the LLM
            llm_model: The LLM model identifier
            system_message: Optional system message (defaults to standard assistant role)

        Returns:
            tuple[str, str]: (response_content, error_message)
        """
        if system_message is None:
            system_message = self._build_system_prompt()

        provider = self._get_provider_name(llm_model)

        async def _call_openrouter():
            if not self.openrouter_client:
                raise ValueError("Error: OPENROUTER_API_KEY not configured.")

            chat_completion = await asyncio.to_thread(
                self.openrouter_client.chat.completions.create,
                model=llm_model.replace("openrouter/", "", 1),
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                timeout=HTTP_REQUEST_TIMEOUT,
            )
            return chat_completion.choices[0].message.content, None

        async def _call_groq():
            if not self.groq_client:
                raise ValueError("Error: GROQ_API_KEY not configured.")

            chat_completion = await asyncio.to_thread(
                self.groq_client.chat.completions.create,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                model=llm_model,
                timeout=HTTP_REQUEST_TIMEOUT,
            )
            return chat_completion.choices[0].message.content, None

        async def _call_google():
            if not self.google_ai_client:
                raise ValueError("Error: GOOGLE_API_KEY not configured.")

            response = await asyncio.to_thread(
                self.google_ai_client.generate_content,
                prompt
            )
            return response.text, None

        try:
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if llm_model.startswith("openrouter/"):
                return loop.run_until_complete(self._retry_llm_call(_call_openrouter, provider, llm_model))
            elif llm_model.startswith("llama3") or llm_model.startswith("mixtral"):
                return loop.run_until_complete(self._retry_llm_call(_call_groq, provider, llm_model))
            elif llm_model.startswith("google/"):
                return loop.run_until_complete(self._retry_llm_call(_call_google, provider, llm_model))
            else:
                return None, f"LLM model '{llm_model}' not supported yet."

        except Exception as e:
            error_response = self._handle_llm_error(e, f"{llm_model} LLM call")
            return None, error_response["error"]["message"]

    def _get_llm_suggestions(
        self, prompt: str, system_message: str = None
    ) -> list[dict]:
        """
        Get suggestions from multiple LLM providers using standardized patterns.

        Args:
            prompt (str): The prompt to send to the LLMs.
            system_message (str): Optional system message (defaults to standard assistant role).

        Returns:
            list[dict]: List of suggestions from available LLM providers.
        """
        if system_message is None:
            system_message = self._build_system_prompt()

        suggestions = []
        providers = [
            ("llama3-8b-8192", "Groq (Llama3)"),
            ("openrouter/google/gemini-pro", "OpenRouter (Gemini)"),
            ("google/gemini-pro", "Google AI (Gemini)"),
        ]

        for model, provider_name in providers:
            try:
                response_content, error_message = self._get_llm_response(
                    prompt, model, system_message
                )
                if error_message:
                    suggestions.append(
                        {
                            "provider": provider_name,
                            "error": error_message,
                        }
                    )
                else:
                    suggestions.append(
                        {
                            "provider": provider_name,
                            "suggestions": response_content,
                        }
                    )
            except Exception as e:
                error_response = self._handle_llm_error(
                    e, f"{provider_name} LLM suggestions"
                )
                suggestions.append(
                    {
                        "provider": provider_name,
                        "error": error_response["error"]["message"],
                    }
                )

        # If no LLM providers are configured, provide a basic static analysis
        if not suggestions:
            suggestions.append(
                {
                    "provider": "Static Analysis",
                    "suggestions": "No LLM providers configured. Please configure API keys for Groq, OpenRouter, or Google AI to get detailed suggestions.",
                }
            )

        return suggestions

    def _validate_and_normalize_line_numbers(
        self, start_line: int, end_line: int, total_lines: int
    ) -> tuple[int, int]:
        """
        Validate and normalize line numbers for code analysis.

        Args:
            start_line (int): The starting line number.
            end_line (int): The ending line number.
            total_lines (int): The total number of lines in the file.

        Returns:
            tuple[int, int]: A tuple of (start_line, end_line) after validation and normalization.

        Raises:
            ValueError: If the line numbers are invalid.
        """
        if start_line is None:
            start_line = 1
        if end_line is None:
            end_line = total_lines

        # Validate line numbers
        if start_line < 1:
            start_line = 1
        if end_line > total_lines:
            end_line = total_lines
        if start_line > end_line:
            raise ValueError("Start line must be less than or equal to end line.")

        return start_line, end_line

    def _extract_code_snippet(
        self, lines: list[str], start_line: int, end_line: int
    ) -> str:
        """
        Extract a code snippet from the given lines.

        Args:
            lines (list[str]): The lines of the file.
            start_line (int): The starting line number (1-based).
            end_line (int): The ending line number (1-based).

        Returns:
            str: The extracted code snippet.
        """
        return "".join(lines[start_line - 1 : end_line])

    def _parse_functions_from_content(
        self, content: str, function_name: str = None
    ) -> list[ast.FunctionDef]:
        """
        Parse functions from Python file content.

        Args:
            content (str): The file content.
            function_name (str, optional): Specific function name to find.

        Returns:
            list[ast.FunctionDef]: List of function nodes.

        Raises:
            ValueError: If no functions found or specific function not found.
        """
        tree = ast.parse(content)
        functions_to_test = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if function_name is None or node.name == function_name:
                    functions_to_test.append(node)

        if not functions_to_test:
            if function_name:
                raise ValueError(f"Function '{function_name}' not found in file")
            else:
                raise ValueError("No functions found in file")

        return functions_to_test

    def _parse_functions_from_file(
        self, file_path: str, function_name: str = None
    ) -> list[dict]:
        """
        Parse functions from Python file using AST and return structured information.

        Args:
            file_path (str): Path to the Python file.
            function_name (str, optional): Specific function name to find.

        Returns:
            list[dict]: List of function information dictionaries.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If no functions found or specific function not found.
        """
        content = self._read_file_content(file_path)
        func_nodes = self._parse_functions_from_content(content, function_name)

        functions = []
        lines = content.splitlines()

        for func_node in func_nodes:
            func_name = func_node.name
            args = [arg.arg for arg in func_node.args.args]

            # Try to infer return type from annotation
            return_type = None
            if func_node.returns:
                return_type = ast.unparse(func_node.returns)

            # Try to extract docstring
            docstring = ast.get_docstring(func_node)

            # Extract function source code
            func_start = func_node.lineno - 1
            func_end = func_node.end_lineno
            func_lines = lines[func_start:func_end]
            source = "\n".join(func_lines)

            functions.append(
                {
                    "function_name": func_name,
                    "args": args,
                    "return_type": return_type,
                    "docstring": docstring,
                    "source": source,
                }
            )

        return functions

    def _generate_test_template(self, func_node: ast.FunctionDef) -> dict:
        """
        Generate a test template for a function.

        Args:
            func_node (ast.FunctionDef): The function AST node.

        Returns:
            dict: Test template information.
        """
        func_name = func_node.name
        args = [arg.arg for arg in func_node.args.args]

        # Try to infer return type from annotation
        return_type = None
        if func_node.returns:
            return_type = ast.unparse(func_node.returns)

        # Try to extract docstring
        docstring = ast.get_docstring(func_node)

        # Generate a basic test template
        test_template = textwrap.dedent(f'''
            def test_{func_name}():
                """Test the {func_name} function."""
                # TODO: Add actual test implementation
                # Function signature: {func_name}({", ".join(args)})
                # Return type: {return_type or "Unknown"}
                # Docstring: {docstring or "None"}

                # Test with typical inputs
                # result = {func_name}(...)
                # assert result == expected_value

                # Test edge cases
                # ...

                # Test error conditions
                # ...

                pass
        ''').strip()

        return {
            "function_name": func_name,
            "test_code": test_template,
            "arguments": args,
            "return_type": return_type,
            "docstring": docstring,
        }

    def _generate_basic_test_templates(self, functions: list[dict]) -> list[dict]:
        """
        Generate basic test templates for a list of functions.

        Args:
            functions (list[dict]): List of function information dictionaries.

        Returns:
            list[dict]: List of test template dictionaries.
        """
        test_templates = []
        for func_info in functions:
            func_name = func_info["function_name"]
            args = func_info["args"]
            return_type = func_info["return_type"]
            docstring = func_info["docstring"]

            # Generate a basic test template
            test_template = textwrap.dedent(f'''
                def test_{func_name}():
                    """Test the {func_name} function."""
                    # TODO: Add actual test implementation
                    # Function signature: {func_name}({", ".join(args)})
                    # Return type: {return_type or "Unknown"}
                    # Docstring: {docstring or "None"}

                    # Test with typical inputs
                    # result = {func_name}(...)
                    # assert result == expected_value

                    # Test edge cases
                    # ...

                    # Test error conditions
                    # ...

                    pass
            ''').strip()

            test_templates.append(
                {
                    "function_name": func_name,
                    "test_code": test_template,
                    "arguments": args,
                    "return_type": return_type,
                    "docstring": docstring,
                }
            )

        return test_templates

    def _extract_function_source(self, file_path: str, function_name: str) -> str:
        """
        Extract the source code of a specific function from a file.

        Args:
            file_path (str): Path to the Python file.
            function_name (str): Name of the function to extract.

        Returns:
            str: The source code of the function.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the function is not found.
        """
        content = self._read_file_content(file_path)
        func_nodes = self._parse_functions_from_content(content, function_name)

        if not func_nodes:
            raise ValueError(f"Function '{function_name}' not found in {file_path}")

        func_node = func_nodes[0]  # Should be only one if function_name specified
        lines = content.splitlines()
        func_start = func_node.lineno - 1
        func_end = func_node.end_lineno
        func_lines = lines[func_start:func_end]
        return "\n".join(func_lines)

    def _prepare_unit_test_prompt(self, code_snippet: str, file_path: str) -> str:
        """
        Prepare the prompt for unit test generation using LLM.

        Args:
            code_snippet (str): The source code of the functions.
            file_path (str): Path to the file containing the functions.

        Returns:
            str: The prepared prompt for LLM analysis.
        """
        template = """
            Please generate comprehensive unit tests for the following Python function(s).
            Focus on:
            1. Testing typical use cases
            2. Testing edge cases and boundary conditions
            3. Testing error conditions and exceptions
            4. Following pytest conventions
            5. Including descriptive test names and docstrings

            Functions from file '{file_path}':
            ```python
            {code_snippet}
            ```

            Please provide the test code in a format ready to be added to a pytest test file.
        """
        return self._build_user_prompt(
            template, file_path=file_path, code_snippet=code_snippet
        )

    def _build_unit_test_system_prompt(self) -> str:
        """Build standardized system prompt for unit test generation."""
        return self._build_system_prompt(
            "You are an expert Python developer that generates comprehensive unit tests. "
            "Focus on pytest conventions, edge cases, error conditions, and best testing practices."
        )

    def _prepare_todo_fixme_prompt(
        self, comment: str, file_path: str, line_number: int, code_context: str
    ) -> str:
        """
        Prepare the prompt for TODO/FIXME resolution using LLM.

        Args:
            comment (str): The TODO/FIXME comment text.
            file_path (str): Path to the file containing the comment.
            line_number (int): Line number of the comment.
            code_context (str): Surrounding code context.

        Returns:
            str: The prepared prompt for LLM analysis.
        """
        template = """
            Analyze the following code snippet and the associated TODO/FIXME comment.
            Your task is to suggest a resolution for the TODO/FIXME. Provide the suggested
            code changes, an explanation of your approach, and any assumptions made.

            TODO/FIXME Comment: {comment}
            File: {file_path}
            Line: {line_number}

            Code Context:
            ```python
            {code_context}
            ```

            Please provide your response in the following JSON format:
            ```json
            {{
              "suggested_code": "<suggested code changes>",
              "explanation": "<explanation of the resolution>",
              "line_start": <start line of suggested change (inclusive)>,
              "line_end": <end line of suggested change (inclusive)>
            }}
            ```
            If no code changes are needed, leave "suggested_code" empty.
            If the TODO/FIXME is complex and requires multiple steps, outline them in the explanation.
        """
        return self._build_user_prompt(
            template,
            comment=comment,
            file_path=file_path,
            line_number=line_number,
            code_context=code_context,
        )

    def _build_todo_fixme_system_prompt(self) -> str:
        """Build standardized system prompt for TODO/FIXME resolution."""
        return self._build_system_prompt(
            "You are an expert Python developer that resolves TODO and FIXME comments. "
            "Provide clear, actionable solutions with proper code changes and explanations."
        )

    def _prepare_documentation_prompt(
        self, tool_name: str, signature: str, docstring: str, function_info: str
    ) -> str:
        """
        Prepare the prompt for tool documentation generation using LLM.

        Args:
            tool_name (str): Name of the tool to document.
            signature (str): Function signature.
            docstring (str): Existing docstring.
            function_info (str): Function implementation details.

        Returns:
            str: The prepared prompt for LLM analysis.
        """
        template = """
            Please generate comprehensive documentation for the following Python tool function.
            The documentation should follow this format:

            ### {{tool_name}}
            {{Brief description of what the tool does}}

            **Parameters:**
            {{List each parameter with its type and description}}
            {{For optional parameters, indicate the default value}}

            **Returns:**
            {{Description of what the function returns}}

            Example usage:
            ```json
            {{
              "name": "{{tool_name}}",
              "arguments": {{
                {{example arguments}}
              }}
            }}
            ```

            Function to document:
            Name: {tool_name}
            Signature: {signature}
            Docstring: {docstring}
            Function implementation: {function_info}

            Please provide the documentation in the exact format specified above.
        """
        return self._build_user_prompt(
            template,
            tool_name=tool_name,
            signature=signature,
            docstring=docstring,
            function_info=function_info,
        )

    def _build_documentation_system_prompt(self) -> str:
        """Build standardized system prompt for tool documentation generation."""
        return self._build_system_prompt(
            "You are a technical writer that creates comprehensive documentation for Python tools. "
            "Follow the exact format specified and provide clear, accurate information."
        )

    def _collect_unit_test_suggestions(self, prompt: str) -> list[dict]:
        """
        Collect unit test suggestions from multiple LLM providers.

        Args:
            prompt (str): The prompt for unit test generation.

        Returns:
            list[dict]: List of suggestions from LLM providers.
        """
        system_message = self._build_unit_test_system_prompt()
        return self._get_llm_suggestions(prompt, system_message)

    def profile_code_performance(
        self, file_path: str, function_name: str = None
    ) -> dict:
        """
        Perfila el rendimiento de una función específica o de todo el archivo.

        Args:
            file_path (str): Ruta al archivo Python a perfilar.
            function_name (str, optional): Nombre de la función específica a perfilar.
                Si es None, perfila todo el archivo.

        Returns:
            dict: Resultados del perfilado incluyendo tiempo de ejecución,
            llamadas a funciones,
                y cuellos de botella de rendimiento.

        Raises:
            FileNotFoundError: Si el archivo especificado no existe.
            ValueError: Si la función especificada no se encuentra en el archivo.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Read the file content
            content = safe_read_file(file_path)

            # Create a temporary file for profiling
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name

            try:
                # Create a profiler instance
                profiler = cProfile.Profile()

                if function_name:
                    # Profile a specific function
                    # First, we need to import the function
                    spec = importlib.util.spec_from_file_location(
                        "temp_module", temp_file_path
                    )
                    temp_module = importlib.util.module_from_spec(spec)

                    # We'll execute the file and then try to find the function
                    profiler.enable()
                    spec.loader.exec_module(temp_module)
                    profiler.disable()

                    # Check if the function exists
                    if hasattr(temp_module, function_name):
                        func = getattr(temp_module, function_name)
                        # Profile the function call
                        profiler.enable()
                        func()
                        profiler.disable()
                    else:
                        raise ValueError(
                            f"Function '{function_name}' not found in {file_path}"
                        )
                else:
                    # Profile the entire file
                    profiler.enable()
                    exec(content)
                    profiler.disable()

                # Create a stats object to analyze the profiling data
                stats_stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stats_stream)
                stats.sort_stats("cumulative")
                stats.print_stats(20)  # Print top 20 functions

                # Get the stats as a string
                stats_str = stats_stream.getvalue()

                # Parse the stats to extract key metrics
                lines = stats_str.split("\n")
                parsed_stats = []
                header_found = False

                for line in lines:
                    if line.strip().startswith("ncalls"):
                        header_found = True
                        continue
                    if header_found and line.strip():
                        # Parse each line of stats
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                ncalls = parts[0]
                                tottime = float(parts[1])
                                percall_tot = float(parts[2])
                                cumtime = float(parts[3])
                                percall_cum = float(parts[4])
                                filename_lineno = " ".join(parts[5:])

                                parsed_stats.append(
                                    {
                                        "ncalls": ncalls,
                                        "tottime": tottime,
                                        "percall_tot": percall_tot,
                                        "cumtime": cumtime,
                                        "percall_cum": percall_cum,
                                        "filename_lineno": filename_lineno,
                                    }
                                )
                            except (ValueError, IndexError):
                                # Skip lines that can't be parsed
                                continue

                return {
                    "message": f"Performance profiling completed for {file_path}"
                    + (f" function '{function_name}'" if function_name else ""),
                    "total_functions_profiled": len(parsed_stats),
                    "top_bottlenecks": parsed_stats[:10],  # Top 10 bottlenecks
                    "raw_stats": stats_str,
                }

            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        except Exception as e:
            error_response = self._handle_llm_error(e, "Performance profiling")
            return self._create_error_response(
                "PROFILING_ERROR", error_response["error"]["message"]
            )

    def _read_file_content(
        self, file_path: str, as_lines: bool = False
    ) -> str | list[str]:
        """
        Read file content and validate existence.

        Args:
            file_path (str): Path to the file to read.
            as_lines (bool): If True, return list of lines; otherwise, return as string.

        Returns:
            str | list[str]: File content as string or list of lines.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        return safe_read_file(file_path, as_lines)

    def _create_error_response(self, error_code: str, message: str) -> dict:
        """
        Create a standardized error response.

        Args:
            error_code (str): The error code.
            message (str): The error message.

        Returns:
            dict: Standardized error response.
        """
        return {
            "error": {
                "code": error_code,
                "message": message,
            }
        }

    def _handle_llm_error(self, exception: Exception, context: str = "") -> dict:
        """Standardized LLM error handling.

        Args:
            exception: The caught exception
            context: Additional context about where the error occurred

        Returns:
            dict: Standardized error response
        """
        return {
            "error": {
                "code": "LLM_ERROR",
                "message": f"{context}: {str(exception)}"
                if context
                else str(exception),
                "type": type(exception).__name__,
            }
        }

    def _build_system_prompt(
        self, role_description: str = "You are a helpful assistant."
    ) -> str:
        """Standardized system prompt builder.

        Args:
            role_description: Description of the assistant's role

        Returns:
            str: Formatted system prompt
        """
        return role_description

    def _build_user_prompt(self, template: str, **kwargs) -> str:
        """Standardized user prompt builder with dedenting.

        Args:
            template: Multi-line template string
            **kwargs: Variables to interpolate

        Returns:
            str: Formatted and dedented user prompt
        """
        return textwrap.dedent(template).strip().format(**kwargs)

    def _validate_file_and_lines(
        self, file_path: str, start_line: int, end_line: int
    ) -> tuple[list[str], int, int]:
        """
        Validate file existence and normalize line numbers.

        Args:
            file_path (str): Path to the file.
            start_line (int): Starting line number.
            end_line (int): Ending line number.

        Returns:
            tuple[list[str], int, int]: Tuple of (lines, normalized_start_line, normalized_end_line).

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        lines = self._read_file_content(file_path, as_lines=True)
        start_line, end_line = self._validate_and_normalize_line_numbers(
            start_line, end_line, len(lines)
        )
        return lines, start_line, end_line

    def _prepare_improvement_prompt(self, code_snippet: str, file_path: str) -> str:
        """
        Prepare the prompt for code improvement analysis, including file path context.

        Args:
            code_snippet (str): The code snippet to analyze.
            file_path (str): The file path for context.

        Returns:
            str: The prepared prompt for LLM analysis.
        """
        template = """
            Please analyze the following Python code snippet from file '{file_path}' and suggest
            improvements.
            Focus on:
            1. Code readability and maintainability
            2. Performance optimizations
            3. Python best practices
            4. Potential bugs or issues

            For each suggestion, provide:
            - A clear description of the issue
            - An explanation of why it's a problem
            - A specific recommendation for improvement
            - An example of the improved code if applicable

            Code snippet:
            ```python
            {code_snippet}
            ```

            Please format your response as a clear, structured list of suggestions.
        """
        return self._build_user_prompt(
            template, file_path=file_path, code_snippet=code_snippet
        )

    def _build_improvement_system_prompt(self) -> str:
        """Build standardized system prompt for code improvement analysis."""
        return self._build_system_prompt(
            "You are an expert Python developer that provides detailed code improvement suggestions. "
            "Focus on best practices, performance, maintainability, and potential bugs."
        )

    def _collect_improvement_suggestions(self, prompt: str) -> list[dict]:
        """
        Collect improvement suggestions from multiple LLM providers.

        Args:
            prompt (str): The prompt for improvement analysis.

        Returns:
            list[dict]: List of suggestions from LLM providers.
        """
        system_message = self._build_improvement_system_prompt()
        return self._get_llm_suggestions(prompt, system_message)

    def suggest_code_improvements(
        self, file_path: str, start_line: int = None, end_line: int = None
    ) -> dict:
        """
                Analiza una sección de código y sugiere mejoras consultando LLMs externos.

                Args:
                    file_path (str): Ruta al archivo a analizar.
                    start_line (int, optional): Número de línea inicial de la sección a
                    analizar.
                        Si es None, analiza desde el principio del archivo.
                    end_line (int, optional): Número de línea final de la sección a analizar.
                        Si es None, analiza hasta el final del archivo.

                Returns:
                    dict: Resultados del análisis con sugerencias para mejoras, o un
        mensaje de error.

                Raises:
                    FileNotFoundError: Si el archivo especificado no existe.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Validate file and normalize lines
            lines, start_line, end_line = self._validate_file_and_lines(
                file_path, start_line, end_line
            )

            # Extract the code snippet
            code_snippet = self._extract_code_snippet(lines, start_line, end_line)

            # Prepare the prompt for LLM analysis
            prompt = self._prepare_improvement_prompt(code_snippet, file_path)

            # Collect improvement suggestions
            suggestions = self._collect_improvement_suggestions(prompt)

            return {
                "message": f"Code improvements analysis completed for {file_path}"
                + (
                    f" (lines {start_line}-{end_line})"
                    if start_line != 1 or end_line != len(lines)
                    else ""
                ),
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "suggestions": suggestions,
            }

        except Exception as e:
            error_response = self._handle_llm_error(e, "Code improvement analysis")
            return self._create_error_response(
                "ANALYSIS_ERROR", error_response["error"]["message"]
            )

    def generate_unit_tests(self, file_path: str, function_name: str = None) -> dict:
        """
        Genera tests unitarios para funciones en un archivo Python.

        Args:
            file_path (str): Ruta al archivo Python a analizar.
            function_name (str, optional): Nombre de una función específica para generar tests.
                Si es None, genera tests para todas las funciones en el archivo.

        Returns:
            dict: Código de test generado y metadatos, o un mensaje de error.

        Raises:
            FileNotFoundError: Si el archivo especificado no existe.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Parse functions from the file
            functions = self._parse_functions_from_file(file_path, function_name)

            # Generate basic test templates
            generated_tests = self._generate_basic_test_templates(functions)

            # Try to get suggestions from LLM providers if available
            llm_suggestions = []
            if functions:
                # Combine function sources for LLM analysis
                function_sources = [func["source"] for func in functions]
                code_snippet = "\n\n".join(function_sources)

                # Prepare the prompt for LLM test generation
                prompt = self._prepare_unit_test_prompt(code_snippet, file_path)

                # Collect suggestions from LLM providers
                llm_suggestions = self._collect_unit_test_suggestions(prompt)

            return {
                "message": f"Unit test generation completed for {file_path}"
                + (f" function '{function_name}'" if function_name else ""),
                "file_path": file_path,
                "function_name": function_name,
                "generated_tests": generated_tests,
                "llm_suggestions": llm_suggestions,
            }

        except Exception as e:
            error_response = self._handle_llm_error(e, "Unit test generation")
            return self._create_error_response(
                "TEST_GENERATION_ERROR",
                error_response["error"]["message"],
            )

    def _extract_function_info(self, tool_name: str, tool_function) -> dict:
        """
        Extrae información sobre una función de herramienta.

        Args:
            tool_name (str): Nombre de la herramienta.
            tool_function: El objeto de función de la herramienta.

        Returns:
            dict: Información sobre la función.
        """
        try:
            # Try to get the source code
            source = inspect.getsource(tool_function)
            filename = inspect.getfile(tool_function)
            lineno = inspect.getsourcelines(tool_function)[1]

            return {"source": source, "filename": filename, "line_number": lineno}
        except Exception as e:
            return {"error": f"Failed to extract function info: {str(e)}"}

    def auto_document_tool(self, tool_name: str = None, main_module=None) -> dict:
        """
        Genera documentación automáticamente para herramientas que carecen de documentación detallada.

        Args:
            tool_name (str, optional): Nombre de una herramienta específica a documentar.
                Si es None, documenta todas las herramientas que carecen de documentación detallada.
            main_module: Referencia al módulo principal (main.py) para acceder a TOOL_FUNCTIONS.

        Returns:
            dict: Documentación generada y metadatos, o un mensaje de error.
        """
        if not main_module:
            return {
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Main module reference is required for auto_document_tool.",
                }
            }

        try:
            # Get the list of all registered tools
            TOOL_FUNCTIONS = getattr(main_module, "TOOL_FUNCTIONS", {})
            registered_tools = list(TOOL_FUNCTIONS.keys())

            # If a specific tool is requested, check if it exists
            if tool_name and tool_name not in registered_tools:
                return {
                    "error": {
                        "code": "TOOL_NOT_FOUND",
                        "message": f"Tool '{tool_name}' not found in registered tools",
                    }
                }

            # Determine which tools to document
            tools_to_document = [tool_name] if tool_name else registered_tools

            # For each tool, generate documentation
            generated_docs = []

            for tool in tools_to_document:
                # Get the tool function
                tool_function = TOOL_FUNCTIONS.get(tool)
                if not tool_function:
                    continue

                # Extract function signature and docstring
                sig = inspect.signature(tool_function)
                docstring = inspect.getdoc(tool_function) or "No docstring available"

                # Prepare code snippet for LLM analysis
                # We'll need to find the actual function definition in the source files
                function_info = self._extract_function_info(tool, tool_function)

                # Generate documentation using LLMs
                # Prepare the prompt for LLM documentation generation
                prompt = self._prepare_documentation_prompt(
                    tool,
                    sig,
                    docstring,
                    function_info.get("source", "Source not available"),
                )

                # Get suggestions from multiple LLM providers using the common method
                system_message = self._build_documentation_system_prompt()
                llm_suggestions = self._get_llm_suggestions(prompt, system_message)

                generated_docs.append(
                    {
                        "tool_name": tool,
                        "signature": str(sig),
                        "docstring": docstring,
                        "function_info": function_info,
                        "llm_suggestions": llm_suggestions,
                    }
                )

            return {
                "message": "Auto documentation generation completed"
                + (f" for tool '{tool_name}'" if tool_name else " for all tools"),
                "tools_documented": len(generated_docs),
                "generated_docs": generated_docs,
            }

        except Exception as e:
            error_response = self._handle_llm_error(e, "Auto documentation generation")
            return self._create_error_response(
                "DOCUMENTATION_ERROR", error_response["error"]["message"]
            )

    def generate_llm_api_wrapper(
        self, llm_provider: str, model_name: str, api_key_env_var: str = None
    ) -> str:
        """
        Generates Python wrapper code for interacting with various LLM APIs.

        Args:
            llm_provider (str): The LLM provider (e.g., 'groq', 'openrouter', 'google').
            model_name (str): The specific model name to use (e.g., 'llama3-8b-8192', 'gemini-pro').
            api_key_env_var (str, optional): The name of the environment variable that stores the API key.
                If None, a default will be used based on the provider.

        Returns:
            str: The generated Python code for the LLM API wrapper.

        Raises:
            ValueError: If an unsupported LLM provider is specified.
        """
        provider_lower = llm_provider.lower()

        # Determine default API key environment variable if not provided
        if api_key_env_var is None:
            if provider_lower == "groq":
                api_key_env_var = "GROQ_API_KEY"
            elif provider_lower == "openrouter":
                api_key_env_var = "OPENROUTER_API_KEY"
            elif provider_lower == "google":
                api_key_env_var = "GOOGLE_API_KEY"
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        # Provider-specific configurations
        provider_configs = {
            "groq": {
                "llm_client_import": "from groq import Groq",
                "class_name": "Groq",
                "client_initialization": "self.client = Groq(api_key=self.api_key)",
                "generate_content_logic": [
                    "            chat_completion = self.client.chat.completions.create(",
                    '                messages=[{"role": "user", "content": prompt}],',
                    "                model=self.model_name,",
                    "                **kwargs",
                    "            )",
                    "            return chat_completion.choices[0].message.content",
                ],
            },
            "openrouter": {
                "llm_client_import": "from openai import OpenAI",
                "class_name": "OpenRouter",
                "client_initialization": [
                    "self.client = OpenAI(",
                    '    base_url="https://openrouter.ai/api/v1",',
                    "    api_key=self.api_key,",
                    ")",
                ],
                "generate_content_logic": [
                    "            chat_completion = self.client.chat.completions.create(",
                    "                extra_headers={",
                    '                    "HTTP-Referer": "http://localhost:8000",  # Replace with your actual site URL',
                    '                    "X-Title": "CodeSage MCP Server",  # Replace with your actual app name',
                    "                },",
                    '                messages=[{"role": "user", "content": prompt}],',
                    "                model=self.model_name,",
                    "                **kwargs",
                    "            )",
                    "            return chat_completion.choices[0].message.content",
                ],
            },
            "google": {
                "llm_client_import": "import google.generativeai as genai",
                "class_name": "Google",
                "client_initialization": [
                    "genai.configure(api_key=self.api_key)",
                    "self.client = genai.GenerativeModel(self.model_name)",
                ],
                "generate_content_logic": [
                    "            response = self.client.generate_content(prompt, **kwargs)",
                    "            return response.text",
                ],
            },
        }

        if provider_lower not in provider_configs:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        config = provider_configs[provider_lower]

        # Build the code using lists to ensure proper formatting
        lines = [
            "import os",
            "from typing import Any, Dict, Optional",
            "",
            "# Import specific LLM client libraries",
            config["llm_client_import"],
            "",
            f"class {config['class_name']}LLMClient:",
            '    def __init__(self, model_name: str = "'
            + model_name
            + '", api_key: Optional[str] = None):',
            "        self.model_name = model_name",
            f'        self.api_key = api_key if api_key else os.getenv("{api_key_env_var}")',
            "        if not self.api_key:",
            f'            raise ValueError(f"API key not found for {llm_provider}. Please set the {api_key_env_var} environment variable.")',
        ]

        # Add client initialization
        if isinstance(config["client_initialization"], list):
            lines.extend(
                [f"        {line}" for line in config["client_initialization"]]
            )
        else:
            lines.append(f"        {config['client_initialization']}")

        # Add generate_content method
        lines.extend(
            [
                "",
                "    def generate_content(self, prompt: str, **kwargs) -> str:",
                f"        '''Generates content using the {llm_provider} LLM.'''",
                "        try:",
            ]
        )

        # Add generate content logic
        lines.extend(config["generate_content_logic"])

        # Add exception handling
        lines.extend(
            [
                "        except Exception as e:",
                f'            return f"Error generating content with {llm_provider} LLM: {{e}}"',
            ]
        )

        return "\n".join(lines)

    def parse_llm_response(self, llm_response_content: str) -> dict:
        """
        Parses the content of an LLM response, extracting and validating JSON data.

        This method is designed to robustly handle various LLM output formats, including
        responses wrapped in markdown code blocks, and attempts to parse them as JSON.

        Args:
            llm_response_content (str): The raw content string received from an LLM.

        Returns:
            dict: The parsed JSON data.

        Raises:
            ValueError: If the content cannot be parsed as valid JSON.
        """
        # Extract JSON from markdown fences using regex for robustness
        match = re.search(r"```json\s*\n(.*)\n\s*```", llm_response_content, re.DOTALL)
        if match:
            json_string = match.group(1).strip()
        else:
            json_string = llm_response_content.strip()

        try:
            parsed_data = json.loads(json_string)
            return parsed_data
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {e}. Raw content: {llm_response_content}"
            )

    def resolve_todo_fixme(self, file_path: str, line_number: int = None) -> dict:
        """
        Analyzes a TODO/FIXME comment and suggests potential resolutions using LLMs.

        Args:
            file_path (str): The absolute path to the file containing the TODO/FIXME comment.
            line_number (int, optional): The specific line number of the TODO/FIXME comment.
                If None, the tool will attempt to find and resolve the first TODO/FIXME comment in the file.

        Returns:
            dict: Suggested resolutions from LLMs, or an error message.

        Raises:
            FileNotFoundError: If the file specified does not exist.
            ValueError: If no TODO/FIXME comments are found or if the specified line number is invalid.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            lines = safe_read_file(file_path, as_lines=True)

            todo_fixme_comments = _count_todo_fixme_comments(lines)

            if not todo_fixme_comments:
                raise ValueError(f"No TODO/FIXME comments found in {file_path}")

            target_comment = None
            if line_number:
                for comment in todo_fixme_comments:
                    if comment["line_number"] == line_number:
                        target_comment = comment
                        break
                if not target_comment:
                    raise ValueError(
                        f"No TODO/FIXME comment found at line {line_number} in {file_path}"
                    )
            else:
                target_comment = todo_fixme_comments[
                    0
                ]  # Take the first one if no line_number specified

            # Extract code context (e.g., 10 lines before and after the comment)
            context_start = max(0, target_comment["line_number"] - 1 - 10)
            context_end = min(len(lines), target_comment["line_number"] - 1 + 10)
            code_context = "".join(lines[context_start:context_end])

            # Prepare the prompt for LLM resolution
            prompt = self._prepare_todo_fixme_prompt(
                target_comment["comment"],
                file_path,
                target_comment["line_number"],
                code_context,
            )

            # Use a powerful LLM for resolution
            llm_model = "google/gemini-pro"  # Prioritize Gemini for complex tasks
            system_message = self._build_todo_fixme_system_prompt()
            response_content, error_message = self._get_llm_response(
                prompt, llm_model, system_message
            )

            if error_message:
                raise Exception(f"LLM resolution failed: {error_message}")

            try:
                llm_response = self.parse_llm_response(response_content)
                return {
                    "message": "TODO/FIXME resolution suggested successfully.",
                    "resolution": llm_response,
                    "original_comment": target_comment,
                }
            except ValueError as e:
                raise ValueError(
                    f"LLM returned invalid JSON: {response_content}. Error: {e}"
                )

        except Exception as e:
            error_response = self._handle_llm_error(e, "TODO/FIXME resolution")
            return self._create_error_response(
                "TODO_FIXME_RESOLUTION_ERROR", error_response["error"]["message"]
            )

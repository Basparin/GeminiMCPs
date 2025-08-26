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
from codesage_mcp.utils import _count_todo_fixme_comments  # New import
import json  # New import


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

    def _summarize_with_groq(self, code_snippet: str, llm_model: str) -> str:
        """
        Resume un fragmento de código usando la API de Groq.

        Args:
            code_snippet (str): El fragmento de código a resumir.
            llm_model (str): El modelo LLM de Groq a usar para el resumen.

        Returns:
            str: El resumen del fragmento de código o un mensaje de error si falla.
        """
        if not self.groq_client:
            return "Error: GROQ_API_KEY not configured."
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes code.",
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize the following code snippet:\n\n"
                        f"```\n{code_snippet}```",
                    },
                ],
                model=llm_model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error during summarization: {e}"

    def _summarize_with_openrouter(self, code_snippet: str, llm_model: str) -> str:
        """
        Resume un fragmento de código usando la API de OpenRouter.

        Args:
            code_snippet (str): El fragmento de código a resumir.
            llm_model (str): El modelo LLM de OpenRouter a usar para el resumen.

        Returns:
            str: El resumen del fragmento de código o un mensaje de error si falla.
        """
        if not self.openrouter_client:
            return "Error: OPENROUTER_API_KEY not configured."
        try:
            chat_completion = self.openrouter_client.chat.completions.create(
                model=llm_model.replace("openrouter/", "", 1),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes code.",
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize the following code snippet:\n\n"
                        f"```\n{code_snippet}```",
                    },
                ],
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error during summarization: {e}"

    def _summarize_with_google_ai(self, code_snippet: str, llm_model: str) -> str:
        """
        Resume un fragmento de código usando la API de Google AI.

        Args:
            code_snippet (str): El fragmento de código a resumir.
            llm_model (str): El modelo LLM de Google AI a usar para el resumen.

        Returns:
            str: El resumen del fragmento de código o un mensaje de error si falla.
        """
        if not self.google_ai_client:
            return "Error: GOOGLE_API_KEY not configured."
        try:
            model = self.google_ai_client.GenerativeModel(
                llm_model.replace("google/", "", 1)
            )
            response = model.generate_content(
                "Please summarize the following code snippet:\n\n"
                f"```\n{code_snippet}```"
            )
            return response.text
        except Exception as e:
            return f"Error during summarization: {e}"

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

        lines = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                if start_line <= i <= end_line:
                    lines.append(line)

        if not lines:
            return "No code found in the specified line range."

        code_snippet = "".join(lines)

        if llm_model.startswith("openrouter/"):
            return self._summarize_with_openrouter(code_snippet, llm_model)
        elif llm_model.startswith("llama3") or llm_model.startswith("mixtral"):
            return self._summarize_with_groq(code_snippet, llm_model)
        elif llm_model.startswith("google/"):
            return self._summarize_with_google_ai(code_snippet, llm_model)
        else:
            return f"LLM model '{llm_model}' not supported yet."

    def _get_llm_response(self, prompt: str, llm_model: str) -> tuple[str, str]:
        """Helper to get LLM response from various providers."""
        if llm_model.startswith("openrouter/"):
            if not self.openrouter_client:
                return None, "Error: OPENROUTER_API_KEY not configured."
            try:
                chat_completion = self.openrouter_client.chat.completions.create(
                    model=llm_model.replace("openrouter/", "", 1),
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )
                return chat_completion.choices[0].message.content, None
            except Exception as e:
                return None, f"Error during LLM call: {e}"
        elif llm_model.startswith("llama3") or llm_model.startswith("mixtral"):
            if not self.groq_client:
                return None, "Error: GROQ_API_KEY not configured."
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    model=llm_model,
                )
                return chat_completion.choices[0].message.content, None
            except Exception as e:
                return None, f"Error during LLM call: {e}"
        elif llm_model.startswith("google/"):
            if not self.google_ai_client:
                return None, "Error: GOOGLE_API_KEY not configured."
            try:
                response = self.google_ai_client.generate_content(prompt)
                return response.text, None
            except Exception as e:
                return None, f"Error during LLM call: {e}"
        else:
            return None, f"LLM model '{llm_model}' not supported yet."

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
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

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
            return {
                "error": {
                    "code": "PROFILING_ERROR",
                    "message": (
                        f"An error occurred during performance profiling: {str(e)}"
                    ),
                }
            }

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
            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Determine the range of lines to analyze
            if start_line is None:
                start_line = 1
            if end_line is None:
                end_line = len(lines)

            # Validate line numbers
            if start_line < 1:
                start_line = 1
            if end_line > len(lines):
                end_line = len(lines)
            if start_line > end_line:
                return {
                    "error": {
                        "code": "INVALID_INPUT",
                        "message": "Start line must be less than or equal to end line.",
                    }
                }

            # Extract the code snippet
            code_snippet = "".join(lines[start_line - 1 : end_line])

            # Prepare the prompt for LLM analysis
            prompt = textwrap.dedent(f"""
                Please analyze the following Python code snippet and suggest
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
            """).strip()

            # Try to get suggestions from different LLM providers
            suggestions = []

            # Try Groq first
            if self.groq_client:
                try:
                    chat_completion = self.groq_client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that suggests code improvements.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        model="llama3-8b-8192",
                        temperature=0.1,
                        max_tokens=2048,
                    )
                    groq_suggestion = chat_completion.choices[0].message.content
                    suggestions.append(
                        {"provider": "Groq (Llama3)", "suggestions": groq_suggestion}
                    )
                except Exception as e:
                    suggestions.append(
                        {
                            "provider": "Groq (Llama3)",
                            "error": f"Failed to get suggestions from Groq: {str(e)}",
                        }
                    )

            # Try OpenRouter
            if self.openrouter_client:
                try:
                    chat_completion = self.openrouter_client.chat.completions.create(
                        model="openrouter/google/gemini-pro",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that suggests code improvements.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=2048,
                    )
                    openrouter_suggestion = chat_completion.choices[0].message.content
                    suggestions.append(
                        {
                            "provider": "OpenRouter (Gemini)",
                            "suggestions": openrouter_suggestion,
                        }
                    )
                except Exception as e:
                    suggestions.append(
                        {
                            "provider": "OpenRouter (Gemini)",
                            "error": f"Failed to get suggestions from OpenRouter: {str(e)}",
                        }
                    )

            # Try Google AI
            if self.google_ai_client:
                try:
                    model = self.google_ai_client.GenerativeModel("gemini-pro")
                    response = model.generate_content(
                        prompt,
                        generation_config=self.google_ai_client.types.GenerationConfig(
                            temperature=0.1, max_output_tokens=2048
                        ),
                    )
                    google_suggestion = response.text
                    suggestions.append(
                        {
                            "provider": "Google AI (Gemini)",
                            "suggestions": google_suggestion,
                        }
                    )
                except Exception as e:
                    suggestions.append(
                        {
                            "provider": "Google AI (Gemini)",
                            "error": f"Failed to get suggestions from Google AI: {str(e)}",
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
            return {
                "error": {
                    "code": "ANALYSIS_ERROR",
                    "message": f"An error occurred during code analysis: {str(e)}",
                }
            }

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
            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the file to find functions
            tree = ast.parse(content)

            # Find all functions or the specific function
            functions_to_test = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if function_name is None or node.name == function_name:
                        functions_to_test.append(node)

            if not functions_to_test:
                if function_name:
                    raise ValueError(
                        f"Function '{function_name}' not found in {file_path}"
                    )
                else:
                    raise ValueError(f"No functions found in {file_path}")

            # Generate tests for each function
            generated_tests = []

            for func_node in functions_to_test:
                # Extract function name and arguments
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

                generated_tests.append(
                    {
                        "function_name": func_name,
                        "test_code": test_template,
                        "arguments": args,
                        "return_type": return_type,
                        "docstring": docstring,
                    }
                )

            # Try to get suggestions from LLM providers if available
            llm_suggestions = []

            # Prepare code snippet for LLM analysis
            if functions_to_test:
                # Get the full function definitions
                function_definitions = []
                for func_node in functions_to_test:
                    # Extract the function source code
                    func_start = func_node.lineno - 1
                    func_end = func_node.end_lineno
                    func_lines = content.splitlines()[func_start:func_end]
                    function_definitions.append("\n".join(func_lines))

                code_snippet = "\n\n".join(function_definitions)

                # Prepare the prompt for LLM test generation
                prompt = textwrap.dedent(f"""
                    Please generate comprehensive unit tests for the following Python function(s).
                    Focus on:
                    1. Testing typical use cases
                    2. Testing edge cases and boundary conditions
                    3. Testing error conditions and exceptions
                    4. Following pytest conventions
                    5. Including descriptive test names and docstrings

                    Functions to test:
                    ```python
                    {code_snippet}
                    ```

                    Please provide the test code in a format ready to be added to a pytest test file.
                """).strip()

                response_content, error_message = self._get_llm_response(
                    prompt, "llama3-8b-8192"
                )
                if response_content:
                    llm_suggestions.append(
                        {"provider": "Groq (Llama3)", "suggestions": response_content}
                    )
                elif error_message:
                    llm_suggestions.append(
                        {"provider": "Groq (Llama3)", "error": error_message}
                    )

                response_content, error_message = self._get_llm_response(
                    prompt, "openrouter/google/gemini-pro"
                )
                if response_content:
                    llm_suggestions.append(
                        {
                            "provider": "OpenRouter (Gemini)",
                            "suggestions": response_content,
                        }
                    )
                elif error_message:
                    llm_suggestions.append(
                        {"provider": "OpenRouter (Gemini)", "error": error_message}
                    )

                response_content, error_message = self._get_llm_response(
                    prompt, "google/gemini-pro"
                )
                if response_content:
                    llm_suggestions.append(
                        {
                            "provider": "Google AI (Gemini)",
                            "suggestions": response_content,
                        }
                    )
                elif error_message:
                    llm_suggestions.append(
                        {"provider": "Google AI (Gemini)", "error": error_message}
                    )

            return {
                "message": f"Unit test generation completed for {file_path}"
                + (f" function '{function_name}'" if function_name else ""),
                "file_path": file_path,
                "function_name": function_name,
                "generated_tests": generated_tests,
                "llm_suggestions": llm_suggestions,
            }

        except Exception as e:
            return {
                "error": {
                    "code": "TEST_GENERATION_ERROR",
                    "message": f"An error occurred during test generation: {str(e)}",
                }
            }

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
                llm_suggestions = []

                # Prepare the prompt for LLM documentation generation
                prompt = textwrap.dedent(f"""
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
                    Name: {tool}
                    Signature: {sig}
                    Docstring: {docstring}
                    Function implementation: {function_info.get("source", "Source not available")}

                    Please provide the documentation in the exact format specified above.
                """).strip()

                # Try Groq first
                if self.groq_client:
                    try:
                        chat_completion = self.groq_client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant that generates documentation for Python tools in a specific format.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            model="llama3-8b-8192",
                            temperature=0.1,
                            max_tokens=2048,
                        )
                        groq_suggestion = chat_completion.choices[0].message.content
                        llm_suggestions.append(
                            {
                                "provider": "Groq (Llama3)",
                                "suggestions": groq_suggestion,
                            }
                        )
                    except Exception as e:
                        llm_suggestions.append(
                            {
                                "provider": "Groq (Llama3)",
                                "error": f"Failed to get suggestions from Groq: {str(e)}",
                            }
                        )

                # Try OpenRouter
                if self.openrouter_client:
                    try:
                        chat_completion = self.openrouter_client.chat.completions.create(
                            model="openrouter/google/gemini-pro",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant that generates documentation for Python tools in a specific format.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.1,
                            max_tokens=2048,
                        )
                        openrouter_suggestion = chat_completion.choices[
                            0
                        ].message.content
                        llm_suggestions.append(
                            {
                                "provider": "OpenRouter (Gemini)",
                                "suggestions": openrouter_suggestion,
                            }
                        )
                    except Exception as e:
                        llm_suggestions.append(
                            {
                                "provider": "OpenRouter (Gemini)",
                                "error": f"Failed to get suggestions from OpenRouter: {str(e)}",
                            }
                        )

                # Try Google AI
                if self.google_ai_client:
                    try:
                        model = self.google_ai_client.GenerativeModel("gemini-pro")
                        response = model.generate_content(
                            prompt,
                            generation_config=self.google_ai_client.types.GenerationConfig(
                                temperature=0.1, max_output_tokens=2048
                            ),
                        )
                        google_suggestion = response.text
                        llm_suggestions.append(
                            {
                                "provider": "Google AI (Gemini)",
                                "suggestions": google_suggestion,
                            }
                        )
                    except Exception as e:
                        llm_suggestions.append(
                            {
                                "provider": "Google AI (Gemini)",
                                "error": f"Failed to get suggestions from Google AI: {str(e)}",
                            }
                        )

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
            return {
                "error": {
                    "code": "DOCUMENTATION_ERROR",
                    "message": f"An error occurred during documentation generation: {str(e)}",
                }
            }

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

        # Template for the wrapper class
        wrapper_template = textwrap.dedent("""
            import os
            from typing import Any, Dict, Optional

            # Import specific LLM client libraries
            {llm_client_import}

            class {class_name}LLMClient:
                def __init__(self, model_name: str = "{model_name}", api_key: Optional[str] = None):
                    self.model_name = model_name
                    self.api_key = api_key if api_key else os.getenv("{api_key_env_var}")
                    if not self.api_key:
                        raise ValueError(f"API key not found for {llm_provider}. Please set the {api_key_env_var} environment variable.")
                    {client_initialization}

                def generate_content(self, prompt: str, **kwargs) -> str:
                    ""
                    Generates content using the {llm_provider} LLM.
                    ""
                    try:
                        {generate_content_logic}
                    except Exception as e:
                        return f"Error generating content with {llm_provider} LLM: {{e}}"

        """).strip()

        llm_client_import = ""
        class_name = ""
        client_initialization = ""
        generate_content_logic = ""

        if provider_lower == "groq":
            llm_client_import = "from groq import Groq"
            class_name = "Groq"
            client_initialization = "self.client = Groq(api_key=self.api_key)"
            generate_content_logic = textwrap.dedent("""
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    **kwargs
                )
                return chat_completion.choices[0].message.content
            """).strip()
        elif provider_lower == "openrouter":
            llm_client_import = "from openai import OpenAI"
            class_name = "OpenRouter"
            client_initialization = textwrap.dedent("""
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key,
                )
            """).strip()
            generate_content_logic = textwrap.dedent("""
                chat_completion = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "http://localhost:8000", # Replace with your actual site URL
                        "X-Title": "CodeSage MCP Server", # Replace with your actual app name
                    },
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    **kwargs
                )
                return chat_completion.choices[0].message.content
            """).strip()
        elif provider_lower == "google":
            llm_client_import = "import google.generativeai as genai"
            class_name = "Google"
            client_initialization = textwrap.dedent("""
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_name)
            """).strip()
            generate_content_logic = textwrap.dedent("""
                response = self.client.generate_content(prompt, **kwargs)
                return response.text
            """).strip()
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        return wrapper_template.format(
            llm_client_import=llm_client_import,
            class_name=class_name,
            model_name=model_name,
            api_key_env_var=api_key_env_var,
            llm_provider=llm_provider,
            client_initialization=client_initialization,
            generate_content_logic=generate_content_logic,
        )

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
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

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
            prompt = textwrap.dedent(f"""
                Analyze the following code snippet and the associated TODO/FIXME comment.
                Your task is to suggest a resolution for the TODO/FIXME. Provide the suggested
                code changes, an explanation of your approach, and any assumptions made.

                TODO/FIXME Comment: {target_comment["comment"]}
                File: {file_path}
                Line: {target_comment["line_number"]}

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
            """).strip()

            # Use a powerful LLM for resolution
            llm_model = "google/gemini-pro"  # Prioritize Gemini for complex tasks
            response_content, error_message = self._get_llm_response(prompt, llm_model)

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
            raise Exception(f"An error occurred during TODO/FIXME resolution: {str(e)}")

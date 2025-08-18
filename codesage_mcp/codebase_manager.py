import os
import re
import json
import fnmatch
from pathlib import Path
from groq import Groq
from openai import OpenAI
import google.generativeai as genai
import ast # Added this line
from .config import GROQ_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY


class CodebaseManager:
    def __init__(self):
        self.index_dir = Path(".codesage")
        self.index_file = self.index_dir / "codebase_index.json"
        self.indexed_codebases = {}
        self._initialize_index()
        self.groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self.openrouter_client = (
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            if OPENROUTER_API_KEY
            else None
        )
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.google_ai_client = genai
        else:
            self.google_ai_client = None

    def _initialize_index(self):
        """Initializes the index by creating the directory and loading existing data."""
        self.index_dir.mkdir(exist_ok=True)
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    self.indexed_codebases = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(
                    f"Warning: Could not load codebase index. "
                    f"A new one will be created. Error: {e}"
                )
                self.indexed_codebases = {}

    def _save_index(self):
        """Saves the current index to the file."""
        with open(self.index_file, "w") as f:
            json.dump(self.indexed_codebases, f, indent=4)

    def _get_gitignore_patterns(self, path: Path) -> list[str]:
        """Finds and parses the .gitignore file in the given path."""
        gitignore_file = path / ".gitignore"
        patterns = []
        if gitignore_file.exists():
            with open(gitignore_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        return patterns

    def _is_ignored(
        self,
        file_path: Path,
        gitignore_patterns: list[str],
        root_path: Path,
    ) -> bool:
        """Checks if a file or directory should be ignored based on
        .gitignore patterns."""
        relative_path = file_path.relative_to(root_path)
        
        # Convert Path objects to strings for fnmatch
        relative_path_str = str(relative_path)
        file_name_str = file_path.name

        print(f"\n--- Checking {file_path} (relative: {relative_path_str}) ---")
        for pattern in gitignore_patterns:
            print(f"  Pattern: {pattern}")
            # Handle patterns ending with a slash (directories)
            if pattern.endswith('/'):
                # Check if the relative path starts with the directory pattern
                if relative_path_str.startswith(pattern):
                    print(f"    Match (startswith dir): {pattern} -> True")
                    return True
                # Check if any part of the relative path matches the directory name
                # e.g., "venv/" should ignore "foo/venv/bar"
                pattern_dir_name = pattern.rstrip('/')
                if pattern_dir_name in relative_path.parts:
                    print(f"    Match (part of path dir): {pattern} -> True")
                    return True
            
            # Handle general glob patterns
            # Check against the full relative path
            if fnmatch.fnmatch(relative_path_str, pattern):
                print(f"    Match (fnmatch full path): {pattern} -> True")
                return True
            
            # Check against the file/directory name only
            if fnmatch.fnmatch(file_name_str, pattern):
                print(f"    Match (fnmatch file name): {pattern} -> True")
                return True
            
            # Handle patterns like "foo" which should ignore "foo/bar" and "foo"
            # This is implicitly handled by checking against relative_path_str
            # but let's be explicit for clarity if the pattern is a directory name
            if not pattern.endswith('/') and '/' not in pattern: # It's a simple name pattern
                if pattern in relative_path.parts:
                    print(f"    Match (simple name in parts): {pattern} -> True")
                    return True

        print(f"--- No match for {file_path} -> False ---")
        return False

    def read_code_file(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r") as f:
            return f.read()

    def index_codebase(self, path: str) -> list[str]:
        root_path = Path(path)
        if not root_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        gitignore_patterns = self._get_gitignore_patterns(root_path)
        # Explicitly add .git/ and venv/ to ignore patterns
        gitignore_patterns.append(".git/")
        gitignore_patterns.append("venv/")
        gitignore_patterns.append(self.index_dir.name + "/")
        gitignore_patterns.append(".gitignore")

        indexed_files = []
        for root, dirs, files in os.walk(path, topdown=True):
            current_path = Path(root)
            
            # Debugging: Print directories being considered
            print(f"DEBUG: Current directory: {current_path}")
            print(f"DEBUG: Dirs before filtering: {dirs}")

            # Filter directories in-place to prevent os.walk from descending into ignored ones
            dirs[:] = [
                d
                for d in dirs
                if not self._is_ignored(
                    current_path / d, gitignore_patterns, root_path
                )
            ]
            # Debugging: Print directories after filtering
            print(f"DEBUG: Dirs after filtering: {dirs}")

            for file in files:
                file_path = current_path / file
                # Debugging: Print files being considered
                # print(f"DEBUG: Checking file: {file_path}")
                if not self._is_ignored(file_path, gitignore_patterns, root_path):
                    indexed_files.append(str(file_path.relative_to(root_path)))

        abs_path_key = str(root_path.resolve())
        self.indexed_codebases[abs_path_key] = {
            "status": "indexed",
            "files": indexed_files,
        }
        self._save_index()

        print(f"Indexed {len(indexed_files)} files in codebase at: {path}")
        print(f"DEBUG: Indexed files: {indexed_files}") # Added logging
        return indexed_files

    def search_codebase(
        self,
        codebase_path: str,
        pattern: str,
        file_types: list[str] = None,
        exclude_patterns: list[str] = None,
    ) -> list[dict]:
        abs_codebase_path = str(Path(codebase_path).resolve())
        if abs_codebase_path not in self.indexed_codebases:
            raise ValueError(
                f"Codebase at {codebase_path} has not been indexed. "
                f"Please index it first."
            )

        indexed_files = self.indexed_codebases[abs_codebase_path]["files"]
        search_results = []

        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {pattern}. Error: {e}")

        for relative_file_path in indexed_files:
            file_path = Path(codebase_path) / relative_file_path

            # Apply file_types filter
            if file_types and file_path.suffix.lstrip(".") not in file_types:
                continue

            # Apply exclude_patterns filter
            if exclude_patterns:
                should_exclude = False
                for exclude_pattern in exclude_patterns:
                    # Check against the full relative path
                    if fnmatch.fnmatch(str(relative_file_path), exclude_pattern):
                        should_exclude = True
                        break
                    # Check against the file name only
                    if fnmatch.fnmatch(file_path.name, exclude_pattern):
                        should_exclude = True
                        break
                if should_exclude:
                    continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line_content in enumerate(f, 1):
                        if re.search(pattern, line_content):
                            search_results.append(
                                {
                                    "file_path": str(file_path),
                                    "line_number": line_num,
                                    "line_content": line_content.strip(),
                                }
                            )
            except FileNotFoundError:
                print(
                    f"Warning: File not found during search: {file_path}. Skipping."
                )
                continue
            except Exception as e:
                print(f"Error processing file {file_path}: {e}. Skipping.")
                continue
        return search_results

    def get_file_structure(self, codebase_path: str, file_path: str) -> list[str]:
        abs_codebase_path = str(Path(codebase_path).resolve())
        if abs_codebase_path not in self.indexed_codebases:
            raise ValueError(
                f"Codebase at {codebase_path} has not been indexed. Please index it first."
            )

        full_file_path = Path(codebase_path) / file_path # file_path is now relative to codebase_path
        if not full_file_path.exists():
            raise FileNotFoundError(f"File not found in codebase: {file_path}")

        structure = []
        try:
            with open(full_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content)

            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    structure.append(f"  Function: {node.name}")
                elif isinstance(node, ast.ClassDef):
                    structure.append(f"  Class: {node.name}")
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            structure.append(f"    Method: {item.name}")
        except Exception as e:
            return [f"Error parsing file structure: {e}"]

        return [f"Structure of {file_path}:"] + structure

    def _summarize_with_groq(self, code_snippet: str, llm_model: str) -> str:
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
        if not self.google_ai_client:
            return "Error: GOOGLE_API_KEY not configured."
        try:
            model = self.google_ai_client.GenerativeModel(llm_model.replace("google/", "", 1))
            response = model.generate_content(
                f"Please summarize the following code snippet:\n\n```\n{code_snippet}```"
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


codebase_manager = CodebaseManager()
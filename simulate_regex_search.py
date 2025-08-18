import os
import sys
import json
import re  # Import re


class MockCodebaseManager:
    def __init__(self):
        self.indexed_codebases = {}

    def index_codebase(self, path: str) -> list[str]:
        if not os.path.isdir(path):
            raise ValueError(f"Path is not a directory: {path}")

        indexed_files = []
        for root, _, files in os.walk(path):
            for file in files:
                full_path = os.path.join(root, file)
                indexed_files.append(full_path)

        self.indexed_codebases[path] = {"status": "indexed", "files": indexed_files}
        return indexed_files

    def search_codebase(
        self, codebase_path: str, pattern: str, file_types: list[str] = None
    ) -> list[dict]:
        if codebase_path not in self.indexed_codebases:
            raise ValueError(f"Codebase at {codebase_path} has not been indexed.")

        indexed_files = self.indexed_codebases[codebase_path]["files"]
        search_results = []

        for file_path in indexed_files:
            if (
                file_types
                and os.path.splitext(file_path)[1].lstrip(".") not in file_types
            ):
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line_content in enumerate(f, 1):
                        if re.search(pattern, line_content):  # Use regex for search
                            search_results.append(
                                {
                                    "file_path": file_path,
                                    "line_number": line_num,
                                    "line_content": line_content.strip(),
                                }
                            )
            except Exception:
                # Suppress for simulation
                # print(f"Error reading file {file_path}: {e}")
                continue
        return search_results


def main():
    if len(sys.argv) < 3:
        print(
            json.dumps(
                {
                    "error": (
                        "Usage: python simulate_regex_search.py <codebase_path> "
                        "<regex_pattern> [file_type1 file_type2 ...]"
                    )
                }
            )
        )
        sys.exit(1)

    codebase_path = sys.argv[1]
    pattern = sys.argv[2]
    file_types = sys.argv[3:] if len(sys.argv) > 3 else None

    manager = MockCodebaseManager()
    try:
        # First, index the codebase
        manager.index_codebase(codebase_path)

        # Then, search it
        results = manager.search_codebase(codebase_path, pattern, file_types)
        print(
            json.dumps(
                {
                    "message": (
                        f"Found {len(results)} matches for pattern '{pattern}'."
                    ),
                    "results": results,
                }
            )
        )
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
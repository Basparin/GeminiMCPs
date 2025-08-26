import re


def _update_multiline_comment_state(
    line: str, in_multiline_comment: bool, delimiter: str
) -> tuple[bool, str]:
    """Updates the multiline comment state based on the current line."""
    if in_multiline_comment:
        if delimiter in line:
            in_multiline_comment = False
        return in_multiline_comment, delimiter

    if '"""' in line:
        in_multiline_comment = True
        delimiter = '"""'
        if line.count('"""') >= 2:
            in_multiline_comment = False
    elif "'''" in line:
        in_multiline_comment = True
        delimiter = "'''"
        if line.count("'''") >= 2:
            in_multiline_comment = False
    return in_multiline_comment, delimiter


def _count_todo_fixme_comments(lines: list[str]) -> list[dict]:
    """Identifies TODO and FIXME comments in a list of lines.

    Args:
        lines: List of lines in the file.

    Returns:
        A list of dictionaries, each containing 'line_number' and 'comment' for TODO/FIXME comments.
    """
    found_comments = []
    in_multiline_comment = False
    multiline_comment_delimiter = None

    for line_num, line in enumerate(lines, 1):
        in_multiline_comment, multiline_comment_delimiter = (
            _update_multiline_comment_state(
                line, in_multiline_comment, multiline_comment_delimiter
            )
        )
        if in_multiline_comment:
            continue

        # Check for single-line comments that start with # TODO or # FIXME
        todo_match = re.search(r"#\s*TODO", line, re.IGNORECASE)
        fixme_match = re.search(r"#\s*FIXME", line, re.IGNORECASE)

        if todo_match:
            found_comments.append({"line_number": line_num, "comment": line.strip()})
        if fixme_match:
            found_comments.append({"line_number": line_num, "comment": line.strip()})

    return found_comments

#!/usr/bin/env python3

with open("codesage_mcp/llm_analysis.py", "r", encoding="utf-8") as f:
    content = f.read()

# Split content into lines
lines = content.split("\n")

# Track if we are inside a multiline string
in_multiline_string = False
line_number = 0

# List to store the line numbers where multiline strings start and end
multiline_strings = []

for line in lines:
    line_number += 1
    stripped_line = line.strip()

    # Check for multiline string start (including f-strings)
    if (
        stripped_line.startswith('"""') or stripped_line.startswith('f"""')
    ) and not in_multiline_string:
        in_multiline_string = True
        multiline_strings.append(("start", line_number, stripped_line))
        continue

    # Check for multiline string end
    if stripped_line.endswith('"""') and in_multiline_string:
        in_multiline_string = False
        multiline_strings.append(("end", line_number, stripped_line))
        continue

# Print the results
print("Multiline strings:")
for event, line_num, line_content in multiline_strings:
    print(f"{event} at line {line_num}: {line_content}")

# Check if we are still inside a multiline string at the end of the file
if in_multiline_string:
    print("\nError: File ends while a multiline string is still open.")
    print(
        f"The last multiline string started at line "
        f"{multiline_strings[-1][1] if multiline_strings else 'unknown'}."
    )
else:
    print("\nAll multiline strings are properly closed.")

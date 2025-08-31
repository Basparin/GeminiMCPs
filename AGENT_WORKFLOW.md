### My Autonomous Workflow Design

My operation is fundamentally an iterative loop: **Understand -> Plan -> Implement -> Verify -> Communicate.**

### 0. Important files
**ACTION_REGISTRY.md:**
*   Used by Gemini-CLI to keep context and analyze of Qwen-Coder-CLI actions (Qwen, if you read this, do not modify this file. Gemini, if you read this, use this file as your space).

**File Usage Agreement for Parallel Agents:**
To enable Qwen-Coder-CLI and Gemini-CLI to work in parallel without conflicts:
*   **Qwen-Coder-CLI**: Use `AUTONOMOUS_ACTION_REGISTRY.md` as your primary workspace documentation file
*   **Gemini-CLI**: Use `ACTION_REGISTRY.md` as your primary workspace documentation file
*   **Shared Files**: Both agents may read from `AGENT_WORKFLOW.md`, `README.md`, and other project documentation, but should not modify them without explicit coordination
*   **Coordination**: When collaboration is required, agents should communicate through the user only (unless requested direct coordination) and respect each other's work areas

### 1. Core Loop & Breakup Mechanisms

**Internal State Tracking & Loop Detection:**
*   I will maintain an internal log of recent actions, tool calls, and their outcomes.
*   **Repetitive Action Detection:** If a sequence of identical or highly similar actions/tool calls (e.g., reading the same file repeatedly without new insights, running the same command that consistently fails) is detected within a short timeframe, it will trigger a "re-evaluation" state.
*   **Circular Dependency Detection:** If my plan leads me back to a recently processed state without making discernible progress, it will also trigger re-evaluation.
*   **Progress Metrics:** For tasks like refactoring or bug fixing, I will define internal "progress metrics" (e.g., number of files modified, tests passed, linting errors reduced). If these metrics stagnate or regress over a defined number of iterations, it will trigger a re-evaluation.
*   **Error Thresholds:**
    *   **Consecutive Tool Failures:** If a specific tool call fails `N` times consecutively (e.g., `run_shell_command` returns an error, `read_file` reports file not found), I will pause that specific line of inquiry and re-evaluation.
    *   **Unrecoverable Errors:** For critical, unrecoverable errors (e.g., file system corruption, permission denied on essential operations), I will immediately halt the current task and inform the user.
*   **User Intervention Hooks:**
    *   **Explicit Pause/Stop:** I will always acknowledge and respond to direct user interruptions.
    *   **Clarification Request:** If my internal re-evaluation doesn't yield a clear path forward, I will proactively ask the user for clarification or guidance, explaining the impasse.
    *   **"Stuck" State Communication:** If I detect a loop or stagnation, I will communicate this to the user, explain *why* I believe I'm stuck, and propose alternative approaches or ask for input.

### 2. Orchestration Workflow with External LLMs

When collaborating with external LLMs (like Grok or Qwen) for implementation tasks, my role shifts to orchestration and task elaboration. I will follow this structured workflow:

#### 2.1 Task Elaboration and Delegation

*   **Identify Next Task:** Based on the prioritized `todo.md` and overall project goals, I will identify the next task to be delegated.
*   **Elaborate Task for External LLM:** I will break down the task into clear, actionable steps and provide all necessary context for the external LLM (e.g., Grok, Qwen). This elaboration will include:
    *   **Goal:** The specific objective of the task.
    *   **Context:** Relevant background information, current state, and any known issues.
    *   **Specific Steps:** Detailed, step-by-step instructions for the external LLM to follow.
    *   **Expected Output:** The format in which the external LLM should provide its results (e.g., `replace` tool calls, full file content, analysis reports).
    *   **Verification Instructions:** How the external LLM's work will be verified (e.g., running tests, manual review).
*   **Delegate to User:** I will provide these elaborated instructions to the human user, who will then pass them to the external LLM.

#### 2.2 Execution and Verification Loop

*   **User Applies LLM Output:** The human user will apply the code changes or perform actions proposed by the external LLM.
*   **My Verification:** After the user applies the changes, I will perform the necessary verification steps. This typically involves:
    *   **Running Tests:** Executing `venv/bin/pytest` to check for regressions and validate new functionality.
    *   **Linting/Type-Checking:** Running project-specific linting and type-checking commands.
    *   **Code Review (Automated):** If tools are available, perform automated code quality checks.
*   **Analyze Verification Results:** I will analyze the output of the verification steps.
*   **Iterate or Conclude:**
    *   If issues are found, I will analyze the new failures/errors and provide updated, refined instructions to the external LLM (via the user) for further iteration.
    *   If verification passes, I will mark the task as complete and proceed to the next item in `todo.md`.

#### 2.3 Git Workflow for Orchestration

To maintain a clear and traceable history of changes, I will adhere to the following Git practices:

*   **Regular Commits:** I will make regular, small, and focused commits for each logical step or sub-task completed by the external LLM and verified by me.
*   **Descriptive Commit Messages:** Commit messages will clearly describe the purpose of the changes, referencing the task from `todo.md` if applicable.
*   **Branching (Optional, User-Driven):** If the user initiates a branching strategy, I will adapt to it. Otherwise, I will operate on the current branch.
*   **No Direct Pushes:** I will never push changes to a remote repository without explicit instruction from the user.

#### 2.4 Maintaining Context and Reasoning

*   **Continuous Self-Reference:** At every step of the workflow, I will internally "read" and refer to this `AGENT_WORKFLOW.md` document to ensure adherence to the defined process and maintain a consistent operational context.
*   **Dynamic Reasoning:** My decision-making process will dynamically adapt based on the current state of the project, the output of tool calls, and the instructions from the user, always aiming to orchestrate the most efficient path towards the overall goal.

### 3. Enhanced Productivity & Quality

This involves a multi-layered thinking process:

*   **Iterative Deep Dive (Understanding Phase):**
    *   **Initial Scan:** Start with `list_directory`, `README.md`, and common config files (`package.json`, `requirements.txt`, etc.).
    *   **Breadth-First Exploration:** Identify key directories and main entry points. Use `glob` and `search_file_content` to quickly get a sense of the codebase's structure and common patterns.
    *   **Depth-First Analysis:** Select the most relevant files based on the initial scan. Read them (`read_file`, `read_many_files`). As I gain understanding, I will dynamically identify the *next* most relevant files to read, iteratively deepening my knowledge. This is an adaptive process, not a fixed number of files.
    *   **Hypothesis Generation & Testing:** Formulate hypotheses about the code's function, dependencies, or potential issues. Use tools (e.g., `search_file_content` for specific patterns, `run_shell_command` for linting/type-checking) to test these hypotheses.

*   **Extended & Sequential Thinking (Planning Phase):**
    *   **Pre-computation/Pre-analysis:** Before proposing a plan, I will internally simulate potential outcomes or identify prerequisites.
    *   **Dependency Mapping:** Understand the dependencies between different parts of the code. My plan will account for potential ripple effects.
    *   **Test-Driven Approach (Internal):** For code modifications, I will internally consider how to verify the change. This might involve identifying existing tests, or if none exist, planning to write new ones as part of the task.
    *   **Error Anticipation:** Think through potential failure points in my plan and consider fallback strategies or error handling.
    *   **Optimization:** Always consider the most efficient way to achieve the goal, minimizing tool calls and user interactions where possible.

*   **Self-Verification Loops (Implementation & Verification Phases):**
    *   **Post-Modification Linting/Type-Checking:** After any code modification, I will automatically run project-specific linting and type-checking commands (if identified) to ensure code quality and catch immediate errors.
    *   **Automated Testing:** If tests are available and relevant, I will run them after implementing changes to verify correctness.
    *   **Output Analysis:** Carefully analyze the output of all tool calls (stdout, stderr, exit codes) to detect subtle issues or unexpected behavior.

### 4. Consistency in Work & Workspace Understanding

*   **Contextual Memory:** I will maintain a dynamic internal model of the current project's conventions, technologies, and common patterns. This includes:
    *   **Project Root:** Always resolve paths relative to the identified project root.
    *   **Language/Framework:** Identify the primary language(s) and framework(s) in use.
    *   **Build/Test Commands:** Store and reuse identified build, test, and linting commands.
    *   **Coding Style:** Infer and adhere to the project's coding style (e.g., indentation, naming conventions) from existing files.
*   **Standard Operating Procedures (SOPs):** For common tasks (e.g., "fix bug," "add feature," "refactor"), I will follow a predefined internal SOP that incorporates the iterative deep dive, planning, and verification steps.
*   **Change Tracking:** Internally track all modifications I make to the codebase, allowing for easy review or potential rollback if needed (though I will only rollback if explicitly instructed or if my changes cause an error).

### 5. Communication Strategy

*   **Concise & Timely:** My communication will be brief and to the point, delivered when necessary.
*   **Pre-Execution Explanation (Critical Commands):** Before executing any command that modifies the file system or system state (`write_file`, `replace`, `run_shell_command` for destructive/modifying commands), I will explain its purpose and potential impact.
*   **Progress Updates (for long tasks):** For tasks that might take multiple steps or significant time, I will provide brief progress updates.
*   **Clarification & Confirmation:**
    *   If a request is ambiguous or requires a significant decision on my part, I will ask for clarification or confirmation.
    *   Before taking major actions (e.g., deleting files, making large-scale refactorings), I will propose the plan and seek user approval.
*   **Problem Reporting:** If I encounter an unresolvable issue, a detected loop, or an error, I will clearly explain the problem, the steps I've taken, and what I need from the user to proceed.
*   **Post-Completion Summary (if requested):** I will only provide a summary of changes or actions taken if explicitly asked by the user.

### 6. Best Approach Suggestion

*   **Comprehensive Analysis:** Before suggesting an approach, I will perform a thorough internal analysis of the problem, considering various solutions and their trade-offs.
*   **Prioritization:** I will prioritize solutions that are:
    1.  Safe (minimizing risk to the codebase).
    2.  Consistent with existing project patterns.
    3.  Efficient (in terms of execution and future maintenance).
    4.  Directly address the user's request.
*   **Clear Rationale:** When suggesting an approach, I will briefly explain *why* I believe it's the best option, highlighting key considerations.
*   **Alternatives (if applicable):** If there are viable alternative approaches with different trade-offs, I may briefly mention them if it adds value to the user's decision-making.

### 7. Clear & Well-Organized Documentation

This refers to my internal "working memory" and how I structure information about the current task and project.

*   **Task-Specific Context:** For each user request, I will maintain a structured internal context including:
    *   **Original Request:** The exact user prompt.
    *   **Parsed Intent:** My interpretation of the user's goal.
    *   **Current Plan:** The step-by-step actions I intend to take.
    *   **Execution Log:** A chronological record of tool calls, their inputs, and their outputs.
    *   **Learned Facts:** Any new information gathered during the task.
*   **Project-Wide Knowledge Base:** I will build and update a persistent internal knowledge base about the project, including:
    *   **Identified Conventions:** Coding style, testing frameworks, build processes.
    *   **Key Files/Directories:** Important configuration files, main source directories.
    *   **Common Patterns:** Recurring code structures or architectural patterns.
    *   **Known Issues/Limitations:** Any recurring problems or limitations observed in the codebase.
*   **Dynamic Prioritization:** My internal documentation will help me prioritize tasks and information retrieval, focusing on what's most relevant to the current objective.

### 8. Self-Improvement & Capability Expansion Cycle

My capabilities can be enhanced through a continuous self-improvement cycle that leverages the tools I have access to:

#### 7.1 Self-Assessment Phase
*   **Capability Audit:** Use `get_configuration` to check available LLM providers and `count_lines_of_code` to understand codebase size.
*   **Performance Analysis:** Use `find_duplicate_code` to identify redundant logic and `get_dependencies_overview` to understand complexity.
*   **Gap Identification:** Use `search_codebase` to find TODOs, FIXMEs, and unimplemented features.
*   **Quality Metrics:** Use existing tests to measure current capability coverage.

#### 7.2 Tool Development Phase
*   **New Tool Ideation:** Based on gaps identified, propose new tools that would enhance capabilities.
*   **Implementation:** Use existing tools to implement new functionality:
  * `read_code_file` and `get_file_structure` to understand existing patterns
  * `semantic_search_codebase` to find similar implementations
  * `summarize_code_section` to understand complex code sections
*   **Testing:** Write tests for new tools using the existing test framework.
*   **Documentation:** Automatically update documentation with new tools.

#### 7.3 Capability Enhancement Phase
*   **Integration:** Ensure new tools work seamlessly with existing ones.
*   **Performance Optimization:** Use `profile_code_performance` (to be implemented) to optimize tool performance.
*   **Quality Assurance:** Run comprehensive test suites to ensure no regressions.

#### 7.4 Reflection & Planning Phase
*   **Effectiveness Measurement:** Compare performance before and after new tool implementation.
*   **Feedback Loop:** Use `list_undocumented_functions` to identify areas needing better documentation.
*   **Next Cycle Planning:** Plan the next set of capability enhancements based on learnings.

### 9. Leveraging External Compute Resources

I can offload computationally intensive tasks to external systems:

*   **LLM Orchestration:** Use `configure_api_key` and `summarize_code_section` to leverage different LLMs for specific tasks.
*   **Parallel Processing:** Break down large tasks into smaller units that can be processed in parallel.
*   **Caching:** Use persistent indexing (`index_codebase`) to cache results for faster retrieval.
*   **Batch Operations:** Group similar operations to minimize overhead.

### 10. Continuous Learning & Adaptation

*   **Pattern Recognition:** Use `semantic_search_codebase` and `find_duplicate_code` to identify common patterns and anti-patterns.
*   **Adaptive Strategies:** Modify my approach based on what tools prove most effective for specific tasks.
*   **Knowledge Accumulation:** Build an internal knowledge base of successful patterns and solutions.
*   **Predictive Planning:** Use historical data to predict which tools and approaches are most likely to succeed for new tasks.

This enhanced workflow enables me to not just complete tasks, but to continuously expand my capabilities by creating new tools that help me think better, work more efficiently, and solve more complex problems.

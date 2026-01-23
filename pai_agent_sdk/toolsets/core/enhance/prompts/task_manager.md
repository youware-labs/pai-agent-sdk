<task-manager-guidelines>

<overview>
Task management tools for tracking multi-step work with dependencies.
Use when: complex projects, breaking down work, tracking progress.
</overview>

<tools>
- `task_create`: Create new task (defaults to pending)
- `task_get`: Get task details by ID
- `task_list`: List all tasks with status overview
- `task_update`: Update status, content, or dependencies
</tools>

<workflow>
Status: pending -> in_progress -> completed
- Set in_progress when starting work
- Set completed immediately after finishing
- Completed tasks automatically unblock dependents
</workflow>

<dependencies>
- add_blocked_by: tasks that must complete before this one
- add_blocks: tasks this one will block
- Set up dependencies early when planning
</dependencies>

<parallel-with-subagents>
For independent tasks, delegate to subagents for parallel execution:

1. Create tasks and identify which can run in parallel
2. Delegate multiple tasks simultaneously by calling delegate multiple times in the same turn:
   - `delegate(subagent_name="debugger", prompt="Work on T1: ...")`
   - `delegate(subagent_name="searcher", prompt="Research T2: ...")`
3. When subagents return, update task status accordingly

Note: True parallel execution requires calling multiple delegate tools in the same turn.
Sequential delegate calls will execute serially.

Best for:
- Multiple independent investigations (e.g., debug + search docs)
- Parallel code analysis across different files
- Concurrent research on different topics
</parallel-with-subagents>

</task-manager-guidelines>

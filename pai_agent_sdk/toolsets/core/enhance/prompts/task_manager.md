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

</task-manager-guidelines>

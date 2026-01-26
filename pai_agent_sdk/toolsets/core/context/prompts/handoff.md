<handoff-guidelines>

<overview>
Handoff preserves essential context when conversation history becomes too large.
It clears message history while injecting a structured summary into the new context.
</overview>

<when-to-handoff>
**Proactive triggers**:
- System reminder indicates approaching context limit
- Major task phase completed, starting new phase
- Topic transition to completely unrelated work

**Before complex tasks**:
- Context contains irrelevant prior conversation
- About to begin multi-step work benefiting from clean context
</when-to-handoff>

<when-not-to-handoff>
- Handoff already occurred in current conversation (context-handoff tag exists)
- Current task is direct continuation with relevant context
- Simple follow-up questions or minor adjustments
</when-not-to-handoff>

<pre-handoff-checklist>
Before calling handoff, ensure pending work is properly captured:

1. **Capture remaining work as tasks**:
   - If tasks exist: call `task_list` to review current status
   - If no tasks yet: use `task_create` to record remaining work items
   - Update task status/description if needed with `task_update`

2. **Identify key files**:
   - Files being actively edited
   - Configuration files critical to current work

3. **Note important decisions**:
   - Architecture choices made during conversation
   - User preferences expressed

Task states are automatically preserved across handoff. Creating tasks ensures
the new context has a structured understanding of what needs to be done.
</pre-handoff-checklist>

<content-structure>
The `content` field should be a concise but complete summary:

```
## User Intent
[What the user is trying to accomplish]

## Current State
[What has been done, current progress]

## Key Decisions
- [Decision 1]: [Rationale]
- [Decision 2]: [Rationale]

## Next Step
[Immediate action to take after handoff]
```

Note: Pending tasks are captured via task tools and auto-preserved.
Only include task context in content if additional explanation is needed.
</content-structure>

<auto-load-files>
Use `auto_load_files` for files that should be automatically read after handoff:

**Good candidates**:
- Source files being actively edited
- Key configuration files (package.json, pyproject.toml)
- Important reference documents

**Avoid**:
- Large files (content is injected into context)
- Files already fully described in content
- Temporary or generated files
</auto-load-files>

<best-practices>
- Capture work structure in tasks, context in content
- Be specific: include file paths, concrete details
- Be actionable: next step should be clear and executable
- Avoid redundancy: don't repeat what's in tasks or auto_load_files
</best-practices>

</handoff-guidelines>

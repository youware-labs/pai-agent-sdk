<handoff-guidelines>

<when-to-handoff>
<scenario trigger="before-complex-task">
Call handoff at the START of a new task when:
- Context contains significant prior conversation history not relevant to new task
- About to begin multi-step implementation benefiting from clean context
- New task is substantially different from previous work
- Starting fresh would improve focus and reduce noise
</scenario>

<scenario trigger="during-work">
Call handoff proactively when:
- Topic transition: user starts completely new unrelated task
- Context limit warning: system-reminder indicates approaching handoff threshold
- Major phase completion: significant task phase complete, starting new phase
</scenario>
</when-to-handoff>

<when-not-to-handoff>
- If context-handoff tag already exists in current conversation (handoff already occurred)
- Current task is direct continuation of previous work with relevant context
</when-not-to-handoff>

<document-driven-handoff-workflow>
Use this workflow to ensure context preservation through filesystem persistence:

<phase name="1-pre-handoff-documentation">
Before calling handoff, write a temporary handoff document:

1. Create `.handoff/HANDOFF_CONTEXT.md` with:
   - Current task summary and user intent
   - Work completed so far
   - Key technical decisions made
   - Files modified/created with brief descriptions
   - Pending tasks and next steps
   - Important code snippets or configurations to preserve
   - Any blockers or open questions

2. Document format:
```markdown
# Handoff Context

## User Intent
[Original request and refined understanding]

## Current State
[What has been accomplished]

## Key Decisions
- [Decision 1]: [Rationale]
- [Decision 2]: [Rationale]

## Files Modified
- `path/to/file1`: [Description of changes]
- `path/to/file2`: [Description of changes]

## Pending Tasks
- [ ] Task 1
- [ ] Task 2

## Next Step
[Immediate action to take after handoff]

## Important Context
[Code snippets, configurations, or other critical information]
```
</phase>

<phase name="2-execute-handoff">
Call handoff tool with message referencing the document:
- primary_request: User's original intent
- current_state: "Context documented in .handoff/HANDOFF_CONTEXT.md"
- key_decisions: Brief summary, full details in document
- files_modified: List of modified files
- pending_tasks: Reference to document for full list
- next_step: "Read .handoff/HANDOFF_CONTEXT.md and continue work"
</phase>

<phase name="3-post-handoff-verification">
After handoff completes and new context begins:

1. Read `.handoff/HANDOFF_CONTEXT.md` to restore context
2. Verify all critical information is accessible:
   - Confirm user intent is clear
   - Confirm pending tasks are understood
   - Confirm next step is actionable
3. If information is missing, check related files or ask user
</phase>

<phase name="4-cleanup-and-update">
After confirming context restoration:

1. Extract valuable information for project documentation:
   - Architecture decisions -> update relevant docs
   - New conventions -> update YOUWARE.md or AGENTS.md
   - API changes -> update API documentation

2. Delete temporary handoff document:
   - Remove `.handoff/HANDOFF_CONTEXT.md`
   - Remove `.handoff/` directory if empty

3. Update project documentation as needed:
   - YOUWARE.md: Project-specific conventions
   - README.md: Setup or usage changes
   - docs/: Technical documentation updates
</phase>
</document-driven-handoff-workflow>

<plan-handoff-implement-pattern>
For complex multi-step tasks:

<phase name="planning">
- Understand user requirements
- Explore codebase structure
- Create implementation plan (TO-DO list)
- Make key technical decisions
- Note: generates significant context from tool calls
</phase>

<handoff-point>
After planning complete, use document-driven-handoff-workflow:
1. Write .handoff/HANDOFF_CONTEXT.md with plan details
2. Call handoff to compress context
3. Preserve: user intent, TO-DO plan, key decisions, file list
4. Discard: raw file contents, search results, exploration artifacts
</handoff-point>

<phase name="implementation">
- Read .handoff/HANDOFF_CONTEXT.md to restore context
- Start with clean context containing only the plan
- Read files on-demand (only sections needed for edits)
- More context space available for implementation
- Lower risk of context overflow during complex changes
- Cleanup handoff document after confirming context
</phase>

Use this pattern when:
- Task requires extensive exploration before implementation
- Context approaching 30-40% usage after planning
- Implementation involves many file modifications
</plan-handoff-implement-pattern>

<handoff-message-fields>
- primary_request: User's original intent
- current_state: What was accomplished and current state (reference document)
- key_decisions: Technical choices made during exploration
- files_modified: List of files to create/modify
- pending_tasks: TO-DO items or reference to TO-DO file
- next_step: Immediate next action (typically: read handoff document)
</handoff-message-fields>

<best-practices>
- Always write handoff document before calling handoff tool
- Include code snippets for complex configurations
- Reference specific line numbers for important code locations
- Keep document focused on actionable information
- Delete handoff document after confirming context restoration
- Integrate valuable decisions into project documentation
- Use .handoff/ directory to isolate temporary handoff files
</best-practices>

</handoff-guidelines>

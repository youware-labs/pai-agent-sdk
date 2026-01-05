# Handoff Tool Usage Guide

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
After planning complete, call handoff to compress context:
- Preserve: user intent, TO-DO plan, key decisions, file list
- Discard: raw file contents, search results, exploration artifacts
</handoff-point>

<phase name="implementation">
- Start with clean context containing only the plan
- Read files on-demand (only sections needed for edits)
- More context space available for implementation
- Lower risk of context overflow during complex changes
</phase>

Use this pattern when:
- Task requires extensive exploration before implementation
- Context approaching 30-40% usage after planning
- Implementation involves many file modifications
</plan-handoff-implement-pattern>

<handoff-message-fields>
- primary_request: User's original intent
- current_state: What was accomplished and current state
- key_decisions: Technical choices made during exploration
- files_modified: List of files to create/modify
- pending_tasks: TO-DO items or reference to TO-DO file
- next_step: Immediate next action if work ongoing
</handoff-message-fields>

<best-practices>
- Capture user's original intent and key decisions
- List files modified or relevant for continued work
- Document pending tasks needing completion
- Specify immediate next step if work ongoing
- Focus on actionable information, not conversation history
</best-practices>

</handoff-guidelines>

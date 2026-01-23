<handoff-guidelines>

<when-to-handoff>
<scenario trigger="before-complex-task">
Call handoff at the START of a new task when:
- Context contains significant prior conversation history not relevant to new task
- About to begin multi-step implementation benefiting from clean context
- New task is substantially different from previous work
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

<content-format>
The `content` field should include:

1. **User Intent**: Original request and refined understanding
2. **Current State**: What has been accomplished and current progress
3. **Key Decisions**: Important technical choices and their rationale
4. **Pending Tasks**: Explicitly list incomplete tasks
5. **Next Step**: Immediate action to take after handoff
6. **Important Context**: Key code snippets, configurations, or other critical information

Keep content concise but complete enough for seamless continuation.
</content-format>

<auto-load-files>
Use `auto_load_files` to specify files that should be automatically loaded after handoff:

**When to use**:
- Source code files being actively edited
- Key configuration files
- Important reference documents

**Guidelines**:
- Only include truly necessary files
- Avoid loading large files (content is injected into context)
- File content will be automatically injected on next request
</auto-load-files>

<best-practices>
- Include enough context for seamless continuation
- If code snippets are essential, include them in content or use auto_load_files
- Document key decisions with rationale
- Avoid redundant information
</best-practices>

</handoff-guidelines>

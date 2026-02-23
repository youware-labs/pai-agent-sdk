<write-tool>
Write or overwrite entire file content.

<best-practices>
- Always use `view` tool first to understand existing content
- For new files, verify parent directory exists with `ls`
- Limit content to ~200 lines per call for large files
- Use mode="a" for appending to existing files
- For partial edits: use `edit` or `multi_edit` tool instead
</best-practices>
</write-tool>

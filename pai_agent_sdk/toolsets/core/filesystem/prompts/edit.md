<edit-tool>
Performs exact string replacement in files.

<best-practices>
- old_string must match file content EXACTLY (including whitespace/indentation)
- Preserve exact indentation from view output (ignore line number prefixes)
- Include 3-5 lines of context to ensure unique matches
- Use replace_all=true for renaming variables across the file
- Empty old_string creates a new file (fails if file exists)
</best-practices>
</edit-tool>

<grep-tool>
Search file contents using regex patterns with context.

<best-practices>
- Use specific include patterns to narrow search scope
- Combine with glob to find files first, then grep specific ones
- Reduce context_lines for overview, increase for detailed context
- Directories are automatically skipped during search
- Files in .gitignore are excluded by default; set include_ignored=true to search ignored files (e.g., node_modules/, .venv/, build/)
</best-practices>
</grep-tool>

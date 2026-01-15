<shell-tool>
Execute shell commands. Commands are executed via `/bin/sh -c` (or `/bin/bash -c` depending on environment).

Parameters:
- command (required): The shell command string to execute
- timeout_seconds (default 180): Maximum execution time in seconds
- environment: Environment variables as key-value pairs
- cwd: Working directory (relative or absolute path)

Examples: `ls -la`, `npm install && npm run build`

Large outputs (>20KB) are saved to temporary files with paths in stdout_file_path/stderr_file_path.

Avoid:
- find/grep for searching - use grep, glob instead
- cat/head/tail/ls to read files - use view and ls tools
- cd command - use cwd parameter instead
</shell-tool>

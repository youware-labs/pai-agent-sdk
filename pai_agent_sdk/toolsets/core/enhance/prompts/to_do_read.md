<todo-read-guidelines>

<when-to-read>
- Only call `to_do_read` when you need to recover lost state (e.g., after handoff or context reset)
- Prefer relying on the last `to_do_write` result; avoid redundant reads
</when-to-read>

</todo-read-guidelines>

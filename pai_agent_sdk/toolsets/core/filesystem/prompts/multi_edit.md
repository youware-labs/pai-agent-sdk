<multi-edit-tool>
Perform multiple find-and-replace operations on a single file.

<best-practices>
- Prefer multi_edit over multiple single edits for the same file
- Each old_string must be unique (or use replace_all=true)
- Edits are applied sequentially - ensure earlier edits don't affect later ones
- All edits must succeed or none are applied (atomic operation)
- Empty old_string in first edit creates a new file
</best-practices>
</multi-edit-tool>

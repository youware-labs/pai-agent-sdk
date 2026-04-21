---
name: skill-creator
description: "Create and update SKILL.md files with proper YAML frontmatter, structured markdown body, and bundled resources (scripts, references, assets). Generates skill directories via init_skill.py, validates and packages skills via package_skill.py, and writes description fields with trigger terms. Use when creating a new skill, updating an existing SKILL.md, writing skill descriptions, structuring skill content, packaging a .skill file, or initializing a skill directory."
license: Complete terms in LICENSE.txt
---

# Skill Creator

## Core Principles

### Concise is Key

The context window is a public good. Skills share the context window with everything else Claude needs: system prompt, conversation history, other Skills' metadata, and the actual user request.

**Default assumption: Claude is already very smart.** Only add context Claude doesn't already have. Challenge each piece of information: "Does Claude really need this explanation?" and "Does this paragraph justify its token cost?"

Prefer concise examples over verbose explanations.

### Set Appropriate Degrees of Freedom

Match the level of specificity to the task's fragility and variability:

**High freedom (text-based instructions)**: Use when multiple approaches are valid, decisions depend on context, or heuristics guide the approach.

**Medium freedom (pseudocode or scripts with parameters)**: Use when a preferred pattern exists, some variation is acceptable, or configuration affects behavior.

**Low freedom (specific scripts, few parameters)**: Use when operations are fragile and error-prone, consistency is critical, or a specific sequence must be followed.

Think of Claude as exploring a path: a narrow bridge with cliffs needs specific guardrails (low freedom), while an open field allows many routes (high freedom).

### Anatomy of a Skill

Every skill consists of a required SKILL.md file and optional bundled resources:

```
skill-name/
├── SKILL.md (required)
│   ├── YAML frontmatter metadata (required)
│   │   ├── name: (required)
│   │   └── description: (required)
│   └── Markdown instructions (required)
└── Bundled Resources (optional)
    ├── scripts/          - Executable code (Python/Bash/etc.)
    ├── references/       - Documentation intended to be loaded into context as needed
    └── assets/           - Files used in output (templates, icons, fonts, etc.)
```

#### SKILL.md (required)

Every SKILL.md consists of:

- **Frontmatter** (YAML): Contains `name` and `description` fields. These are the only fields that Claude reads to determine when the skill gets used, thus it is very important to be clear and comprehensive in describing what the skill is, and when it should be used.
- **Body** (Markdown): Instructions and guidance for using the skill. Only loaded AFTER the skill triggers (if at all).

#### Bundled Resources (optional)

| Directory | Purpose | When to include |
|-----------|---------|-----------------|
| `scripts/` | Deterministic executable code (Python/Bash) | Same code rewritten repeatedly; reliability needed |
| `references/` | Documentation loaded into context on demand | Schemas, API docs, domain knowledge, policies |
| `assets/` | Output files (templates, images, fonts) | Skill produces output using these files |

**Key rules:**
- Information lives in SKILL.md **or** references — never both. Prefer references for detailed material.
- For large reference files (>10k words), include grep patterns in SKILL.md.
- Do NOT create auxiliary files (README.md, CHANGELOG.md, etc.) — only files the agent needs to do the job.

### Progressive Disclosure Design Principle

Skills use a three-level loading system to manage context efficiently:

1. **Metadata (name + description)** - Always in context (~100 words)
2. **SKILL.md body** - When skill triggers (<5k words, under 500 lines)
3. **Bundled resources** - As needed (scripts can execute without loading into context)

**Key principle:** Keep only the core workflow in SKILL.md. Move variant-specific details, lengthy examples, and domain-specific content into reference files. Link to them with clear guidance on when to read each one.

**Guidelines:**
- Keep references one level deep from SKILL.md — no nested chains
- For reference files >100 lines, include a table of contents at the top
- When splitting content, always reference the file from SKILL.md with a description of when to read it

## Skill Creation Process

Skill creation involves these steps:

1. Understand the skill with concrete examples
2. Plan reusable skill contents (scripts, references, assets)
3. Initialize the skill (run init_skill.py)
4. Edit the skill (implement resources and write SKILL.md)
5. Package the skill (run package_skill.py)
6. Iterate based on real usage

Follow these steps in order, skipping only if there is a clear reason why they are not applicable.

### Step 1: Understanding the Skill with Concrete Examples

Skip this step only when the skill's usage patterns are already clearly understood. It remains valuable even when working with an existing skill.

To create an effective skill, clearly understand concrete examples of how the skill will be used. This understanding can come from either direct user examples or generated examples that are validated with user feedback.

For example, when building an image-editor skill, relevant questions include:

- "What functionality should the image-editor skill support? Editing, rotating, anything else?"
- "Can you give some examples of how this skill would be used?"
- "I can imagine users asking for things like 'Remove the red-eye from this image' or 'Rotate this image'. Are there other ways you imagine this skill being used?"
- "What would a user say that should trigger this skill?"

To avoid overwhelming users, avoid asking too many questions in a single message. Start with the most important questions and follow up as needed for better effectiveness.

Conclude this step when there is a clear sense of the functionality the skill should support.

### Step 2: Planning the Reusable Skill Contents

For each concrete example, analyze: (1) how to execute from scratch, (2) what reusable resources would help when repeating the workflow.

| Pattern | Example | Resource |
|---------|---------|----------|
| Repeated code | PDF rotation → same code each time | `scripts/rotate_pdf.py` |
| Repeated boilerplate | Frontend app → same HTML/React setup | `assets/hello-world/` template |
| Repeated discovery | BigQuery → rediscovering schemas | `references/schema.md` |

Output: a list of scripts, references, and assets to include in the skill.

### Step 3: Initializing the Skill

Skip if the skill already exists. For new skills, run:

```bash
scripts/init_skill.py <skill-name> --path <output-directory>
```

This generates the skill directory with a SKILL.md template (frontmatter + TODO placeholders) and example `scripts/`, `references/`, `assets/` directories. Customize or remove generated files as needed.

### Step 4: Edit the Skill

Include information that is beneficial and non-obvious to Claude — procedural knowledge, domain-specific details, and reusable assets that another Claude instance couldn't infer.

**Design pattern references:**
- **Multi-step processes**: See [references/workflows.md](references/workflows.md)
- **Output formats/quality standards**: See [references/output-patterns.md](references/output-patterns.md)

**Implement resources first:** Start with `scripts/`, `references/`, and `assets/`. Test added scripts by running them. Delete unused example files from initialization.

#### Update SKILL.md

**Writing Guidelines:** Always use imperative/infinitive form.

##### Frontmatter

Write the YAML frontmatter with `name` and `description`:

- `name`: The skill name
- `description`: This is the primary triggering mechanism for your skill, and helps Claude understand when to use the skill.
  - Include both what the Skill does and specific triggers/contexts for when to use it.
  - Include all "when to use" information here - Not in the body. The body is only loaded after triggering, so "When to Use This Skill" sections in the body are not helpful to Claude.
  - Example description for a `docx` skill: "Comprehensive document creation, editing, and analysis with support for tracked changes, comments, formatting preservation, and text extraction. Use when Claude needs to work with professional documents (.docx files) for: (1) Creating new documents, (2) Modifying or editing content, (3) Working with tracked changes, (4) Adding comments, or any other document tasks"

Do not include any other fields in YAML frontmatter.

##### Body

Write instructions for using the skill and its bundled resources.

### Step 5: Packaging a Skill

```bash
scripts/package_skill.py <path/to/skill-folder> [output-directory]
```

The script validates (frontmatter, naming, structure, descriptions) then packages into a `.skill` file (zip with .skill extension). Fix any validation errors and re-run if it fails.

### Step 6: Iterate

After testing the skill, users may request improvements. Often this happens right after using the skill, with fresh context of how the skill performed.

**Iteration workflow:**

1. Use the skill on real tasks
2. Notice struggles or inefficiencies
3. Identify how SKILL.md or bundled resources should be updated
4. Implement changes and test again

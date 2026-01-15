#!/bin/bash
# Sync docs and .env.example to paintress_cli/paintress_cli/skills/building-agents

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

SKILL_DIR="paintress_cli/paintress_cli/skills/building-agents"

rm -rf "$SKILL_DIR"
mkdir -p "$SKILL_DIR/docs" "$SKILL_DIR/examples"
cp -r docs/* "$SKILL_DIR/docs/"
cp -r examples/* "$SKILL_DIR/examples/"
cp examples/.env.example "$SKILL_DIR/examples/"
cp README.md "$SKILL_DIR/"
cp SKILL.md "$SKILL_DIR/"

echo "Synced docs, examples, SKILL.md and .env.example to $SKILL_DIR"

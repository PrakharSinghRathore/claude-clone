---
name: git_workflow
description: Git workflow skill for managing branches, commits, merges, and collaboration using best practices.
version: 1.0.0
tags: [git, version-control, branching, collaboration]
dependencies: []
author: Hermes Team
---

# Git Workflow Skill

When this skill is active, follow established Git best practices for branching, committing, and collaborating.

## Overview

This skill provides a structured Git workflow that promotes clean history, safe collaboration, and easy rollback. It follows conventions suitable for both solo and team projects.

## Steps

1. **Assess current state**: Run `git status` and `git log --oneline -10` to understand the current branch and recent history. Check for any uncommitted changes or stashed work. Identify the target branch (main, develop, etc.).

2. **Branch management**:
   - Create a feature branch: `git checkout -b feature/descriptive-name`
   - Use a consistent naming convention (feature/, bugfix/, hotfix/, chore/)
   - Keep branches short-lived and focused on a single concern
   - Pull latest changes from the target branch before starting work

3. **Commit discipline**:
   - Make atomic commits (one logical change per commit)
   - Use imperative mood in commit messages ("Add feature" not "Added feature")
   - Follow the format: `type(scope): description` where type is feat|fix|refactor|docs|test|chore
   - Limit the first line to 72 characters
   - Include a blank line and detailed body for complex changes

4. **Before merging/rebasing**:
   - Run all tests and ensure they pass
   - Run the linter and fix any issues
   - Review your own changes with `git diff`
   - Update the branch with latest target: `git rebase target-branch`

5. **Merge and cleanup**:
   - Prefer `git merge --no-ff` for feature branches (preserves history)
   - Use `git rebase` for keeping a clean linear history on long-lived branches
   - Delete merged branches: `git branch -d feature/name`
   - Push tags for releases: `git tag -a v1.0.0 -m "Release 1.0.0"`

## Expected Outcome

A clean Git history with meaningful commits, properly merged branches, and no orphaned work, making it easy to understand the project's evolution and revert changes if needed.

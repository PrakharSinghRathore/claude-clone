---
name: debug
description: Systematic debugging skill for identifying, isolating, and fixing bugs in code.
version: 1.0.0
tags: [debug, troubleshooting, bugfix, investigation]
dependencies: []
author: Hermes Team
---

# Debug Skill

When this skill is active, follow a structured debugging methodology to identify and fix issues efficiently.

## Overview

This skill provides a systematic approach to debugging that minimizes time spent chasing false leads. It combines binary search techniques, hypothesis-driven investigation, and root cause analysis.

## Steps

1. **Reproduce the bug**: Get the exact error message or unexpected behavior. Determine the minimal steps to reproduce. Note the environment (OS, Python version, dependencies). Check if the issue is deterministic or intermittent.

2. **Gather information**: Read the full error traceback if available. Check relevant log files. Examine the input that triggered the issue. Review recent changes to the affected code (git log, git diff).

3. **Form hypotheses**: Based on the evidence, list 2-3 most likely causes. Prioritize by likelihood and ease of verification. Consider both code-level and environment-level causes.

4. **Test hypotheses (binary search)**:
   - Add strategic print/logging statements at key points
   - Use the Python debugger or equivalent to step through code
   - Isolate the issue by commenting out sections or using minimal inputs
   - Verify each hypothesis in order of likelihood
   - Narrow down the exact line or function causing the issue

5. **Identify root cause**: Determine the fundamental reason for the bug. Check if similar bugs could exist elsewhere (pattern search). Verify that the fix addresses the root cause, not just the symptom.

6. **Implement and verify the fix**:
   - Write the minimal fix that addresses the root cause
   - Verify the original issue is resolved
   - Check for regressions (run existing tests)
   - Add a test case that would have caught this bug
   - Consider if defensive coding can prevent similar issues

## Expected Outcome

A clearly identified root cause with a verified fix, a regression test to prevent recurrence, and documentation of the debugging process for future reference.

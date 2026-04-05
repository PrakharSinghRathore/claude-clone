---
name: code_review
description: Systematic code review skill for analyzing code quality, identifying issues, and providing improvement suggestions.
version: 1.0.0
tags: [code, review, quality, best-practices]
dependencies: []
author: Hermes Team
---

# Code Review Skill

When this skill is active, perform a thorough, multi-pass code review following a structured methodology.

## Overview

This skill provides a systematic approach to reviewing code for correctness, maintainability, performance, security, and adherence to best practices. Use it when the user requests a code review or when reviewing pull requests.

## Steps

1. **Understand the context**: Read the file(s) under review. Understand the project structure and conventions. Identify the purpose of the code and its expected behavior. Check for existing tests and documentation.

2. **First pass — Correctness**: Check for syntax errors and typos. Verify logic flow is correct. Ensure edge cases are handled. Look for off-by-one errors, null checks, and type mismatches. Verify error handling is comprehensive.

3. **Second pass — Code quality**: Check naming conventions (variables, functions, classes). Assess function/method length (aim for under 30 lines). Evaluate abstraction levels and separation of concerns. Look for code duplication that could be extracted. Verify proper use of language idioms.

4. **Third pass — Performance**: Identify O(n^2) or worse algorithms. Look for unnecessary allocations or copies. Check for N+1 query patterns (in database code). Verify proper use of caching where applicable. Look for synchronous operations that could be async.

5. **Fourth pass — Security**: Check for injection vulnerabilities (SQL, XSS, command). Verify proper input validation and sanitization. Review authentication and authorization logic. Check for sensitive data exposure (logs, error messages). Verify secure handling of secrets and credentials.

6. **Produce review report**: Structure findings by severity:
   - **Critical**: Bugs, security issues, data loss risks
   - **Major**: Significant quality or performance problems
   - **Minor**: Style issues, naming, minor improvements
   - **Suggestions**: Optional enhancements and refactoring ideas

   For each finding, include:
   - Location (file:line or function name)
   - Category and severity
   - Description of the issue
   - Suggested fix with code example

## Expected Outcome

A structured code review report with prioritized findings and actionable suggestions, enabling the developer to address issues systematically.

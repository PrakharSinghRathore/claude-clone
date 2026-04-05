---
name: deploy
description: Deployment skill for packaging, testing, and deploying applications to target environments.
version: 1.0.0
tags: [deploy, release, deployment, ci-cd]
dependencies: []
author: Hermes Team
---

# Deploy Skill

When this skill is active, follow a structured deployment process to ensure safe and reliable releases.

## Overview

This skill provides a comprehensive deployment checklist and workflow that covers pre-deployment checks, the deployment process itself, and post-deployment verification. It is designed to minimize deployment risk.

## Steps

1. **Pre-deployment checklist**:
   - Verify all tests pass in the target branch
   - Run linting and static analysis
   - Check that the version number has been bumped appropriately
   - Review the changelog for completeness
   - Verify environment-specific configuration is correct
   - Confirm database migrations are ready (if applicable)
   - Check that dependencies are locked and consistent

2. **Build and package**:
   - Clean the build directory
   - Build the application with production settings
   - Run the production build to verify it compiles/start without errors
   - Generate any required assets (compiled code, bundled JS, etc.)
   - Create a deployment artifact with a unique version identifier

3. **Staging deployment** (if available):
   - Deploy to the staging environment first
   - Run smoke tests against the staging deployment
   - Verify critical user flows work correctly
   - Check monitoring dashboards for errors
   - Get sign-off from required reviewers

4. **Production deployment**:
   - Notify the team of the impending deployment
   - Deploy using the established deployment method
   - Monitor the deployment process for errors
   - Verify the health endpoint returns healthy status
   - Run automated post-deployment tests

5. **Post-deployment verification**:
   - Monitor error rates and latency for 15-30 minutes
   - Verify key functionality through manual spot-checks
   - Check that database migrations ran successfully
   - Verify logging is working and at appropriate levels
   - Confirm monitoring alerts are properly configured
   - Document the deployment in the changelog

6. **Rollback plan**: If issues are detected:
   - Assess severity and impact
   - If critical: initiate immediate rollback to previous version
   - If non-critical: document the issue and plan a hotfix
   - Post-mortem: document what went wrong and how to prevent it

## Expected Outcome

A successfully deployed application with all pre and post checks verified, monitoring in place, and a clear rollback path documented.

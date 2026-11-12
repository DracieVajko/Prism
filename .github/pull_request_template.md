---
name: Pull request template
about: Template for contributing to Prism
title: "[FEATURE/FIX] "
labels: ""
assignees: ""
---

## Summary

Briefly describe what this PR changes and why.

## Related issue

Link to the issue this resolves, or "none".

## Changes

- 
- 

## Verification

- [ ] PC tests pass: `cd pc && python -m pytest tests -q`
- [ ] Android builds: `cd android && .\gradlew.bat :app:assembleDebug`
- [ ] Manual test against a running server / device
- [ ] Docs updated (`docs/CHECKLIST.md`, `docs/CHECKPOINTS.md`, README if needed)

## Checklist

- [ ] All code, docs, commit messages and UI strings are in English (`AGENTS.md`)
- [ ] No model weights are bundled or committed
- [ ] Unavailable features return `503` with a reason, never faked output
- [ ] No new runtime dependencies added without justification

# GitHub Sponsors + Project Presentation — task list (COMPLETE)

## Phase 1: Understand (done)
- [x] Read repository structure (pc/, android/, docs/, .github/, root files)
- [x] Read README.md, pc/README.md, android/README.md, LICENSE, AGENTS.md, CHANGELOG.md, VERSION
- [x] Read docs/ARCHITECTURE.md, ONDEVICE-PLAN.md, RESEARCH-IMPROVEMENTS.md, ANDROID-MODELS.md, MODEL-CHECKLIST.md
- [x] Read .github/FUNDING.yml, ci.yml, release.yml
- [x] Identified project: two independent products (PC server + Android app), OpenAI-compatible API, port 8101, lazy models, single resident model, honest 503s, MIT, v0.7.0

## Phase 2: README redesign (done)
- [x] Root README.md rewritten: intro, problem solved, main features, architecture, platforms, installation, configuration, usage examples, screenshots reference, development setup, roadmap, known limitations, contribution guidelines, Sponsors section with tiers

## Phase 3: Project identity (done)
- [x] Reviewed consistency: description, headings, terminology, badges
- [x] Added badges: License MIT, Version 0.7.0, Sponsor on GitHub, PC CI

## Phase 4: GitHub Sponsors (done)
- [x] .github/FUNDING.yml: verified `github: DracieVajko` (already present, YAML valid)
- [x] Added project-specific sponsorship section to README
- [x] Realistic tier descriptions (User/Contributor/Sponsor/Partner), no exclusive features or guaranteed dates
- [x] Noted: activate only after Sponsors application approved

## Phase 5: Open-source quality audit (done)
- [x] License: MIT (present)
- [x] Contribution policy: AGENTS.md (present)
- [x] Issue templates: added .github/ISSUE_TEMPLATE/bug_report.md
- [x] PR template: added .github/pull_request_template.md
- [x] Security policy: added .github/SECURITY.md (LAN-only model, no auth, disclosure process)
- [x] CI config: ci.yml + release.yml (present, YAML valid)
- [x] Dependency management: requirements.txt, build.gradle.kts (present)
- [x] Documentation completeness: docs/ + pc/docs/ + android/README.md (present)

## Phase 6: Validation (done)
- [x] YAML syntax (.github/FUNDING.yml, workflows) — PASS
- [x] Markdown rendering (README.md, SECURITY.md, templates) — PASS
- [x] Relative links, external links, badge URLs — PASS (all 14 relative links resolve)
- [x] Installation instructions — verified against pc/README.md and android/README.md
- [x] Existing tests pass — 143 PC tests pass

## Phase 7: Git workflow (done)
- [x] Branch: feat/github-sponsors-and-project-presentation
- [x] Commit: 142f661 (4 files, +294/-85)
- [ ] Not merged, not pushed (per instruction)

## Files changed in this commit
- M README.md (redesigned)
- A .github/SECURITY.md (new)
- A .github/ISSUE_TEMPLATE/bug_report.md (new)
- A .github/pull_request_template.md (new)

## Not included in this commit (separate task artifacts, untracked)
- docs/MODEL-CHECKLIST.md (modified — model cleanup: Gemma removed, Whisper turbo+large kept)
- docs/SPONSORS-TASKS.md (this file — internal tracking)
- pc/scripts/compare_models/results-2026-11-11-nemotron.txt (Nemotron ASR benchmark)
- pc/scripts/transcribe_income.py (batch transcriber, recreated into pc/scripts/)

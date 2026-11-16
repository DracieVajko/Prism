# Full Repository Cleanup & Improvement — Task Tracker

## Phase 1: Project Understanding & Desloppify (IN PROGRESS)
- [ ] Read AGENTS.md, READMEs, CI config, source code (done via exploration)
- [ ] Install/update Desloppify via uvx
- [ ] Verify `.desloppify/` excluded from Git (confirmed)
- [ ] Run `desloppify scan --path .`
- [ ] Review findings
- [ ] Execute next → fix → resolve → rescan loop
- [ ] Pay attention to: monolithic modules, duplicate implementations, unnecessary abstractions, complex lifecycle/state, weak separation of layers, missing tests, inconsistent naming/error handling

## Phase 2: PC Server Architecture
- [ ] Investigate `pc/unified_server.py` responsibilities
- [ ] Separate API endpoints, model lifecycle, audio, image, config, background tasks into logical modules
- [ ] Preserve existing API endpoints, contracts, WebUI behavior
- [ ] Add regression tests before and after refactoring

## Phase 3: Security Improvements
- [ ] Default bind to 127.0.0.1 for local use
- [ ] Allow LAN access explicitly via config or startup argument
- [ ] Clear warning about unauthenticated API
- [ ] Review all launch scripts, env vars, server configuration

## Phase 4: Dependencies & GPU Support
- [ ] Review `pc/requirements.txt`
- [ ] Organize for CPU / NVIDIA CUDA / AMD installation options
- [ ] Clear installation commands and platform-specific requirements

## Phase 5: CI & Testing
- [ ] Fix `|| true` in Schemathesis fuzzing
- [ ] CI must fail on genuine critical regressions
- [ ] Separate optional experiments from mandatory tests
- [ ] Add useful Python tests
- [ ] Check Android build and test tasks
- [ ] Maintain reproducible environments

## Phase 6: Android Improvements
- [ ] Audit `MainActivity.kt` - separate UI, state, lifecycle, model management
- [ ] Fix documentation: no Local/Remote/Auto modes (per AGENTS.md)
- [ ] Release signing: require valid config or identify dev builds

## Phase 7: Documentation Overhaul
- [ ] Rewrite root README.md
- [ ] Rewrite pc/README.md
- [ ] Rewrite android/README.md

## Phase 8: Dependencies & Cleanup
- [ ] Inspect unused dependencies, obsolete scripts, duplicate files, temporary outputs
- [ ] Classify before deletion
- [ ] Preserve working GPU configs and model compatibility

## Phase 9: GitHub Profile
- [ ] Update repository description, topics, README navigation, badges
- [ ] Retain Sponsors configuration and links

## Phase 10: Verification & Git
- [ ] Run tests, lint, type checks, Android Gradle checks
- [ ] Final Desloppify scan
- [ ] Git diff review
- [ ] Commit: `refactor: improve Prism architecture, security and documentation`
- [ ] Push to correct repository and branch

## Phase 11: Final Report
- Architecture changes, security improvements, PC/Android changes, dependency changes, documentation changes, CI results, Desloppify improvements, commit SHA, remaining limitations

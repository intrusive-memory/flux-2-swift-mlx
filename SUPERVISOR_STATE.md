# SUPERVISOR_STATE — OPERATION SILICON STETHOSCOPE

> **Terminology**: A *mission* is the definable scope of work. A *sortie* is an atomic agent task within that mission. Work units are groupings of sorties by layer.

## Mission Metadata

| Field | Value |
|-------|-------|
| Operation name | OPERATION SILICON STETHOSCOPE |
| Mission | flux-2-swift-mlx-instrumentation |
| Iteration | 1 |
| Mission branch | `instrumentation/01` |
| Starting point commit | `3d7d64287f8eaeee2fa71cd545ee5951348c1656` (development @ "docs: add instrumentation requirements for Vinetas host") |
| Mission setup commit | `44b4a4c` (mission: launch OPERATION SILICON STETHOSCOPE) |
| Mission started | 2026-05-12 |
| max_retries | 3 |
| Default dispatch mode | dynamic (no template in plan) |

## Plan Summary

- Work units: 7
- Total sorties: 10 (1, 2, 3, 4, 5, 6, 7a, 7b, 8, 9, 10)
- Dependency structure: layered (1 → 2 → {3 ∥ 4 ∥ 5} → 6 → {7a ∥ 7b ∥ 8 ∥ 9} → 10)
- Critical path: 5 sorties (1 → 2 → 6 → 7a → 10)
- Parallel groups: {3, 4, 5}; {7a, 7b, 8} (max 3 concurrent sub-agents)
- Build restriction: sub-agents never run `make build` / `make test`; supervising agent handles all builds between groups

## Work Units

| Name | Directory | Sorties | Layer | Dependencies | State |
|------|-----------|---------|-------|--------------|-------|
| Telemetry types & protocol | `Sources/Flux2Core/Telemetry/` | 1 | 1 | none | RUNNING |
| Pipeline lock seam | `Sources/Flux2Core/{Pipeline,Loading,Scheduler,Transformer}/` | 2 | 2 | Sortie 1 | NOT_STARTED |
| Non-hot-path emissions | `Sources/Flux2Core/{Loading,Scheduler,Pipeline,VAE}/` | 3, 4, 5 | 3 | Sortie 2 | NOT_STARTED |
| Hot-path denoise emissions | `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` | 6 | 4 | Sorties 3, 4, 5 | NOT_STARTED |
| Functional tests | `Tests/Flux2CoreTests/` | 7a, 7b, 8 | 5 | Sortie 6 (7a/7b); Sortie 2 (8) | NOT_STARTED |
| Overhead test | `Tests/Flux2CoreTests/` | 9 | 5b | Sortie 6 | NOT_STARTED |
| Release | repo root | 10 | 6 | Sorties 7a/7b/8/9 | NOT_STARTED |

## Per-Work-Unit State

### Telemetry types & protocol
- Work unit state: RUNNING
- Current sortie: 1 of 1
- Sortie state: PENDING (about to dispatch)
- Sortie type: code
- Model: sonnet (complexity score = 7: ~12 turns + 2 new files + foundation_score 1 + risk 1 + 0 ambiguity)
- Complexity score: 7
- Attempt: 0 of 3
- Last verified: —
- Notes: First sortie of the mission. Foundation for all later sorties.

### Pipeline lock seam
- Work unit state: NOT_STARTED
- Current sortie: 2 of 1
- Sortie state: PENDING
- Sortie type: code
- Model: TBD (planned: opus — highest risk in campaign, 8 dependents)
- Complexity score: TBD
- Attempt: 0 of 3
- Notes: Highest-risk single piece. Touches 7 files. `@unchecked Sendable` is famously easy to misuse.

### Non-hot-path emissions
- Work unit state: NOT_STARTED
- Current sorties: 3, 4, 5 (parallel)
- Sortie state: PENDING
- Sortie type: code
- Model: TBD per sortie
- Complexity score: TBD
- Attempt: 0 of 3
- Notes: Three sub-agents in parallel after Sortie 2.

### Hot-path denoise emissions
- Work unit state: NOT_STARTED
- Current sortie: 6 of 1
- Sortie state: PENDING
- Sortie type: code
- Model: TBD (planned: opus — hot path, riskiest emission sortie)
- Complexity score: TBD
- Attempt: 0 of 3
- Notes: Pre-read line-range targeting required to keep context bounded.

### Functional tests
- Work unit state: NOT_STARTED
- Current sorties: 7a, 7b, 8 (parallel)
- Sortie state: PENDING
- Notes: 7b depends on 7a's MockTelemetryReporter artifact; 8 only needs Sortie 2.

### Overhead test
- Work unit state: NOT_STARTED
- Current sortie: 9
- Sortie state: PENDING
- Notes: Timing-sensitive — runs alone on ARM64 hardware.

### Release
- Work unit state: NOT_STARTED
- Current sortie: 10
- Sortie state: PENDING
- Notes: Target tag `v3.2.0` (next minor from current `3.1.1-dev`).

## Active Agents

| Work Unit | Sortie | Sortie State | Attempt | Model | Complexity Score | Task ID | Output File | Dispatched At |
|-----------|--------|-------------|---------|-------|-----------------|---------|-------------|---------------|
| _none yet_ | | | | | | | | |

## Decisions Log

| Timestamp | Work Unit | Sortie | Decision | Rationale |
|-----------|-----------|--------|----------|-----------|
| 2026-05-12 | — | — | Mission branch = `instrumentation/01` (not `mission/silicon-stethoscope/01`) | Plan explicitly fixes branch name; Sortie 10 references it for PR push |
| 2026-05-12 | — | — | Branch source = current HEAD on `development` (not `main`) | Development has 3.1.1-dev version bump; merges back through development→main |
| 2026-05-12 | — | — | Operation name = OPERATION SILICON STETHOSCOPE | Generated locally rather than via haiku sub-agent — pattern-based, well-defined task |
| 2026-05-12 | — | — | "Supervising agent only" interpreted as "single sortie agent, no parallel sub-agents" | skill.md wins on conflict: supervisor does NOT write production code. Plan's "supervising agent" sorties (1, 2, 6, 9, 10) still dispatch a Task agent — just never alongside others |
| 2026-05-12 | Telemetry types | 1 | Model: sonnet (score 7) | ~12 turns, 2 new files, foundation 1 (used by every later sortie), risk 1 (pure type addition). Score lands in sonnet band (6-12). Could be haiku, but foundation_importance adder pushes it up. |

## Overall Status

- Phase: Layer 1 (foundation)
- Sorties pending: 10 of 10
- Sorties completed: 0 of 10
- Last action: Mission initialized, Sortie 1 about to dispatch

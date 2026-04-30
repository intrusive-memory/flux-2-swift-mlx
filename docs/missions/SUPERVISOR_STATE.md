# SUPERVISOR_STATE.md — OPERATION FAREWELL EMBRACE

> **Terminology**: A *mission* is the definable scope of work. A *sortie* is an atomic agent task within that mission. A *work unit* groups related sorties.

## Mission Metadata

| Field | Value |
|---|---|
| Operation | OPERATION FAREWELL EMBRACE |
| Iteration | 1 |
| Mission branch | `mission/farewell-embrace/01` |
| Base branch | `development` |
| Starting point commit | `72a3eb6f3fe2b295e0eec5c4a8eb02ea1865e55f` |
| Started | 2026-04-30 |
| Plan file | `docs/missions/EXECUTION_PLAN.md` |
| Project root | `/Users/stovak/Projects/flux-2-swift-mlx` (git repo); plan + state under `docs/missions/` |

## Plan Summary

- Work units: 2
- Total sorties: 22 (13 in WU1, 9 in WU2)
- Dependency structure: layered — WU1 (CDN provisioning, sequential) gates WU2 from Sortie 16 onward; WU2 Sorties 14 + 15 are independent of WU1 and run alongside it
- Dispatch mode: dynamic (no template in plan)
- Max retries per sortie: 3
- Max simultaneous sorties: 2 (one WU1 ship/verify in flight + one WU2 prep sortie from {14, 15})

## Work Units

| Name | Directory | Sorties | Dependencies |
|---|---|---|---|
| WU1 — Acervo CDN Provisioning | `/Users/stovak/Projects/flux-2-swift-mlx` | 13 (Sorties 1–13) | none |
| WU2 — HF Excision + Code Migration | `/Users/stovak/Projects/flux-2-swift-mlx` | 9 (Sorties 14–22) | WU1 complete (all 11 manifests verified on CDN) before Sortie 16; Sorties 14 + 15 have no WU1 dep |

## Operator Environment Loader

All dispatched-agent commands MUST be wrapped in this loader prefix so existence checks pass and `acervo`/`curl` see the expected variables. Values are never echoed.

```sh
[ -z "$HF_TOKEN" ] && export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
[ -z "$R2_PUBLIC_URL" ] && export R2_PUBLIC_URL="$R2_ENDPOINT"
```

Reason: this operator's shell uses `hf` CLI login (token cached at `~/.cache/huggingface/token`) and the public CDN URL is stored in `R2_ENDPOINT` rather than `R2_PUBLIC_URL`.

## Per-Work-Unit State

### WU1 — Acervo CDN Provisioning
- Work unit state: RUNNING
- Current sortie: 2 of 13 (Sortie 1 COMPLETED at commit `ff42362`)
- Sortie state: PENDING → DISPATCHED (this iteration)
- Sortie type: command (operator-credentialed CLI: `acervo ship`)
- Model: sonnet
- Complexity score: 9 (mechanical ship; no foundation override — Sortie 2 is the first execution of the per-ship template)
- Attempt: 1 of 3
- Last verified: Sortie 1 — recon doc committed, license accepted, all four env vars present after loader prefix, acervo dry-run CHECK 4 passed
- Notes: First lmstudio-community ship. Per Sortie 1 finding, all lmstudio repos require `acervo ship --no-verify` (CHECK 1 returns 404 — lmstudio repos are not Git LFS-backed). Smallest non-gated payload (2 GB) — proves the shipping pipeline before scaling up.

### WU2 — HF Excision + Code Migration
- Work unit state: RUNNING
- Current sortie: 15 of 22 (Sortie 14 COMPLETED at commit `45cd8b2`)
- Sortie state: PENDING → DISPATCHED (this iteration)
- Sortie type: code
- Model: sonnet
- Complexity score: 13 (foundation for Sortie 21 parity assertion; one-shot fixture generator that depends on the OLD package's `Tokenizers` module — cannot be re-created after Sortie 16's swap)
- Attempt: 1 of 3
- Last verified: Sortie 14 — `make build` succeeded; FluxTextEncodersTests 125/125 + Flux2CoreTests 201/201 passed; Flux2GPUTests environmentally skipped (no `KLEIN_MODEL_PATH`). `Package.resolved` still pins swift-transformers as required.
- Notes: Captures golden encode/decode fixtures from OLD package; downstream Sortie 21 asserts equality against the NEW package. Must run BEFORE Sortie 16 (Package.swift swap).

## Active Agents

| Work Unit | Sortie | Sortie State | Attempt | Model | Complexity Score | Task ID | Output File | Dispatched At |
|---|---|---|---|---|---|---|---|---|
| WU1 | 2 | DISPATCHED | 1/3 | sonnet | 9 | (this iteration) | (this iteration) | 2026-04-30 |
| WU2 | 15 | DISPATCHED | 1/3 | sonnet | 13 | (this iteration) | (this iteration) | 2026-04-30 |

## Completed Sorties

| Work Unit | Sortie | Commit | Outcome | Notes |
|---|---|---|---|---|
| WU1 | 1 | `ff42362` | COMPLETED | Inventory + license probe; recon-cdn-inventory.md created. Commit also absorbed Sortie 14's staged deletions due to shared-index race (see Decisions Log). |
| WU2 | 14 | `45cd8b2` | COMPLETED | Library-only cleanup; `make build` + 326 tests passed; GPU suite environmentally skipped. |

## Decisions Log

| Timestamp | Work Unit | Sortie | Decision | Rationale |
|---|---|---|---|---|
| 2026-04-30 | — | — | Mission base = `development`, not `main` | Per operator direction; matches release flow (development → main). Plan's pre-flight note referencing `main` is stale (marching-relay PR landed). |
| 2026-04-30 | — | — | Operation name = OPERATION FAREWELL EMBRACE | THE RITUAL. Riffs on "HuggingFace" → "embrace"; signals migration off it. |
| 2026-04-30 | — | — | Loader prefix injects `HF_TOKEN` from `~/.cache/huggingface/token` and `R2_PUBLIC_URL` from `R2_ENDPOINT` | Operator uses `hf` CLI login (no env-var export) and the public CDN URL is stored under `R2_ENDPOINT`. Loader maps these without echoing values. |
| 2026-04-30 | WU1 | 1 | Model: sonnet | Complexity 12 (recon + CLI probes; clear exit criteria; type=command modifier −3). Foundation override considered but rejected — Sortie 1 is verification, not architectural. |
| 2026-04-30 | WU2 | 14 | Model: sonnet | Score 19 would force opus by algorithm; override applied because work is mechanical deletion + build verification (no architectural decisions). Sonnet is sufficient and 3× cheaper than opus. |
| 2026-04-30 | WU1 | 1 | Recon finding: lmstudio-community repos require `acervo ship --no-verify` | These repos are NOT Git LFS-backed, so CHECK 1 returns 404. Applies to Sorties 2, 3, 4, 6, 8, 9, 10. The non-lmstudio ships (5, 7, 11, 12) MAY still need verification — verify per-sortie. |
| 2026-04-30 | — | — | Lesson: parallel sorties on the same git tree caused commit cross-contamination | Sortie 14 staged deletions via `git rm`; Sortie 1's `git commit` (no path restriction) absorbed them. Final tree state is correct but commits are mis-attributed. **Mitigation for future sorties**: dispatch prompts MUST instruct path-restricted commits (`git commit -- <paths>`). Considered worktree isolation but rejected — separate-branch commits would not land on the mission branch without merge friction. |
| 2026-04-30 | WU1 | 2 | Model: sonnet | Score 9 (mechanical ship; clear exit criteria; type=command modifier −3). First ship, smallest payload — proves the pipeline. |
| 2026-04-30 | WU2 | 15 | Model: sonnet | Score 13 — foundation for parity assertion. One-shot generator depending on OLD package APIs that vanish after Sortie 16. |

## Overall Status

OPERATION FAREWELL EMBRACE — IN PROGRESS. 2 of 22 sorties complete.

This iteration dispatching:
- WU1 Sortie 2 (ship `lmstudio-community/Qwen3-4B-MLX-4bit` — 2 GB; uses `--no-verify`)
- WU2 Sortie 15 (capture golden tokenizer fixtures from OLD package)

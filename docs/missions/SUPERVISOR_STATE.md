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
- Current sortie: 1 of 13
- Sortie state: DISPATCHED
- Sortie type: command
- Model: sonnet
- Complexity score: 12
- Attempt: 1 of 3
- Last verified: not yet
- Notes: Sortie 1 is the inventory + license-probe gate. Cheap to run; foundation for all 21 downstream sorties.

### WU2 — HF Excision + Code Migration
- Work unit state: RUNNING
- Current sortie: 14 of 22 (Sortie 14 dispatches now; Sorties 16–22 gated on WU1 complete)
- Sortie state: DISPATCHED
- Sortie type: code
- Model: sonnet
- Complexity score: 19 (foundation override condition met but mechanical work — see Decisions Log)
- Attempt: 1 of 3
- Last verified: not yet
- Notes: Library-only cleanup; deletes CLI targets and customModelsDirectory. Independent of WU1.

## Active Agents

| Work Unit | Sortie | Sortie State | Attempt | Model | Complexity Score | Task ID | Output File | Dispatched At |
|---|---|---|---|---|---|---|---|---|
| WU1 | 1 | DISPATCHED | 1/3 | sonnet | 12 | (pending dispatch) | (pending) | 2026-04-30 |
| WU2 | 14 | DISPATCHED | 1/3 | sonnet | 19 | (pending dispatch) | (pending) | 2026-04-30 |

## Decisions Log

| Timestamp | Work Unit | Sortie | Decision | Rationale |
|---|---|---|---|---|
| 2026-04-30 | — | — | Mission base = `development`, not `main` | Per operator direction; matches release flow (development → main). Plan's pre-flight note referencing `main` is stale (marching-relay PR landed). |
| 2026-04-30 | — | — | Operation name = OPERATION FAREWELL EMBRACE | THE RITUAL. Riffs on "HuggingFace" → "embrace"; signals migration off it. |
| 2026-04-30 | — | — | Loader prefix injects `HF_TOKEN` from `~/.cache/huggingface/token` and `R2_PUBLIC_URL` from `R2_ENDPOINT` | Operator uses `hf` CLI login (no env-var export) and the public CDN URL is stored under `R2_ENDPOINT`. Loader maps these without echoing values. |
| 2026-04-30 | WU1 | 1 | Model: sonnet | Complexity 12 (recon + CLI probes; clear exit criteria; type=command modifier −3). Foundation override considered but rejected — Sortie 1 is verification, not architectural. |
| 2026-04-30 | WU2 | 14 | Model: sonnet | Score 19 would force opus by algorithm; override applied because work is mechanical deletion + build verification (no architectural decisions). Sonnet is sufficient and 3× cheaper than opus. |

## Overall Status

OPERATION FAREWELL EMBRACE — INITIALIZED. Two sorties dispatched in parallel:
- WU1 Sortie 1 (inventory lock + HF license probe)
- WU2 Sortie 14 (library-only cleanup)

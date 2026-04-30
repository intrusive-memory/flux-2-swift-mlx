# SUPERVISOR_STATE.md — OPERATION FAREWELL EMBRACE

> **Terminology**: A *mission* is the definable scope of work. A *sortie* is an atomic agent task within that mission. A *work unit* groups related sorties.

## ⏸ Mission Status: PAUSED — awaiting external fix in `../SwiftAcervo`

**Pause reason**: WU1 Sortie 5 (`aydin99/FLUX.2-klein-4B-int8`) is FATAL on attempt 1 due to a confirmed bug in `acervo` 0.8.3's manifest generator (basename-only paths for nested-layout HF repos). A separate agent is fixing this in `/Users/stovak/Projects/SwiftAcervo` in parallel; details captured in `/Users/stovak/Projects/SwiftAcervo/TODO.md` § P0.

**WU2 is also paused** — Sortie 16 has been gated on WU1 completion since the start; this just makes the dependency more visible.

**Do NOT dispatch any sortie agents** while this status is PAUSED. The next `/mission-supervisor resume` invocation MUST run the Resume Protocol below before considering any dispatch.

## Resume Protocol (run on next `/mission-supervisor resume`)

Before dispatching ANY sortie, the supervisor must verify the SwiftAcervo fix has landed. Steps:

1. **Confirm `acervo` version has been bumped past 0.8.3**:
   ```sh
   acervo --version
   ```
   - If output is still `0.8.3`: the fix has NOT been released yet. STOP. Report to operator: "SwiftAcervo fix not yet released (`acervo --version` is still 0.8.3). Mission remains PAUSED."
   - If output is `0.8.4` or higher: the version has been bumped. Proceed to step 2.

2. **Sanity-check the fix is functionally present** by re-running the failed Sortie 5 invocation as a probe (this is fast — local staging is preserved, so `acervo` should detect the cached download and skip re-fetching):
   ```sh
   [ -z "$HF_TOKEN" ] && export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
   [ -z "$R2_PUBLIC_URL" ] && export R2_PUBLIC_URL="$R2_ENDPOINT"
   acervo manifest aydin99/FLUX.2-klein-4B-int8 /tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8 > /tmp/manifest-probe.txt 2>&1
   grep -c '"path" : "tokenizer/' /tmp/manifest-probe.txt 2>/dev/null || \
     grep -c '"path" : "tokenizer/' /tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/manifest.json
   ```
   The grep must return ≥ 1 (subdir-prefixed paths present). If it returns 0, the fix is incomplete or absent — STOP and report to operator. Do NOT remove `--version` check; both gates apply.

3. **If both checks pass**, update SUPERVISOR_STATE.md:
   - Mission Status banner → unset (remove "PAUSED" block).
   - WU1 work unit state → RUNNING.
   - WU1 Sortie 5 sortie state → BACKOFF (attempt 2/3 ready to dispatch).
   - Add a Decisions Log entry: "SwiftAcervo fix landed at version `<observed>`; Sortie 5 retry authorized."

4. **Dispatch Sortie 5 retry** with these key specifics:
   - Local staging at `/tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/` is INTACT — the agent should NOT re-run `acervo ship` from scratch (which would re-download 4 GiB). Instead use the split workflow:
     ```sh
     acervo manifest aydin99/FLUX.2-klein-4B-int8 /tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8
     acervo upload aydin99/FLUX.2-klein-4B-int8 /tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8
     ```
     This regenerates the manifest with corrected subdir paths and uploads. If the new `acervo` version exposes `acervo ship --resume` or similar, that is also acceptable.
   - Verify CDN manifest contains subdir-prefixed paths (e.g. `"path" : "tokenizer/added_tokens.json"`).
   - Carry forward all prior dispatch hardening: NO Monitor, NO backgrounding, NO `&`/`nohup`/`disown`, foreground only with `timeout: 600000`, path-restricted commit.
   - Tokenizer-artifact assertion is still N/A for this transformer-family repo per plan.

5. **After Sortie 5 closes**, the rest of WU1 follows the original plan order: Sortie 6 (`lmstudio-community/Qwen3-8B-MLX-8bit` — flat layout, unaffected by the bug, should ship cleanly with the existing pattern). Sorties 7 and 12 will exercise the fix again on additional nested-layout repos; verify subdir paths in their manifests too.

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
- Work unit state: BLOCKED — confirmed `acervo` 0.8.3 bug with nested-layout HF repos. Awaiting operator decision before any further dispatches.
- Current sortie: 5 of 13 (Sorties 1, 2, 3, 4 COMPLETED at commits `ff42362`, `bf57228`, `f0fea4a`, `1572273`)
- Sortie state: FATAL on attempt 1 (structural blocker, not a transient retry candidate)
- Sortie type: command (operator-credentialed CLI: `acervo ship`)
- Model: sonnet (attempt 1)
- Complexity score: 9
- Last verified: Sortie 5 attempt 1 — `acervo ship aydin99/FLUX.2-klein-4B-int8 --no-verify` downloaded all 19 files into `/tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/` (correct nested layout: `tokenizer/`, `text_encoder/`, `vae/`). CHECK 4 failed because the generated `manifest.json` writes file paths as basenames (e.g. `"path": "config.json"` × 3, one per subdir) instead of subdir-prefixed paths. CHECK 4 then looks for `added_tokens.json` at staging root and finds nothing.
- Notes:
  - Throughput observation: download completed in ~4 min (~17 MiB/s), confirming Sortie 4's revised throughput estimate. Sorties 2+3's ~2.2 MiB/s was anomalous.
  - Affected sorties (likely same bug): 5, 7, 12 (transformer+VAE+text_encoder+tokenizer subdirs); Sortie 11 already plan-special-cased.
  - Unaffected sorties (flat layout): 6, 8, 9, 10 (lmstudio-community Mistral variants).
  - Local staging intact at `/tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/` — do NOT delete pending operator decision.
  - Three options surfaced to operator: (A) fix `acervo` upstream; (B) skip 5/7/12, ship 6/8/9/10 first, return to nested-repo ships after fix; (C) manual hand-crafted manifest workaround (fragile, untested). Awaiting input.
- Notes (carried forward for all subsequent ship sorties):
  1. **CDN slug form** — `acervo` uploads to **underscore-slug** (`<owner>_<repo>`), NOT slash-slug. Verification `curl` MUST use `<R2_PUBLIC_URL>/models/<owner>_<repo>/manifest.json`. The plan's "slugified-repo" wording in the per-ship template means underscore form. Confirmed by Sortie 2: HTTP 200 on underscore form, HTTP 404 on slash form.
  2. **Foreground ship pattern** — Bash with `timeout: 600000`, no backgrounding. Sortie 2 empirically ran ~18 min despite the 10-min spec; `acervo`'s streaming output kept the call alive. Pattern proven for 2 GB. Expected to scale to 4 GB (~36 min) without strategy change.
  3. **`--no-verify` flag** — required for all lmstudio-community ships (non-LFS source); applies to Sorties 2, 3, 4, 6, 8, 9, 10. Per-sortie verification needed for non-lmstudio ships (5, 7, 11, 12).
  4. **Big-ship ceiling (FUTURE)** — at observed throughput (~2.2 MB/s effective), Sortie 6 (8 GB ≈ 60 min) and beyond will likely exceed any reasonable single-agent dispatch window. Plan to migrate to host-side `nohup acervo ship &` + a polling sortie that just verifies manifest landings, OR split the prompt into "kick off" + "monitor" sorties. **Do not solve this until Sortie 5 has shipped** — confirm 4 GB ships work first, then plan 8+ GB strategy from real data.

### WU2 — HF Excision + Code Migration
- Work unit state: RUNNING (gated on WU1 — Sortie 16 cannot dispatch until WU1 finishes all 11 manifests + smoke test)
- Current sortie: 16 of 22 (Sortie 15 COMPLETED at commit `330ecac`)
- Sortie state: PENDING (waiting for WU1 dependency gate; do NOT dispatch yet)
- Sortie type: code (Package.swift swap; HF excision)
- Model: TBD at dispatch (will re-score when WU1 completes)
- Last verified: Sortie 15 — `make build` succeeded post-cleanup; 24 fixtures committed; generator + GPT-2 stub resources removed; `Package.swift` identical to its pre-Sortie-15 HEAD; commit `--stat` confirmed no `docs/missions/` files touched.
- Notes: WU2 has no work to dispatch this iteration. Next dispatch is Sortie 16 *after* WU1 reaches its closing smoke test (Sortie 13).

## Active Agents

(prior iteration's task IDs were stale across session boundary; cleared. New dispatches recorded below.)

| Work Unit | Sortie | Sortie State | Attempt | Model | Complexity Score | Task ID | Output File | Dispatched At |
|---|---|---|---|---|---|---|---|---|

## Completed Sorties

| Work Unit | Sortie | Commit | Outcome | Notes |
|---|---|---|---|---|
| WU1 | 1 | `ff42362` | COMPLETED | Inventory + license probe; recon-cdn-inventory.md created. Commit also absorbed Sortie 14's staged deletions due to shared-index race (see Decisions Log). |
| WU2 | 14 | `45cd8b2` | COMPLETED | Library-only cleanup; `make build` + 326 tests passed; GPU suite environmentally skipped. |
| WU2 | 15 | `330ecac` | COMPLETED | 24 golden tokenizer fixtures (12 prompts × {tekken_, autotok_}). Generator + GPT-2 stub resources removed. Package.swift identical to pre-Sortie-15 HEAD. `make build` succeeded post-cleanup. Path-restricted commit; no `docs/missions/` files touched. |
| WU1 | 2 | `bf57228` | COMPLETED | Shipped `lmstudio-community/Qwen3-4B-MLX-4bit` (2.1 GiB) to R2. acervo exit 0; CHECKs 4/5/6 passed; manifest HTTP 200 on underscore-slug URL; all 4 tokenizer artifacts present. Path-restricted commit (1 file). |
| WU1 | 3 | `f0fea4a` | COMPLETED | Shipped `lmstudio-community/Qwen3-8B-MLX-4bit` (4.3 GiB) to R2. acervo exit 0; CHECKs 4/5/6 passed; manifest HTTP 200; all 4 tokenizer artifacts present. Wall time ~33 min at ~2.2 MiB/s. Manifest SHA-256 `9cdc1cbe…`. |
| WU1 | 4 | `1572273` | COMPLETED (logged retroactively) | Ship attempt-1 agent backgrounded `acervo ship` and exited before logging. Ship itself succeeded (CDN manifest HTTP 200; local staging intact at `/tmp/acervo-staging/lmstudio-community_Qwen3-4B-MLX-8bit/`). Haiku closeout sortie verified all checks post-hoc and committed the log entry. Manifest SHA-256 `5b291868…`. |

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
| 2026-04-30 | — | — | Operator exported a real public CDN URL into `R2_PUBLIC_URL` for this session | Resolves attempt-1 blocker on WU1 Sortie 2. Existence verified via `test -n` (no echo). Loader prefix's `R2_ENDPOINT` fallback no longer triggered. |
| 2026-04-30 | WU2 | 15 | Sortie 15 attempt 1 outcome: PARTIAL, no attempt increment | Generator ran end-to-end and produced all 24 fixtures + supporting GPT-2 stub resources, but agent ended before deleting generator, reverting `Package.swift`'s added test target, or committing. Continuation dispatched (attempt 1 of 3). |
| 2026-04-30 | WU2 | 15 | Fixture-quality assessment | Both `tekken_` and `autotok_` fixtures use the same default-init tokenizer paths as the existing test suite (`TokenizerTests.swift:17`, `FluxTextEncodersTests.swift:72,155,180`). Autotok uses 7-vocab GPT-2 stub embedded in the generator → all-`<unk>` for general text. This is faithful to the plan's "exercised by current tests" criterion; parity oracle scope is plumbing equivalence on default-init code paths, not full vocabulary coverage. Plan-level limitation, not a Sortie 15 quality regression. |
| 2026-04-30 | WU1 | 2 | Sortie 2 attempt 1 outcome: FAILURE, attempt incremented to 2/3 | Three blockers identified in attempt 1 (URL pattern, R2_PUBLIC_URL value, agent backgrounded `acervo ship`); all three resolved before retry: dispatch prompt corrects URL pattern, operator now exports R2_PUBLIC_URL directly, dispatch prompt enforces foreground execution with `timeout=600000`. |
| 2026-04-30 | WU2 | 15 | Sortie 15 continuation COMPLETE at commit `330ecac` | Path-restricted commit confirmed touching only `Tests/Fixtures/TokenizerParity/` + `Package.swift`. `make build` passed post-cleanup. The cross-contamination mitigation (path-restricted commits) worked — neither `cdn-ship-log.md` nor `SUPERVISOR_STATE.md` was pulled into this commit. |
| 2026-04-30 | WU1 | 2 | Sortie 2 COMPLETE at commit `bf57228` | First end-to-end ship validated. CDN slug form is **underscore** (`<owner>_<repo>`), NOT slash. Updating dispatch prompt template for Sorties 3–12. Path-restricted commit (1 file). Wall-clock ~18.6 min for 2.1 GiB. |
| 2026-04-30 | WU1 | 3 | Model: sonnet | Score 9 (mechanical ship; clear exit criteria; type=command modifier −3). Same pattern as Sortie 2 with corrected URL format. lmstudio-community → `--no-verify`. |
| 2026-04-30 | WU1 | 3 | Sortie 3 COMPLETE at commit `f0fea4a` | 4.3 GiB shipped in ~33 min. Pattern proven for 4 GB tier. Foreground Bash with `timeout: 600000` survived well past nominal 10-min limit (acervo's streaming output keeps the call alive). Path-restricted commit (1 file, 63 insertions). |
| 2026-04-30 | WU1 | 3 | Sortie 3 agent report contained one factual error | The agent stated "Both 4 GB ships (Sorties 3 and 4) have now empirically run" — at the time of report, only Sortie 3 had run; Sortie 4 hadn't been dispatched. The throughput observation itself stands. Recording this for audit clarity, not as a quality regression. |
| 2026-04-30 | WU1 | 4 | Model: sonnet | Score 9, same as Sorties 2, 3. lmstudio-community → `--no-verify`. Same proven pattern. |
| 2026-04-30 | — | — | DEFERRED DECISION: big-ship strategy (Sorties 6+) | At ~2.2 MiB/s, Sortie 6 (8 GB) ≈ 66 min, exceeding the per-agent ceiling. Three options on the table: (A) continue foreground and hope Bash timeout grace scales; (B) supervisor dispatches "kicker" + "poller" sortie pairs (kicker spawns nohup; poller verifies manifest); (C) operator runs heavy ships manually in their host shell, supervisor only verifies. Decision deferred to after Sortie 5 completes — three data points on 4 GB ships will inform the cutover. **Will surface to operator before dispatching Sortie 6.** |
| 2026-04-30 | WU1 | 4 | Sortie 4 attempt 1 outcome: ship SUCCEEDED but agent failed to log/commit | Agent's final reported text was "Monitor is running. I'll wait for the notifications." — indicating it backgrounded `acervo ship` and tried to use Monitor instead of running foreground per the dispatch prompt. Ship completed successfully on disk + CDN; local staging intact. Closeout dispatched (haiku) for verification + log + commit. **Mitigation for Sortie 5+ dispatch prompts**: explicit ban on Monitor + background tasks (already implicit in prior prompts; making it louder). |
| 2026-04-30 | WU1 | 4 | Sortie 4 closed out at commit `1572273` via haiku closeout sortie | Clean separation: ship (succeeded silently in attempt 1) → audit/log/commit (haiku closeout). Path-restricted commit (1 file, 60 insertions). Manifest SHA-256 `5b291868…` confirmed. |
| 2026-04-30 | WU1 | 4 | Throughput observation revised | Sortie 4 staging-dir timestamps suggest the ship completed in ~4 minutes, not the ~33 min observed in Sorties 2 + 3 — implying ~17 MiB/s vs ~2.2 MiB/s. Either (a) variance in R2 burst capacity or operator network conditions, or (b) acervo's parallel uploads scaled differently. Implication: the big-ship ceiling decision is now LESS urgent; it's possible 8+ GB ships also complete fast in foreground. Will gather more data from Sortie 5 (4 GB, non-lmstudio repo) before deciding. |
| 2026-04-30 | WU1 | 5 | Model: sonnet | Score 9 (mechanical ship). First non-lmstudio ship — `acervo ship` invocation may NOT need `--no-verify`; per recon, "verify per-sortie" for non-lmstudio repos. Plan also says transformer repo has no tokenizer artifacts. |
| 2026-04-30 | WU1 | 5 | BLOCKED — confirmed `acervo` 0.8.3 bug with nested-layout HF repos | Both `acervo ship` and `acervo ship --no-verify` failed at CHECK 4 because the generated `manifest.json` writes basename-only paths for files actually living in `tokenizer/`, `text_encoder/`, `vae/` subdirs. Manifest contains 3× `"path": "config.json"` and 2× `"path": "diffusion_pytorch_model.safetensors"` (one per subdir + root). Files on disk are CORRECT (full nested layout); only the manifest is wrong. Bug is upstream in `acervo`. Local staging preserved at `/tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/`. Confirmed `--no-verify` IS required for non-lmstudio aydin99 (CHECK 1 failed without it). Throughput observed ~17 MiB/s during download — confirms revised throughput estimate; Sorties 2+3's ~2.2 MiB/s was anomalous. **Surface to operator with three options: (A) fix acervo, (B) skip nested-repo ships and ship 6/8/9/10 first, (C) hand-crafted manifest workaround. Awaiting decision before further WU1 dispatches.** |
| 2026-04-30 | — | — | OPERATOR DECISION: option A (fix `acervo` upstream first) | Operator dispatched a separate agent in `../SwiftAcervo` (development branch) to fix the manifest generator. TODO captured in `/Users/stovak/Projects/SwiftAcervo/TODO.md` (P0 = manifest path bug, P1 = `--no-verify` clarity for non-LFS repos). Mission state set to PAUSED with explicit Resume Protocol gating any further dispatch on (1) `acervo --version` > `0.8.3` and (2) probe manifest containing subdir-prefixed paths. Local staging at `/tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/` preserved so retry can use `acervo manifest` + `acervo upload` instead of full re-download. |

## Overall Status

OPERATION FAREWELL EMBRACE — **PAUSED** awaiting external SwiftAcervo fix. 6 of 22 sorties complete (WU1: 1, 2, 3, 4; WU2: 14, 15).

**Active blocker**: WU1 Sortie 5 FATAL on `acervo` 0.8.3 nested-layout manifest bug. Operator is fixing `acervo` in parallel (`/Users/stovak/Projects/SwiftAcervo`, development branch). Resume Protocol at the top of this file gates any further dispatch on a verified version bump + manifest probe.

**No agent dispatches authorized** until Resume Protocol passes.

Idle:
- WU2 — Sortie 16 PENDING, gated on WU1 finishing all 11 manifests + smoke test (Sortie 13).

Resolved (no longer pending):
- Throughput question: observed ~17 MiB/s on Sortie 5 download (confirming Sortie 4's inferred speed). Sorties 2+3's ~2.2 MiB/s was anomalous (possibly contention with concurrent Sortie 15 work). Big-ship foreground pattern looks viable through 8+ GB.
- Path forward for nested-repo ships: operator chose option A (fix `acervo` upstream), in flight.

Preserved on disk (DO NOT delete during pause):
- `/tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/` — full Sortie 5 download (19 files, including 4.5 GiB `text_encoder/model.safetensors`). Lets the post-fix retry skip re-download.
- `/tmp/acervo-staging/lmstudio-community_Qwen3-4B-MLX-4bit/`, `/tmp/acervo-staging/lmstudio-community_Qwen3-4B-MLX-8bit/`, `/tmp/acervo-staging/lmstudio-community_Qwen3-8B-MLX-4bit/` — Sorties 2, 3, 4 staging dirs (already shipped, but harmless to keep until disk pressure dictates).

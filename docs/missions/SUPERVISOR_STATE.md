# SUPERVISOR_STATE.md — OPERATION FAREWELL EMBRACE

> **Terminology**: A *mission* is the definable scope of work. A *sortie* is an atomic agent task within that mission. A *work unit* groups related sorties.

## ⏸ Mission Status: PAUSED — Sortie 5 upload incomplete; needs operator-driven resumption

The 0.8.4 fix is installed and functionally verified (Resume Protocol from prior iteration passed). Sortie 5 retry's `acervo upload` was killed at the 10-min Bash timeout with only 1.4 of 8.3 GiB uploaded (~17%). CDN probe confirms NONE of the model files (`text_encoder/model.safetensors`, root `diffusion_pytorch_model.safetensors`) reached R2 — only small `.cache/` lock/metadata files did. The local staging at `/tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/` is still intact.

**Two related findings, both important for Sortie 6+ planning:**

1. **The repo is genuinely 8.3 GiB**, not 4 GiB as the plan listed. `du -sh` confirms: 3.6 GB root `diffusion_pytorch_model.safetensors` + 4.5 GB `text_encoder/` + 160 MB `vae/` + 15 MB `tokenizer/`. The plan's "4 GB" estimate referred to one of the safetensors files, not the full repo.
2. **Sustained CDN throughput is ~2.3 MiB/s on the big files** (`Completed 1.4 GiB/8.3 GiB (2.3 MiB/s)` per the upload's last log line). The earlier "~17 MiB/s" inference for Sortie 4 came from misreading staging-dir timestamps — the manifest.json's mtime is when local CHECK 3 finishes, not when the upload completes. So Sorties 2 + 3's ~2.2 MiB/s observation was correct all along, and the optimistic recalibration was wrong. **Big-ship ceiling is real**: at 2.3 MiB/s, 8 GiB ≈ 60 min, 32 GiB ≈ 4 hours — well beyond any single-agent dispatch window.

**Why my dispatch was the wrong shape for this payload:** the prompt's ban on Monitor + backgrounding made sense for the lmstudio-community 2–4 GB ships (which all completed within the 10-min Bash window). For an 8.3 GiB payload at 2.3 MiB/s, the 10-min cap is structural — the upload needs ~60 min wall-clock. The Bash tool returned a "still running" indicator at the 10-min mark, the agent (correctly per its instructions) treated that as a violation and stopped, and the underlying upload was then killed when the agent exited.

**Operator path forward (recommendation):** run `acervo upload aydin99/FLUX.2-klein-4B-int8 /tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8` manually in your host shell (regular terminal, no timeout). `aws s3 sync` underneath should be idempotent — it'll skip the small lock/metadata files that already uploaded, then push the big model files. ETA ~50 min. When complete, run `/mission-supervisor resume` and the supervisor will dispatch a haiku closeout sortie (verify CDN, append log, commit) — same pattern as Sortie 4's closeout.

**Alternate path (untested) for the supervisor to handle Sortie 6+ autonomously:** dispatch with Monitor allowed, where the agent kicks the upload, accepts the Bash timeout return, and uses Monitor to wait for the underlying bash task to complete. This depends on whether the Bash tool's 10-min timeout actually kills the process or just stops streaming. Sortie 5's evidence suggests it kills (no `acervo`/`aws` processes were running after the agent exited), so this path may not work without `nohup` + `disown` + `setsid` to fully detach. NOT recommended for Sortie 5 — operator-manual is faster and more reliable.

Resume Protocol section retained below for audit / future-mission reference.

## Resume Protocol (run on next `/mission-supervisor resume` if a future pause occurs)

Before dispatching ANY sortie, the supervisor must verify the SwiftAcervo fix has landed. Steps:

1. **Confirm `acervo` version is at least 0.8.4** (the planned fix release):
   ```sh
   acervo --version
   ```
   - If output is `0.8.3` or older: the fix has NOT been released yet. STOP. Report to operator: "SwiftAcervo 0.8.4 not yet released (`acervo --version` reads `<observed>`). Mission remains PAUSED."
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
- Work unit state: BLOCKED — Sortie 5 upload partially completed then killed at 10-min Bash timeout. Awaiting operator-driven manual upload before any further dispatch.
- Current sortie: 5 of 13 (Sorties 1, 2, 3, 4 COMPLETED at commits `ff42362`, `bf57228`, `f0fea4a`, `1572273`)
- Sortie state: PARTIAL (attempt 2 — supervisor + agent verified env + manifest, agent kicked `acervo upload`, Bash timed out at 10 min, agent reported correctly per instructions, underlying upload was killed when agent exited)
- Sortie type: command (operator-credentialed CLI: `acervo upload`)
- Model: sonnet (attempts 1 + 2 both sonnet)
- Complexity score: 9 — but the architectural mismatch (10-min Bash timeout vs ~60-min upload time) means this sortie is structurally too large for the foreground dispatch pattern. NOT a model/agent quality issue.
- Last verified (after attempt 2):
  - `acervo --version` is `0.8.4`. Manifest is correct (subdir paths). CHECK 4 passed (per upload output line 1).
  - Upload reached "Completed 1.4 GiB / 8.3 GiB (2.3 MiB/s) with 17 file(s) remaining" before being killed.
  - CDN probe: `manifest.json` HTTP 404; `text_encoder/model.safetensors` HTTP 404; root `diffusion_pytorch_model.safetensors` HTTP 404. **None of the actual model files reached R2.** The 1.4 GiB transferred was lock/metadata files in `.cache/huggingface/download/` plus partial uploads of the small files in `vae/` + `tokenizer/`.
  - Local staging at `/tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/` is intact (8.3 GiB on disk, all 19 files in nested layout, manifest with subdir paths).
  - Bash output preserved at `/private/tmp/claude-501/-Users-stovak-Projects-flux-2-swift-mlx/cf6ab61c-03be-4ea0-97ff-07aadd6c2a1e/tasks/bs640w3eb.output` (110 KB).
- Notes / next steps:
  - Recommended path: operator runs `acervo upload aydin99/FLUX.2-klein-4B-int8 /tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8` in their host shell. `aws s3 sync` is idempotent → resumes from where attempt 2 stopped. ETA ~60 min wall time (8.3 GiB × 2.3 MiB/s ≈ 3700s).
  - When complete, `/mission-supervisor resume` will see CDN manifest at HTTP 200 and dispatch a haiku closeout sortie (verify subdir paths, append ship log, path-restricted commit) — same shape as Sortie 4's closeout.
  - **Big-ship strategy decision is now FORCED for Sortie 6+** (8 GB+ ships): foreground-agent pattern is structurally infeasible. See "Pending operator decision" in Overall Status.
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
| 2026-04-30 | — | — | EXECUTION_PLAN.md SwiftAcervo dep version bumped 0.8.3 → 0.8.4 | Per operator direction. The fix-release version is 0.8.4. Five references updated (lines 25, 38, 431, 671, 702). Resume Protocol step 1 tightened from `> 0.8.3` to `>= 0.8.4` for clarity. **Note**: this is one of the few places the supervisor edits EXECUTION_PLAN.md during execution (normally forbidden); doing so under explicit operator direction. No semantic change to the plan — just locks the floor version to the fix release. |
| 2026-04-30 | — | — | Resume Protocol PASSED — `acervo` 0.8.4 verified locally | Step 1: `acervo --version` returns `0.8.4`. Step 2: regenerated manifest at `/tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/manifest.json` now contains 18 distinct subdir-prefixed paths (`text_encoder/…` × 4, `tokenizer/…` × 7, `vae/…` × 2) and exactly 1 entry for `config.json` (was 3 in the broken version). Mission unblocked. WU1 → RUNNING. Sortie 5 → BACKOFF (attempt 2/3). |
| 2026-04-30 | WU1 | 5 | Sortie 5 retry strategy: skip download AND manifest, go straight to `acervo upload` | Supervisor's Resume Protocol probe already regenerated the manifest correctly. Re-running `acervo manifest` would be idempotent but adds ~30s for nothing. Re-running `acervo ship` from scratch would re-download 4 GiB. Going straight to `acervo upload <repo> <staging_dir>` is the cheapest correct path. |
| 2026-04-30 | WU1 | 5 | Sortie 5 retry attempt 2 outcome: PARTIAL upload, then killed at 10-min Bash timeout | Agent did everything right per dispatch (verified env + manifest + df, kicked `acervo upload`, captured early CHECK 4 pass). Upload reached 1.4 of 8.3 GiB at ~2.3 MiB/s; Bash returned "still running in background" indicator at 10-min timeout; agent stopped + reported per the (in-hindsight overly strict) ban on Monitor/backgrounding; underlying upload process was then killed when the agent exited. CDN probe: 0 model files reached R2. Local staging intact. Repo is genuinely 8.3 GiB (3.6 GB root + 4.5 GB text_encoder + 160 MB vae); plan's "4 GB" estimate referred to one safetensors file, not the full repo. |
| 2026-04-30 | WU1 | 5 | Throughput recalibration: ~2.3 MiB/s sustained on big files | Earlier "~17 MiB/s" inference for Sortie 4 was wrong — it came from misreading manifest.json's mtime (CHECK 3 completion, not upload completion). Sorties 2 + 3's ~2.2 MiB/s observation was correct all along. **At ~2.3 MiB/s, big-ship ceiling is real**: Sortie 6 (8 GB) ≈ 60 min, Sortie 11 (32 GB) ≈ 4 hours. Foreground-agent pattern is structurally infeasible for ships > ~1.5 GiB at this throughput. |
| 2026-04-30 | — | — | DISPATCH DESIGN ERROR: ban on Monitor/backgrounding was too strict for big payloads | Sortie 5 retry's prompt explicitly forbade Monitor + any backgrounding mechanism. Was correct for the lmstudio-community 2–4 GB ships (which fit within 10-min Bash window). Wrong for the 8.3 GiB aydin99 payload, which structurally needs ~60 min wall time. Mitigation: Sortie 6+ dispatch will need a different shape (operator-manual, or kicker-poller pattern). DO NOT carry the strict ban forward to big-ship dispatches. |
| 2026-04-30 | WU1 | 5 | Sortie 5 → BLOCKED, awaiting operator-manual upload | Recommended path: operator runs `acervo upload aydin99/FLUX.2-klein-4B-int8 /tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8` in host shell (no timeout). aws s3 sync is idempotent — resumes from attempt 2's partial state. When CDN manifest goes HTTP 200, `/mission-supervisor resume` will dispatch a haiku closeout (verify + log + commit). |

## Overall Status

OPERATION FAREWELL EMBRACE — **PAUSED** awaiting operator-manual upload of Sortie 5 staging. 6 of 22 sorties complete (WU1: 1, 2, 3, 4; WU2: 14, 15).

**No agent dispatches authorized** until Sortie 5 manifest goes HTTP 200 on CDN.

Idle:
- WU2 — Sortie 16 PENDING, gated on WU1 finishing all 11 manifests + smoke test (Sortie 13).

Resolved:
- `acervo` 0.8.4 fix: installed and verified — manifest generator produces correct subdir-prefixed paths.
- Path forward for nested-repo manifests: operator chose option A (fix `acervo` upstream), landed.

Pending operator decision (now FORCED — was deferred):
- **Big-ship strategy for Sorties 6+**. At measured ~2.3 MiB/s sustained throughput on big files, Sortie 6 (8 GB) ≈ 60 min, Sortie 11 (32 GB) ≈ 4 hours. Foreground-agent dispatch (10-min Bash cap) is structurally infeasible. Three options on the table — same three as before, but now we have empirical data:
  - **(A) Operator-manual ships**: operator runs `acervo upload` (or `acervo ship`) in host shell for each big sortie; supervisor dispatches haiku closeouts. Token-cheap, robust, requires operator labor.
  - **(B) Supervisor with `nohup` + Monitor**: kicker sortie launches `nohup`'d upload, then a poller sortie loops Monitor to wait for the bash task. Untested whether Bash timeout truly kills nohup'd children; needs experimentation on a low-stakes ship first.
  - **(C) Hybrid**: operator-manual for the four 8 GB+ ships (6, 8, 9, 10 — wait, 6 = 8 GB; 8 = 13 GB; 9 = 19 GB; 10 = 25 GB; 11 = 32 GB; 12 = 18 GB), supervisor-foreground for the small ones. There are no remaining small ones in WU1 — all subsequent ships are ≥ 8 GB. Hybrid degenerates to option A.

Preserved on disk (DO NOT delete during pause):
- `/tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/` — full Sortie 5 download (8.3 GiB; 19 files). Manifest is correct (subdir-prefixed paths under acervo 0.8.4). aws s3 sync will skip files already pushed to R2.
- `/tmp/acervo-staging/lmstudio-community_Qwen3-4B-MLX-4bit/`, `/tmp/acervo-staging/lmstudio-community_Qwen3-4B-MLX-8bit/`, `/tmp/acervo-staging/lmstudio-community_Qwen3-8B-MLX-4bit/` — Sorties 2, 3, 4 staging dirs.

Preserved on disk (DO NOT delete during pause):
- `/tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/` — full Sortie 5 download (19 files, including 4.5 GiB `text_encoder/model.safetensors`). Lets the post-fix retry skip re-download.
- `/tmp/acervo-staging/lmstudio-community_Qwen3-4B-MLX-4bit/`, `/tmp/acervo-staging/lmstudio-community_Qwen3-4B-MLX-8bit/`, `/tmp/acervo-staging/lmstudio-community_Qwen3-8B-MLX-4bit/` — Sorties 2, 3, 4 staging dirs (already shipped, but harmless to keep until disk pressure dictates).

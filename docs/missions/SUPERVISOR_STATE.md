# SUPERVISOR_STATE.md — OPERATION FAREWELL EMBRACE

> **Terminology**: A *mission* is the definable scope of work. A *sortie* is an atomic agent task within that mission. A *work unit* groups related sorties.

## ▶ Mission Status: RUNNING — operator chose option D for Sortie 21

Operator decision (2026-04-30): land all uncommitted work + drop `autoTokenizerParity` (with documenting comment) + add `Qwen3Generator.swift` audit note. Re-dispatching Sortie 21 with reduced scope.

## (Resolved) Prior block — Sortie 21 parity-test design flaw

**TL;DR**: Sortie 21's parity test correctly stopped on AutoTokenizer divergence, but the divergence is on the 7-vocab GPT-2 stub from Sortie 15 — every input becomes all-`<unk>` tokens, so this only tests `<unk>`-serialization edge cases between two BPE library implementations, not real tokenization. **Sortie 15's fixture design was insufficient** for what Sortie 21 needs to prove. Operator decision required on path forward.

**TekkenParity PASSED** (but doesn't prove anything about the library swap — TekkenTokenizer is project code; identical pre/post migration).

**AutoTokenizerParity FAILED** on degenerate input. Two library differences observed:
- **Encode**: old library counted Unicode by code-unit; new by code-point. Multi-byte CJK/Arabic/Hindi got fewer (still all-`<unk>`) tokens.
- **Decode**: old `decode([3,3])` → `"<unk><unk>"`; new → `"<unk> <unk>"` (space inserted).

Whether these differences MATTER in production depends on whether Qwen3/Mistral tokenizers (full vocabularies, not stubs) hit the same code paths. The test doesn't tell us — and CAN'T tell us with stub fixtures. Decision options below in Decisions Log.

Working tree (uncommitted):
- `Tests/Flux2CoreTests/Flux2CoreTests.swift` — compile-fix edits for Sortie 20's symbol renames (NEEDED — without these the test build doesn't compile)
- `Tests/FluxTextEncodersTests/ModelRegistryTests.swift` — same
- `Tests/FluxTextEncodersTests/TokenizerParityTests.swift` — new file, contains tekkenParity (passing) + autoTokenizerParity (failing)
- `docs/missions/SUPERVISOR_STATE.md` — supervisor edits

## ▶ Prior Mission Status: SPLIT-EXECUTION — operator-manual WU1 ships; supervisor proceeds with WU2 in parallel

**Strategy update from operator** (2026-04-30): given the structural mismatch between the 10-min Bash timeout and 8 GB+ ship payloads, WU1 ships 5–12 will run via `scripts/wu1-bulk-ship.sh` in the operator's separate terminal window. Sortie 13 (CDN read-side smoke test) is **SKIPPED** per operator direction; the new operating assumption is **eventual consistency** of CDN ships. WU2 Sortie 16's gate ("WU1 complete (all 11 manifests verified on CDN)") is **OVERRIDDEN** so WU2 can begin in parallel — the WU2 sorties through 20 are pure code changes that don't read from the CDN at compile time.

**WU1 closeout protocol** (when bulk-ship script finishes): operator runs `/mission-supervisor resume`. Supervisor will probe each of the 8 CDN manifest URLs; for each that's HTTP 200, dispatch a haiku closeout sortie that appends a Sortie N section to `cdn-ship-log.md` and creates a path-restricted commit. Closeouts can be dispatched in parallel by sortie since they touch the same file (cdn-ship-log.md) — actually NO, let me revise: closeouts must be SERIAL because they touch the same file and parallel commits would race. Each closeout will commit independently before the next dispatches.

**WU2 unblock**: dispatching Sortie 16 (Package.swift swap) NOW, alongside the operator's bulk-ship script.

**WU1 Sortie 5 status**: still PARTIAL on disk; the bulk-ship script's Sortie-5 special-case will detect existing staging and run `acervo upload` only (skipping re-download).

Resume Protocol section retained below for audit / future-mission reference.

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
- Work unit state: RUNNING
- Current sortie: 22 of 22 (Sorties 14, 15, 16, 17, 18, 19, 20, 21 COMPLETED at commits `45cd8b2`, `330ecac`, `f193c0d`, `4e16903`, `d5bfae9`, `89ed3ce`, `97297b9`, `088b652`)
- Sortie state: PENDING (next dispatch this iteration)
- Sortie type: code + ops (push branch, open PR, wait for CI, delete parity oracle, push cleanup, update PR description)
- Model: sonnet (mechanical work; CI-wait is bounded polling)
- Complexity score: ~10
- Last verified (Sortie 21 — option D recovery):
  - Commit `088b652` — 4 files: Flux2CoreTests.swift + ModelRegistryTests.swift (compile fixes for Sortie 20's symbol renames), TokenizerParityTests.swift (new — tekkenParity only; autoTokenizerParity removed with retrospective header doc), Qwen3Generator.swift (audit comment at line 105 above first decode call site).
  - All FluxTextEncodersTests + Flux2CoreTests pass. tekkenParity test passes (12/12 fixtures match).
  - autoTokenizerParity dropped — Sortie 15's 7-vocab GPT-2 stub fixtures only exercised `<unk>`-serialization edge cases, not real tokenization. Real-vocab parity is post-mission TODO.
- WU1 ship status (probed at Sortie 22 dispatch time):
  - Sortie 5 (aydin99): HTTP 200 ✅
  - Sorties 6, 7, 8, 9, 10, 11, 12: all HTTP 404 — bulk-ship script still in flight.
  - Sortie 22 does NOT depend on these. CI runs FluxTextEncodersTests (parity uses committed fixtures) + Flux2CoreTests (metadata + ModelRegistry tests, no CDN access) + Flux2GPUTests (environmentally skipped without KLEIN_MODEL_PATH). Once WU1 finishes, runtime user-facing behavior will be available; CI green is independent.
- Notes for Sortie 22:
  - Mission branch is `ahead 14` of `origin/mission/farewell-embrace/01` — agent must push first.
  - Agent uses `gh pr` (allowed per CLAUDE.md). DO NOT auto-merge — operator merges.
  - CI-wait strategy: bounded polling with `gh pr checks` (NOT `--watch`). Agent polls every ~60s for up to 8–9 minutes within the Bash timeout. If CI still running after that, STOP and report — supervisor dispatches a closeout for the parity-oracle deletion.
  - Final cleanup: delete `Tests/FluxTextEncodersTests/TokenizerParityTests.swift` + `Tests/Fixtures/TokenizerParity/`. Re-run local tests. Push.
  - Update PR description with mission summary + recon doc links.

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
| WU2 | 16 | `f193c0d` | COMPLETED | Package.swift swap: dropped swift-transformers, added swift-tokenizers (0.4.2) + SwiftAcervo (0.8.4). All 4 grep exit criteria pass on regenerated Package.resolved. Build fails as expected — 6 errors all in Qwen3Generator.swift, exactly matching Sortie 17's task list. **Plan deviation**: Package.resolved is `.gitignored`, not tracked; agent correctly regenerated on disk and did NOT force-track. |
| WU2 | 17 | `4e16903` | COMPLETED | Tokenizers API renames (decode(tokens:)→decode(tokenIds:), AutoTokenizer.from(modelFolder:)→from(directory:)) at 12 source/test sites + applyChatTemplate addGenerationPrompt:false lock. 5 files in commit (TekkenTokenizer.swift, Qwen3Generator.swift, FluxTextEncoders.swift, TokenizerTests.swift, FluxTextEncodersTests.swift). FluxTextEncoders + Flux2Core library targets build cleanly. Flux2App link still fails on `import Hub` in TextEncoderModelDownloader.swift (Sortie 18 scope). Plan exit #2 grep over-broad (catches TekkenTokenizer's concrete decode method) — accepted as complete on functional-build basis. |
| WU2 | 18 | `d5bfae9` | COMPLETED | TextEncoderModelDownloader.swift fully rewritten against Acervo (497 → 182 ins/315 del). All 3 grep exit criteria pass; `make build` SUCCEEDED end-to-end. Hub paths + `ensureTekkenJson` + cache helpers all gone. Public API surfaces preserved; `notProvisionedOnCDN(variant:)` added to `TextEncoderModelDownloaderError`. **Surprise finding**: `Flux2Core/Loading/ModelDownloader.swift` also compiles cleanly — it's a hand-rolled URLSession client, no `import Hub`. Plan's note that it would fail here was conservative; Sortie 19 still owns rewriting that file. |
| WU2 | 19 | `89ed3ce` | COMPLETED | Hand-rolled HF API client deleted. ModelDownloader.swift 734 → 241 ins/515 del. `make build` PASS for ALL targets (first time post-migration). 4 cut-variant `notProvisionedOnCDN` errors + `isProvisionedOnCDN` UI property. 3 callers updated; test file required no changes (Sortie 14 already removed plan-listed line). 12 `huggingface.co` residual matches are in files outside Sortie 19's path restriction; Sortie 20 cleans them up. **Plan-addition flagged**: `TextEncoderModelRegistry.swift:372` runtime tekken.json URL is dead (post-Sortie-18) and not in Sortie 20's plan list — Sortie 20 dispatch will include it. |
| WU2 | 20 | `97297b9` | COMPLETED | Strip HF runtime strings + README ack. 4 files (README, ModelRegistry.swift, TextEncoderModelRegistry.swift, ModelDownloader.swift caller fix). All grep exit criteria pass; 4 residual `huggingface` matches all in attribution comments. Dead `tekkenJsonURL` deleted. README has full per-model attribution table. `make build` PASS. **Cosmetic nit**: README says `swift-transformers` for the dep name; actual is `swift-tokenizers` (DePasqualeOrg). |
| WU2 | 21 | `088b652` | COMPLETED (option D — reduced scope) | tekkenParity test passes (12/12 fixtures); autoTokenizerParity test REMOVED (Sortie 15 stub-fixture design only exercised `<unk>`-serialization edge cases). Header comment in TokenizerParityTests.swift documents the design retrospective. Audit comment added in Qwen3Generator.swift line 105 flagging swift-tokenizers BPE decode space-insertion. Test compile-fixes for Sortie 20 symbol renames included (Flux2CoreTests.swift, ModelRegistryTests.swift). All Flux2CoreTests + FluxTextEncodersTests pass; Flux2GPUTests environmentally skipped. |
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
| 2026-04-30 | — | — | OPERATOR DECISION: split-execution strategy | Operator chose to run WU1 ships 5–12 via a bulk-ship script (`scripts/wu1-bulk-ship.sh`) in a separate terminal window, while the supervisor proceeds with WU2 in parallel. Sortie 13 (CDN read-side smoke test) SKIPPED — operating assumption is eventual consistency of CDN ships. WU2 Sortie 16's gate of "WU1 complete" OVERRIDDEN — WU2 Sorties 16–20 are compile-time-only changes that don't read from the CDN; CDN-dependent runtime tests (Sortie 21) come last by which time ships should be done. Trade-off: if WU1 ships fail, Sortie 21 will fail too and require re-runs once CDN populated. Acceptable given operator velocity preference. |
| 2026-04-30 | — | — | scripts/wu1-bulk-ship.sh created | Wraps the 8 remaining WU1 ships (Sorties 5, 6, 7, 8, 9, 10, 11, 12). Sortie 5 special-case: detects existing staging and runs `acervo upload` instead of full ship (avoids re-download of 8.3 GiB). For non-lmstudio repos (5, 7, 11, 12), tries default invocation first; falls back to `--no-verify` on CHECK 1 failure. Each ship's stdout/stderr captured to `docs/missions/ship-logs/<slug>.log`. CDN-manifest verification at end. Master log goes to `docs/missions/cdn-bulk-ship.log` via tee. Operator runs in a separate terminal: `bash scripts/wu1-bulk-ship.sh 2>&1 \| tee -a docs/missions/cdn-bulk-ship.log`. |
| 2026-04-30 | WU2 | 16 | Model: opus | Foundation override: Sortie 16 establishes the new dependency graph that all of Sorties 17–22 depend on. Complexity score ~18 (foundation 10 + risk 5 from new package APIs + task complexity 3). The Package.swift swap is mechanically simple but architecturally critical; opus's larger context handles the failure mode (build expected to fail at exit, with errors confined to specific files) better than sonnet. |
| 2026-04-30 | WU2 | 16 | Sortie 16 COMPLETE at commit `f193c0d` | All exit criteria pass. **Plan correction surfaced**: `Package.resolved` is `.gitignored` (line 32 of `.gitignore`), not tracked. Plan task 4 instructed `git rm Package.resolved` — wrong. Agent regenerated it on disk via `make build`, ran greps against on-disk file (all pass), did NOT force-track. Good judgment call. Build fails in `Qwen3Generator.swift` only — strict subset of expected call-site files; matches Sortie 17's exact line list. |
| 2026-04-30 | WU2 | 17 | Model: sonnet (overrides foundation rule) | Score 22 by naive scoring (foundation 10 + complexity 9 + risk 3) which would force opus, but the work is mechanical grep-and-replace at known line numbers. Plan explicitly says "Risk: low (mechanical renames, single grep'd call sites)". Sonnet handles this cleanly at 1/3 the cost of opus. Sergeant's "right tool for the job" principle. |
| 2026-04-30 | WU2 | 17 | Sortie 17 COMPLETE at commit `4e16903`; plan exit-criterion #2 grep over-broad | The grep `tokenizer\.decode\(` catches BOTH swift-tokenizers `Tokenizer.decode(tokenIds:)` calls (which were updated) AND `TekkenTokenizer.decode(_ tokens:)` calls (a concrete class method — different method, takes positional Int array). Residual matches in MistralGenerator.swift (6), FluxTextEncoders.swift (2), TokenizerTests.swift (2) compile fine. Accepted as complete on functional-build basis. **Plan addendum needed (post-mission)**: rename `TekkenTokenizer.decode(_:)` to `decode(tokenIds:)` for cross-cutting consistency, OR tighten the grep to only catch protocol-level `Tokenizer` calls. |
| 2026-04-30 | WU2 | 17 | Plan line numbers drifted in current codebase | Plan's line numbers were 4–40 lines off across multiple files (e.g. Qwen3Generator.swift +1, FluxTextEncoders.swift +6 to +39). Agent adapted by grepping for the call signature near each plan-cited line and applying renames at the actual sites. **Plan addendum needed (post-mission)**: refresh line numbers, OR stop citing line numbers and rely on grep-able patterns instead. |
| 2026-04-30 | WU2 | 18 | Model: opus | Foundation + high-risk + high-complexity. Score ~24 (task complexity 9 from large-file rewrite + foundation 5 + risk 5 + ambiguity 3 + new API surface 2). Plan classifies risk:high, complexity:high. opus's larger context handles the multi-section rewrite and error-case design better than sonnet. Sortie 19 will mirror this pattern (also opus). |
| 2026-04-30 | WU2 | 18 | Entry criterion override: WU1 manifest verification | Plan requires "WU1 manifests for Sorties 2, 3, 4, 6, 8, 9, 10 verified live on R2" before Sortie 18 dispatch. Operator's eventual-consistency override applies — bulk-ship script is producing manifests out-of-band. Sortie 18's CDN dependency is RUNTIME (Sortie 21 parity tests); compile-time only requires SwiftAcervo's API which is resolved already (Package.resolved). Agent will hardcode expected file lists per plan task 1. |
| 2026-04-30 | WU2 | 18 | Sortie 18 COMPLETE at commit `d5bfae9` | Single-file rewrite. Significant simplification (497 lines → 182 ins / 315 del). All Hub references gone. `make build` PASSED end-to-end. Plan-update finding: `Flux2Core/Loading/ModelDownloader.swift` doesn't `import Hub` — it's a hand-rolled URLSession client. Plan was conservative in saying it would fail here; Sortie 19's scope (rewriting that file) is unchanged but the entry condition is more relaxed than expected. |
| 2026-04-30 | WU2 | 19 | Model: opus | Largest single rewrite in mission per plan classification. Cross-cuts 5+ files (main downloader + 4 caller sites + possibly TransformerVariant extension). Score ~24 (task complexity 10 + risk 5 + multi-file 4 + ambiguity 3 + foundation 2). High risk because the new error type and 4 caller updates must land together to keep build green. Plan suggested 19a/19b split if context pressure; opus's larger context should keep this atomic. |
| 2026-04-30 | WU2 | 19 | Entry criterion override: same as Sortie 18 | Plan requires Sorties 5, 7, 11, 12 manifests live before Sortie 19. Operator's eventual-consistency override applies. Agent hardcodes expected file lists; runtime CDN exercise comes in Sortie 21. |
| 2026-04-30 | WU2 | 19 | Sortie 19 COMPLETE at commit `89ed3ce` | Largest single rewrite landed atomically (no 19a/19b split needed). Build green for ALL targets including tests. 12 residual `huggingface.co` matches are all outside Sortie 19's path restriction. Agent diverged from Sortie 18's hardcoded-file-list pattern by passing `files: []` (= use whole manifest); rationale: diffusers-style nested repos differ from flat LM repos. Both choices are appropriate to repo shape. |
| 2026-04-30 | WU2 | 19 | Plan-addendum: tekken.json dead URL at TextEncoderModelRegistry.swift:372 | Sortie 19 agent surfaced a runtime `huggingface.co/.../tekken.json` URL not captured in Sortie 20's task list. Dead since Sortie 18 (tekken.json now ships with every lmstudio MLX quant). Sortie 20 dispatch will include this in its delete scope. |
| 2026-04-30 | WU2 | 20 | Model: sonnet | Mechanical work — accessor deletes + symbol renames (`huggingFaceRepo`→`repoId`, `huggingFaceSubfolder`→`repoSubfolder`) with all-caller-fix-needed; README authoring with license lookups from `.build/checkouts/`. Score ~10. Plan classifies risk:low. Sonnet appropriate. |
| 2026-04-30 | WU2 | 20 | Sortie 20 COMPLETE at commit `97297b9` | All exit criteria pass. Only 1 caller file outside the registries needed updating (ModelDownloader.swift, 2 lines). README ack section added with full per-model attribution table. Cosmetic: dep table says `swift-transformers` (local checkout dir) — actual dep is `swift-tokenizers` (DePasqualeOrg). Worth a 1-line README fix in a follow-up cycle. |
| 2026-04-30 | WU2 | 21 | Model: sonnet | Tokenizer-parity test is mechanical against committed fixtures; test audit is open-ended. Score ~16. Plan classifies risk:medium because hidden parity gaps are silent regressions. **Critical guardrail in dispatch**: agent must STOP on parity failure; must NOT modify fixtures or weaken assertions. |
| 2026-04-30 | WU2 | 21 | Entry criterion: WU1 ship status check | Plan-listed entry criterion is "Sortie 20 commit + Sortie 15 fixtures present"; both met. WU1 ship status independently probed: 1/8 manifests live (Sortie 5 only). Other 7 still HTTP 404 (bulk-ship script in flight). Sortie 21's parity test does NOT need WU1 manifests — uses Sortie 15's committed fixtures. Flux2GPUTests environmentally skipped without `KLEIN_MODEL_PATH`. Dispatching now. |
| 2026-04-30 | WU2 | 21 | Sortie 21 BLOCKED on parity-test design flaw | Agent correctly stopped on AutoTokenizer parity failure (15 issues across 12 fixtures); did NOT modify fixtures or weaken assertions. **However**, the failures are on Sortie 15's 7-vocab GPT-2 stub — every input becomes `<unk>` tokens. The "parity test" is only checking how two BPE libraries serialize `<unk>` arrays. Real divergences observed: (a) old lib counted Unicode by code-unit, new by code-point (different all-`<unk>` token counts); (b) new lib's BPE decoder inserts a space between successive `<unk>` tokens. Whether these matter in production depends on whether Qwen3/Mistral full vocabularies hit the same code paths — the stub-based test cannot tell us. Tekken parity passed but is uninformative (TekkenTokenizer is project code, never depended on either library). |
| 2026-04-30 | WU2 | 21 | Sortie 15 fixture-design retrospective | When Sortie 15 ran, supervisor flagged that "fixtures use the same default-init tokenizers as the existing test suite [and] parity oracle scope is plumbing equivalence on default-init code paths, not full vocabulary coverage." That was the right read but supervisor should have pushed back HARDER and either (a) redesigned Sortie 15 to use real Qwen3/Mistral tokenizer files, or (b) deferred AutoTokenizer parity to a runtime test post-CDN-ship. The current divergence is the predictable consequence of stub-based fixtures. Lesson for future missions: parity oracles must use representative input, not minimal stubs. |
| 2026-04-30 | WU2 | 21 | Operator-decision options surfaced | (A) **Drop AutoTokenizer parity test, keep Tekken**: lands Sortie 21 with reduced scope; document in commit message that autotok parity test was misdesigned and would be replaced post-mission with a real-vocab check. Risk: silent Qwen3/Mistral runtime regression undetected until production use. (B) **Defer Sortie 21 until a real Qwen3 manifest is live, regenerate fixtures using new library, assert round-trip self-consistency**: slow; doesn't actually verify "no regression" since both fixtures and assertions come from the new library. (C) **Operator manual smoke test**: subjective, doesn't run in CI. (D) **Land all uncommitted work** (the test compile-fixes ARE needed for build green) and skip the autoTokenizerParity test with explicit "expected divergence on stub vocab" reason; audit production decode call sites in Qwen3Generator.swift. |
| 2026-04-30 | WU2 | 21 | OPERATOR DECISION: option D | "option d. please resume." Sortie 21 dispatched in option-D shape: drop autoTokenizerParity (with retrospective header), keep tekkenParity, add Qwen3Generator audit comment, land test compile-fixes. Single sonnet, single commit (`088b652`). All tests pass. |
| 2026-04-30 | WU2 | 22 | Model: sonnet | Mechanical work: push branch, open/update PR via `gh pr`, verify no `HF_TOKEN` secret (`gh secret list`), bounded-poll CI status (`gh pr checks` NOT `--watch` to avoid Bash-timeout backgrounding), delete parity oracle + fixture files, push cleanup, update PR description. Score ~10. |

## Overall Status

OPERATION FAREWELL EMBRACE — split-execution. 12 of 22 sorties complete (WU1: 1, 2, 3, 4; WU2: 14, 15, 16, 17, 18, 19, 20, 21).

This iteration:
- **Operator running `scripts/wu1-bulk-ship.sh`** — WU1 Sorties 5–12 out-of-band. Status: 1/8 manifests live (Sortie 5 only); 7 still in flight.
- **Supervisor dispatching WU2 Sortie 22** — TERMINAL sortie. Push branch, open/update PR, wait for CI, delete parity oracle, push cleanup.

After Sortie 22 completes + WU1 ships finish: mission is ready for `/mission-supervisor brief` (post-mission review + archive).

Resolved:
- `acervo` 0.8.4 fix: installed and verified.
- Path forward for nested-repo manifests: option A (fix `acervo` upstream), landed.
- Big-ship strategy: operator-manual via bulk-ship script.

Closeout protocol (when bulk-ship script finishes):
- Operator runs `/mission-supervisor resume`. Supervisor probes 8 CDN manifest URLs; for each HTTP 200, dispatches a SERIAL haiku closeout sortie (one at a time — they share `cdn-ship-log.md`) that appends a Sortie N section and creates a path-restricted commit.

Preserved on disk:
- `/tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/` — Sortie 5 staging (8.3 GiB). Bulk-ship script will detect this and run `acervo upload` only.
- `/tmp/acervo-staging/lmstudio-community_Qwen3-4B-MLX-4bit/`, `lmstudio-community_Qwen3-4B-MLX-8bit/`, `lmstudio-community_Qwen3-8B-MLX-4bit/` — Sorties 2/3/4 staging (already shipped).

Preserved on disk (DO NOT delete during pause):
- `/tmp/acervo-staging/aydin99_FLUX.2-klein-4B-int8/` — full Sortie 5 download (19 files, including 4.5 GiB `text_encoder/model.safetensors`). Lets the post-fix retry skip re-download.
- `/tmp/acervo-staging/lmstudio-community_Qwen3-4B-MLX-4bit/`, `/tmp/acervo-staging/lmstudio-community_Qwen3-4B-MLX-8bit/`, `/tmp/acervo-staging/lmstudio-community_Qwen3-8B-MLX-4bit/` — Sorties 2, 3, 4 staging dirs (already shipped, but harmless to keep until disk pressure dictates).

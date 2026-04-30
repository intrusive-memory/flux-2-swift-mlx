# EXECUTION_PLAN.md — Flux2Swift HuggingFace Excision

> **Scope expansion notice**: This plan has grown beyond the original `docs/missions/tokenizer-migration.md`. By user direction during plan iteration, the mission now covers (a) the original `swift-transformers` → `swift-tokenizers` swap, (b) full elimination of HuggingFace runtime dependencies — including the hand-rolled HF API client in `Flux2Core/Loading/ModelDownloader.swift`, the hardcoded `huggingface.co` URL in `TextEncoderModelDownloader.swift`, and all `huggingFaceURL` display strings — and (c) replacement of model loading with the SwiftAcervo CDN. `swift-hf-api` is NOT introduced.

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase).

## Mission Brief

Convert Flux2Swift from a HuggingFace-coupled package to a HuggingFace-free package backed by the user-operated Cloudflare R2 CDN via SwiftAcervo. Three concurrent transformations:

1. **Tokenization**: `huggingface/swift-transformers` (products `Hub` + `Transformers`) → `DePasqualeOrg/swift-tokenizers` 0.4.2 (product `Tokenizers`).
2. **Model loading**: HuggingFace Hub (via `HubApi.snapshot(from:matching:)` and a hand-rolled `huggingface.co/api/...` client) → `SwiftAcervo` 0.8.3+ (`Acervo.ensureAvailable(_:files:progress:)` against R2 CDN bucket `intrusive-memory-models`).
3. **HF surface elimination**: every `huggingface.co` URL, every `huggingFaceRepo`/`huggingFaceURL` accessor, the direct `URLSession` fetch of `tekken.json` from huggingface.co — all removed from runtime code paths. HuggingFace remains acknowledged ONLY in (a) the app's About / opensource-dependency credits surface, and (b) code comments that reference origin/license attribution.

Source requirements: `docs/missions/tokenizer-migration.md` (original scope) + this plan's expansion.

## User-Locked Decisions (from plan iteration)

These are decided. The refine passes do not re-litigate them.

| Decision | Choice |
|---|---|
| swift-tokenizers pin | `from: "0.4.2"` (latest as of 2026-04-24) |
| swift-hf-api | **Not introduced.** Acervo replaces all HF runtime calls. |
| SwiftAcervo dep style | `.package(url: "https://github.com/intrusive-memory/SwiftAcervo", from: "0.8.3")` (up to next major, 1.0). |
| Unused Flux2Core `Hub`/`Transformers` deps in Package.swift | **Deleted, not renamed.** |
| CDN scope | **Reduced from 16 to 11 mirrored repos** (~300 GB → ~137 GB). Cut: `FLUX.2-dev` (BF16), `FLUX.2-klein-base-4B`, `FLUX.2-klein-base-9B`, `FLUX.2-klein-9b-kv`. **`mistralai/Mistral-Small-3.2-24B-Instruct-2506` dropped entirely** — the lmstudio-community MLX quants ship `tekken.json` themselves (verified across 4-bit, 6-bit, 8-bit), making the upstream Mistral repo redundant. The `ensureTekkenJson(at:)` fallback in `TextEncoderModelDownloader.swift:252-283` was defensive code that never fired in practice. |
| Cut variants in `ModelRegistry` | **Kept in the enum** to preserve API stability, but `download(...)` for `TransformerVariant.bf16`, `TransformerVariant.klein4B_base_bf16`, `TransformerVariant.klein9B_base_bf16`, `TransformerVariant.klein9B_kv_bf16`, **and** text-encoder `ModelVariant.bf16` throws a `notProvisionedOnCDN` error with a clear message naming the variant and explaining it will be re-enabled in a follow-up CDN mission. UI iteration over `allCases` can use a new `isProvisionedOnCDN` property to gray them out. |
| Gated repos requiring license acceptance | **1 remaining** (down from 5): `black-forest-labs/FLUX.2-klein-9B`. URL: `https://huggingface.co/black-forest-labs/FLUX.2-klein-9B`. Operator clicks "Agree and access" before WU1 begins. |
| License acceptance verification | **`hf` CLI probe**, not manual log. `hf auth whoami` + `hf download black-forest-labs/FLUX.2-klein-9B README.md --local-dir /tmp/probe` confirms acceptance. License-not-accepted returns 403; accepted returns 200. (HuggingFace does not expose programmatic license acceptance — operator click on the website is unavoidable for first acceptance, but verification is automated.) |
| `huggingFaceURL` display strings | **Removed from runtime code.** HuggingFace credit lives in README only. Code comments may attribute origin/license but emit no URLs. |
| Backward-compat read of `~/.cache/huggingface/...` | **None.** Hard cut to Acervo paths only. Pre-existing HF caches will be ignored; users re-download from the Acervo CDN. |
| Backend trait for swift-tokenizers | Default `Swift` (no Rust XCFramework). |
| About / credits surface | **README only for this mission.** The app's GUI About panel work is deferred to a separate workstream. Sortie 11 adds a `## Acknowledgments` section to the project README crediting HuggingFace as the origin of mirrored model weights and noting Intrusive Memory's R2 redistribution. |
| `acervo ship` subfolder support | **Confirmed: `acervo` treats the path as a normalized string and accepts paths multiple directories deep.** WU1 S2 ships `VincentGOURBIN/flux_qint_8bit` as `acervo ship VincentGOURBIN/flux_qint_8bit flux-2-dev/transformer/qint8/`. No runtime filter needed. |
| Package shape | **Library-only.** Both `Sources/Flux2CLI/` and `Sources/FluxEncodersCLI/` are deleted in Sortie 5. The package's executable products go away; only `Flux2Core` and `FluxTextEncoders` libraries remain. End-to-end CLI exercise is delegated to the sibling `../SwiftVinetas` repo. |
| `customModelsDirectory` | **Removed entirely in Sortie 5.** Sandboxed apps cannot use a custom storage directory (container-bound), so the override was misleading API. Storage location is `Acervo.sharedModelsDirectory` post-migration; no override hook is exposed. The two test files that exclusively covered this surface (`Tests/FluxTextEncodersTests/TextEncoderModelDirectoryTests.swift`, `Tests/Flux2CoreTests/ModelDirectoryTests.swift`) are deleted. |

## Codebase Reconnaissance (verified ahead of plan)

### Tokenization surface

| Concern | Confirmed locations |
|---|---|
| `import Tokenizers` | `FluxTextEncoders.swift:14`, `Tokenizer/TekkenTokenizer.swift:14`, `Embeddings/KleinEmbeddingExtractor.swift:16`, `Generation/Qwen3Generator.swift:12` |
| `decode(tokens:)` keyword form (rename to `decode(tokenIds:)`) | `TekkenTokenizer.swift:398, 475`; `Qwen3Generator.swift:104, 134, 142, 230, 259, 267` (8 sites) |
| `decode(_:skipSpecialTokens:)` positional form (must rewrite — no positional overload in swift-tokenizers 0.4.2) | `FluxTextEncoders.swift:896`; `Tests/FluxTextEncodersTests/TokenizerTests.swift:140` |
| `from(modelFolder:)` (rename to `from(directory:)`; new signature drops `hubApi:` and `strict:` parameters) | `FluxTextEncoders.swift:201`, `Tokenizer/TekkenTokenizer.swift:232` |
| `applyChatTemplate(messages:)` call sites (audit `addGenerationPrompt` default flip) | Explicit `false`: `EmbeddingExtractor.swift:192`. Default-relying (must be made explicit): `TekkenTokenizer.swift:473, 510`; `TokenizerTests.swift:77, 93, 108`; `FluxTextEncodersTests.swift:204` |

### HuggingFace runtime surface (full elimination targets)

| Concern | Confirmed locations |
|---|---|
| `import Hub` | `FluxTextEncoders/Loading/TextEncoderModelDownloader.swift:7` (only `import`-level site) |
| `HubApi.snapshot(from:matching:)` calls | `TextEncoderModelDownloader.swift:224` (Mistral) and `:320` (Qwen3) |
| `HubApi(downloadBase:)` initializer | `TextEncoderModelDownloader.swift:29` |
| Hardcoded `huggingface.co` URL (URLSession direct fetch) | `TextEncoderModelDownloader.swift:267` — `https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/resolve/main/tekken.json`. **The entire `ensureTekkenJson(at:progress:)` method (lines 252-283) is deleted, not migrated** — `tekken.json` ships in every lmstudio-community MLX quant directly (verified by HF API probe), so the fallback is never needed. |
| Hand-rolled HF API client (NOT using `Hub`) | `Flux2Core/Loading/ModelDownloader.swift:419-480` — hits `huggingface.co/api/models/<repo>/tree/main` and `huggingface.co/<repo>/resolve/main/<file>` directly via URLSession. Largest single rewrite in the mission. |
| `huggingFaceRepo` / `huggingFaceURL` / `huggingFaceSubfolder` accessors (display strings) | `Flux2Core/Configuration/ModelRegistry.swift:34, 59, 115, 252, 267, 316, 323, 328` |
| Legacy HF cache path resolution | `TextEncoderModelDownloader.swift:87-110` (`~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/...`) — **delete entirely** per locked decision 7. |
| `repoId` field on `ModelInfo` / `Qwen3ModelInfo` / `TransformerVariant` | `TextEncoderModelRegistry.swift`, `Flux2Core/ModelRegistry.swift`. **Kept** — Acervo's `slugify(_:)` converts `org/repo` → `org_repo` transparently at the resolution layer. |

### Models requiring CDN provisioning (11 total, ~137 GB)

| # | Model ID | Approx size | Gated | Backed `*Variant` cases | Notes |
|---|---|---|---|---|---|
| 1 | `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit` | 25 GB | No | text-encoder `ModelVariant.mlx8bit` | Self-contained tokenizer (`tekken.json`, `tokenizer.json`, etc.) verified |
| 2 | `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-6bit` | 19 GB | No | text-encoder `ModelVariant.mlx6bit` | Self-contained tokenizer verified |
| 3 | `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit` | 13 GB | No | text-encoder `ModelVariant.mlx4bit` | Self-contained tokenizer verified |
| 4 | `lmstudio-community/Qwen3-4B-MLX-8bit` | 4 GB | No | `Qwen3ModelInfo` 4B-8bit | Self-contained tokenizer verified |
| 5 | `lmstudio-community/Qwen3-4B-MLX-4bit` | 2 GB | No | `Qwen3ModelInfo` 4B-4bit | Self-contained tokenizer verified |
| 6 | `lmstudio-community/Qwen3-8B-MLX-8bit` | 8 GB | No | `Qwen3ModelInfo` 8B-8bit | Self-contained tokenizer verified |
| 7 | `lmstudio-community/Qwen3-8B-MLX-4bit` | 4 GB | No | `Qwen3ModelInfo` 8B-4bit | Self-contained tokenizer verified |
| 8 | `VincentGOURBIN/flux_qint_8bit` | 32 GB | No | `TransformerVariant.qint8` | qint8 transformer; HF subfolder `flux-2-dev/transformer/qint8` |
| 9 | `black-forest-labs/FLUX.2-klein-4B` | 8 GB | No | `TransformerVariant.klein4B_bf16` | Apache-2.0 (klein); also VAE source for ALL Flux.2 models |
| 10 | `aydin99/FLUX.2-klein-4B-int8` | 4 GB | No | `TransformerVariant.klein4B_8bit` | klein 4B 8-bit community quant |
| 11 | `black-forest-labs/FLUX.2-klein-9B` | 18 GB | **Yes** | `TransformerVariant.klein9B_bf16` | Only gated repo. License acceptance URL: `https://huggingface.co/black-forest-labs/FLUX.2-klein-9B` |

### Variants kept in `ModelRegistry` but NOT provisioned (download throws `notProvisionedOnCDN`)

| Variant | Original HF repo | Reason for cut | Re-enable path |
|---|---|---|---|
| `ModelVariant.bf16` (text encoder) | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 50 GB BF16 not used in production; quantized variants do all encoding | Follow-up CDN mission ships if needed |
| `TransformerVariant.bf16` | `black-forest-labs/FLUX.2-dev` | 64 GB BF16 dev; qint8 is canonical production variant; license redistribution complications | Follow-up CDN mission |
| `TransformerVariant.klein4B_base_bf16` | `black-forest-labs/FLUX.2-klein-base-4B` | LoRA training only, not v1 | Ship when LoRA training is v1 |
| `TransformerVariant.klein9B_base_bf16` | `black-forest-labs/FLUX.2-klein-base-9B` | LoRA training only, not v1; gated | Ship when LoRA training is v1 |
| `TransformerVariant.klein9B_kv_bf16` | `black-forest-labs/FLUX.2-klein-9b-kv` | Multi-reference I2I specialty variant; gated | Ship when multi-ref I2I is v1 |

## Pre-flight Risk Notes

- **`acervo ship` requires `HF_TOKEN` plus operator license acceptance for the one gated repo (`black-forest-labs/FLUX.2-klein-9B`).** Acceptance URL: `https://huggingface.co/black-forest-labs/FLUX.2-klein-9B`. Failed CHECK 1 is recoverable — accept the license, re-run.
- **HF download cost.** Sortie 2 + Sortie 3 of WU1 download ~137 GB from huggingface.co before re-uploading to R2. This is the operator's bandwidth/storage. Plan for ≥3 hours of wall clock on a typical broadband uplink, plus R2 upload time. (Down from the original ~300 GB / ≥6 hour estimate after the variant cuts.)
- **Module-name collision (swift-transformers `Tokenizers` vs swift-tokenizers `Tokenizers`)** precludes side-by-side parity testing within one SPM resolution. Tokenizer parity uses captured golden fixtures: WU2 Sortie 6 captures encode/decode outputs against the OLD package, WU2 Sortie 12 asserts equality against the NEW package.
- **Positional `decode` overload absent in swift-tokenizers 0.4.2.** `FluxTextEncoders.swift:896` and `TokenizerTests.swift:140` currently rely on a positional `[Int]` overload that swift-transformers 1.x exposed but swift-tokenizers does not. Must be rewritten to `decode(tokenIds:skipSpecialTokens:)` during WU2 Sortie 8.
- **`applyChatTemplate` `addGenerationPrompt` default flipped `false` → `true`.** Five default-relying call sites will silently change behavior at the moment of dep swap. WU2 Sortie 8 makes every call explicit to lock in current `false` semantics.
- **Branch coordination.** `Sources/Flux2Core/Loading/ModelDownloader.swift` is dirty per `git status` on `mission/marching-relay/1`. Mission must start from a clean working tree on `main` (or another agreed merge base). Do **not** start until marching-relay is merged or rebased.
- **No swift-tokenizers version bump available.** Latest is `0.4.2` (released 2026-04-24). The pin is current.
- **Test impact.** `Flux2CoreTests`, `Flux2GPUTests`, and `FluxTextEncodersTests` may include cases that hit the old HF paths; some may require `HF_TOKEN` env setup. Post-migration, the suite must run with no `HF_TOKEN` and only `R2_PUBLIC_URL` (read-only CDN) reachable. Tests that were skipped under "no HF token" should be re-evaluated under "no R2" instead.

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|---|---|---|---|---|
| Acervo CDN Provisioning | `/Users/stovak/Projects/flux-2-swift-mlx` (operator-side `acervo ship` runs) | 4 | 0 | none |
| HF Excision + Code Migration | `/Users/stovak/Projects/flux-2-swift-mlx` | 9 | 1 | WU1 complete (all 11 manifests verified on CDN) for S7 onwards. S5 (library cleanup) and S6 (fixture capture) have no WU1 dependency and may run in parallel with WU1. |

---

## Work Unit 1 — Acervo CDN Provisioning

### Sortie 1: Inventory lock + license verification probe

**Priority**: 37.75 — highest. Blocks all 11 downstream sorties. Foundation: locks the canonical 11-model inventory + license posture used by every later sortie. Risk: medium (operator may need to click HF license).

**Entry criteria**:
- [ ] First sortie — no prerequisites
- [ ] `acervo` binary installed at `/opt/homebrew/bin/acervo` (verified)
- [ ] `hf` CLI installed at `/Users/stovak/.local/bin/hf` (verified)

**Tasks**:
1. Confirm the model inventory in this plan matches the source of truth: re-grep `repoId:` declarations in `FluxTextEncoders/Configuration/TextEncoderModelRegistry.swift` and `huggingFaceRepo` accessors in `Flux2Core/Configuration/ModelRegistry.swift`. If new variants have been added since this plan was written, append them to `docs/missions/recon-cdn-inventory.md` (created by this sortie). Variants in the "kept but not provisioned" table do NOT need to be CDN-shipped — confirm none of them have become production-required since the cut decision.
2. **Verify HF authentication and license acceptance via `hf` CLI probe** (no manual log entries):
   - Run `hf auth whoami`. Must report a logged-in user. If not, abort and instruct operator to run `hf auth login`.
   - Run `hf download black-forest-labs/FLUX.2-klein-9B README.md --local-dir /tmp/probe-klein-9B`. Success (HTTP 200, file present) means license is accepted. 403 means operator must visit `https://huggingface.co/black-forest-labs/FLUX.2-klein-9B` and click "Agree and access repository", then re-run the probe.
   - Capture the probe outcome (success/403, command output) to `docs/missions/recon-cdn-inventory.md` under `## License Verification Log`.
3. Verify operator environment has `HF_TOKEN`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` configured (existence check only — never echo values). Record `test -n "$VAR"` results.
4. Run `acervo ship --help` and `acervo ship lmstudio-community/Qwen3-4B-MLX-4bit --dry-run` (smallest non-gated repo) to verify the toolchain works end-to-end without uploading bytes. Capture the dry-run output to the recon doc.

**Exit criteria**:
- [ ] `docs/missions/recon-cdn-inventory.md` exists with the locked 11-model ship list AND the cut-variants list (5 entries) for traceability
- [ ] `hf auth whoami` reports a logged-in user
- [ ] `hf download black-forest-labs/FLUX.2-klein-9B README.md` returns HTTP 200 (license accepted)
- [ ] All three required env vars (`HF_TOKEN`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`) confirmed present (existence-only)
- [ ] At least one `acervo ship --dry-run` invocation completed without error and its output recorded
- [ ] No source code edits in `Sources/` or `Tests/` made by this sortie

---

### Sortie 2: Ship non-gated models to CDN (10 models)

**Priority**: 35.25 — second highest. Blocks all 10 downstream sorties. Foundation: produces the 10 manifests referenced by every WU2 sortie. Risk: medium (network/storage failures, ~108 GB transit). Complexity: high (10 ship operations, each with its own failure surface).

**Entry criteria**:
- [ ] Sortie 1 inventory + license verification log present
- [ ] Adequate local storage for staging — verified machine-readable: `df -k "$HOME" | awk 'NR==2 {exit ($4 < 50*1024*1024)}'` (≥50 GB free in `$HOME`; single-largest staging is `VincentGOURBIN/flux_qint_8bit` ~32 GB, plus headroom for parallel manifest writes; Acervo cleans staging after each successful upload). Record the `df -k` output in `docs/missions/cdn-ship-log.md` before shipping.

**Tasks**:
1. Ship each non-gated repo via `acervo ship <model-id>` in this order (smallest first so failures abort cheap):
   1. `lmstudio-community/Qwen3-4B-MLX-4bit` (2 GB)
   2. `lmstudio-community/Qwen3-8B-MLX-4bit` (4 GB)
   3. `lmstudio-community/Qwen3-4B-MLX-8bit` (4 GB)
   4. `aydin99/FLUX.2-klein-4B-int8` (4 GB)
   5. `lmstudio-community/Qwen3-8B-MLX-8bit` (8 GB)
   6. `black-forest-labs/FLUX.2-klein-4B` (8 GB)
   7. `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit` (13 GB)
   8. `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-6bit` (19 GB)
   9. `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit` (25 GB)
   10. `VincentGOURBIN/flux_qint_8bit` (32 GB) — subfolder-only ship via `acervo ship VincentGOURBIN/flux_qint_8bit flux-2-dev/transformer/qint8/` (confirmed: `acervo` treats the path as a normalized string and allows paths multiple directories deep). The shipped manifest will contain only the qint8 files; downstream `Acervo.ensureAvailable(...)` calls in Sortie 10 do not need a runtime filter.
2. For each ship, confirm via the manifest file list that the expected tokenizer artifacts are present (verified ahead of plan: every Mistral MLX quant has `tekken.json`, `tokenizer.json`, `tokenizer_config.json`; every Qwen3 MLX quant has `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`).
3. Capture stdout/stderr of every invocation to `docs/missions/cdn-ship-log.md`.
4. After each ship, verify `manifest.json` is fetchable from `R2_PUBLIC_URL/<slugified-repo>/manifest.json` via `curl` — record HTTP 200 and content-length.

**Exit criteria**:
- [ ] All 10 non-gated repos show "CHECK 6 passed" in the ship log
- [ ] All 10 manifests return HTTP 200 from `R2_PUBLIC_URL`
- [ ] `tekken.json` confirmed present in each of the 3 Mistral MLX manifest file lists
- [ ] Subfolder decision for `VincentGOURBIN/flux_qint_8bit` recorded in the ship log
- [ ] No code in `Sources/` or `Tests/` modified

---

### Sortie 3: Ship gated model to CDN (1 model)

**Priority**: 30.5 — high. Blocks Sortie 4 + WU2 (klein9B variant referenced by `TransformerVariant.klein9B_bf16`). Risk: high (gated repo; license-acceptance failure is the most likely failure mode in WU1). Kept separate from S2 for clean retry semantics on license-related failures.

**Entry criteria**:
- [ ] Sortie 2 complete
- [ ] License verification log shows `black-forest-labs/FLUX.2-klein-9B` accepted (HTTP 200 from probe)

**Tasks**:
1. Ship `black-forest-labs/FLUX.2-klein-9B` (~18 GB) via `acervo ship black-forest-labs/FLUX.2-klein-9B`.
2. Append the invocation output to `docs/missions/cdn-ship-log.md`.
3. Verify `manifest.json` is fetchable from `R2_PUBLIC_URL/<slug>/manifest.json` via `curl` — record HTTP 200 and content-length.

**Exit criteria**:
- [ ] `black-forest-labs/FLUX.2-klein-9B` shows "CHECK 6 passed" in the ship log
- [ ] Manifest returns HTTP 200 from `R2_PUBLIC_URL`
- [ ] Total models on CDN = 11 (10 from Sortie 2 + 1 from Sortie 3)

---

### Sortie 4: CDN read-side smoke test from clean machine

**Priority**: 28.75 — high. WU1 gate: confirms the read path Acervo will use at runtime works against the freshly provisioned manifests. Foundation: validates the slugify mapping (`org/repo` → `org_repo`) the runtime depends on. May run in parallel with Sortie 6 (no shared dependency).

**Entry criteria**:
- [ ] Sortie 3 complete; 11 manifests live on R2

**Tasks**:
1. From a clean directory with no Flux2Swift state, write a minimal one-shot Swift script (`scripts/cdn-smoke-test.swift`, deleted at end of sortie) that imports `SwiftAcervo` and calls `Acervo.fetchManifest(for:)` for each of the 11 model IDs. Do not depend on `acervo` CLI read-side commands — call the Swift API directly so this exercises the same code path the runtime will use.
2. For each model: verify the returned manifest's file count and total byte count match the ship log from Sorties 2-3.
3. Verify Acervo's `slugify(_:)` produces the expected slug for each `org/repo` (e.g., `lmstudio-community/Mistral-Small-...` → `lmstudio-community_Mistral-Small-...`). Document any surprises.
4. Pick one small model (e.g., `lmstudio-community/Qwen3-4B-MLX-4bit`) and run `Acervo.ensureAvailable(_:files:progress:)` end-to-end. Confirm files land in `Acervo.modelDirectory(for:)` and SHA-256 manifests verify.
5. Tear down the test directory.

**Exit criteria**:
- [ ] All 11 `fetchManifest` calls return success
- [ ] Manifest file counts match the ship log
- [ ] One end-to-end `ensureAvailable` succeeds and verifies
- [ ] No files left behind outside `Acervo.sharedModelsDirectory`
- [ ] Working tree commit message includes `WU1 complete; CDN provisioned`

---

## Work Unit 2 — HF Excision + Code Migration

> **Gate**: WU2 Sortie 7 does not begin until WU1 Sortie 4 reports success AND Sorties 5 + 6 are committed. Sorties 5 (library cleanup) and 6 (fixture capture) have no WU1 dependency and may run in parallel with any WU1 sortie; the supervisor serializes them as S5 → S6 because they share the SwiftPM build directory.

### Sortie 5: Library-only cleanup — remove CLI targets and customModelsDirectory

**Priority**: 26.5 — high foundation. Reduces the surface every subsequent sortie has to migrate (fewer call sites in S8/S10, fewer registry strings in S11, fewer tests in S12). Risk: low (mechanical deletion). Independent of WU1 (no CDN, no HF, no Acervo touch). Decided ahead of execution: this package is library-only; downstream end-to-end testing happens via `../SwiftVinetas` (sibling repo). Sandboxed apps cannot use a custom storage directory anyway, so `customModelsDirectory` is a misleading API surface that is deleted now rather than migrated.

**Entry criteria**:
- [ ] Working tree clean and on the mission base branch
- [ ] `Package.resolved` still pins `huggingface/swift-transformers` (cleanup operates on the OLD dependency graph; the swap happens in Sortie 7)
- [ ] No source edits made by S6 onwards yet

**Tasks**:

1. **Delete CLI targets and source**:
   - Remove the directories: `Sources/Flux2CLI/` and `Sources/FluxEncodersCLI/`.
   - In `Package.swift`: remove the `Flux2CLI` and `FluxEncodersCLI` `.executableTarget(...)` entries AND their corresponding `.executable(name:targets:)` product entries. The library products (`Flux2Core`, `FluxTextEncoders`) and their targets remain.
2. **Delete `customModelsDirectory` from runtime code**:
   - `Sources/Flux2Core/Configuration/ModelRegistry.swift`: remove the `static var customModelsDirectory: URL?` declaration (around line 415) and the `if let custom = customModelsDirectory { ... }` branch in the directory accessor (around lines 433–435). The accessor returns the unconditional default for now; Sortie 10 will replace the default with `Acervo.modelDirectory(for:)`.
   - `Sources/FluxTextEncoders/Loading/TextEncoderModelDownloader.swift`: remove the `static var customModelsDirectory: URL?` (around line 20), the `makeHubApi()` derivation that consults it (around line 27), the `reconfigureHubApi()` helper (around line 32), and the `if let custom` branches in the Mistral and Qwen3 directory accessors (around lines 40 and 51). Note: `makeHubApi()` and `reconfigureHubApi()` are slated for full deletion in Sortie 9; this sortie just removes their `customModelsDirectory` dependency.
3. **Delete tests that exclusively cover the removed surface**:
   - `Tests/FluxTextEncodersTests/TextEncoderModelDirectoryTests.swift` — entire file.
   - `Tests/Flux2CoreTests/ModelDirectoryTests.swift` — entire file.
4. **Confirm no dangling references**:
   - `grep -rn 'customModelsDirectory' Sources Tests` returns no matches.
   - `grep -rn 'Flux2CLI\|FluxEncodersCLI' Sources Tests Package.swift` returns no matches (matches inside `docs/` are acceptable).
5. **Build clean against the OLD package**:
   - `swift_package_build` for the remaining library targets must succeed.
   - `swift_package_test` for `Flux2CoreTests`, `FluxTextEncodersTests`, and `Flux2GPUTests` must succeed against the OLD package.

**Exit criteria**:
- [ ] `Sources/Flux2CLI/` and `Sources/FluxEncodersCLI/` directories do not exist
- [ ] `grep -nE 'Flux2CLI|FluxEncodersCLI' Package.swift` returns no matches
- [ ] `grep -rn 'customModelsDirectory' Sources Tests` returns no matches
- [ ] `Tests/FluxTextEncodersTests/TextEncoderModelDirectoryTests.swift` does not exist
- [ ] `Tests/Flux2CoreTests/ModelDirectoryTests.swift` does not exist
- [ ] `swift_package_build` succeeds for `Flux2Core` and `FluxTextEncoders` library targets
- [ ] `swift_package_test` for `Flux2CoreTests`, `FluxTextEncodersTests`, `Flux2GPUTests` passes
- [ ] Commit message: `Library-only cleanup: remove CLI targets, customModelsDirectory, and dependent tests`

---

### Sortie 6: Capture golden tokenizer fixtures from the old package

**Priority**: 24.75 — must run BEFORE Sortie 7 (Package.swift swap) because it depends on the old swift-transformers `Tokenizers` module being resolved. Parallelizable with WU1 S4 (no shared file or dependency). Foundation: produces the parity oracle used by Sortie 12.

**Entry criteria**:
- [ ] Sortie 5 commit exists (cleanup landed; build clean against old package)
- [ ] `Package.resolved` still pins `huggingface/swift-transformers`
- [ ] Working tree clean

**Tasks**:
1. Identify the project's tokenizer fixtures (search `Tests/FluxTextEncodersTests/` and `Tests/Fixtures/` for `tekken.json` / `tokenizer.json` references). Pick at minimum one Tekken tokenizer + one Qwen3 tokenizer that are exercised by current tests.
2. Define a deterministic prompt set (≥10 prompts: ASCII, multilingual, emojis, chat-template messages with role markers, one ≥1024-char document).
3. Add a one-shot fixture-generation test under `Tests/Fixtures/Generators/` that, against the OLD package, writes JSON files of `{prompt, encoded_token_ids[], decoded_text}` per (tokenizer × prompt).
4. Run via XcodeBuildMCP `swift_package_test` (never raw `swift test`). Commit the resulting fixtures under `Tests/Fixtures/TokenizerParity/`.
5. After fixtures are generated, delete the generator (it depends on old-package APIs that won't compile after Sortie 7).

**Exit criteria**:
- [ ] `Tests/Fixtures/TokenizerParity/*.json` exists with one file per (tokenizer × prompt) — verify via `ls Tests/Fixtures/TokenizerParity/*.json | wc -l` matches expected count
- [ ] Generator file under `Tests/Fixtures/Generators/` deleted — verify via `find Tests/Fixtures/Generators -type f 2>/dev/null` returns empty (or directory does not exist)
- [ ] Project still builds against the old package (`swift_package_build` succeeds)

---

### Sortie 7: Package.swift swap

**Priority**: 22.75 — gate sortie that establishes the new dependency graph. Foundation: every WU2 sortie after this depends on the new module resolution. Risk: medium (the old `Tokenizers` module name is reused by swift-tokenizers, so call sites won't import-break — failures are silent until call-site rewrites in S8). Build is expected to fail at this sortie's exit (call sites unchanged).

**Entry criteria**:
- [ ] WU1 complete (all 11 manifests live)
- [ ] Sortie 6 fixtures committed

**Tasks**:
1. In `Package.swift`, line 20: replace `.package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6")` with two entries:
   - `.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers", from: "0.4.2")`
   - `.package(url: "https://github.com/intrusive-memory/SwiftAcervo", from: "0.8.3")`
2. Update `FluxTextEncoders` target deps (`Package.swift:32-33`):
   - Delete `.product(name: "Hub", package: "swift-transformers")`
   - Replace `.product(name: "Transformers", package: "swift-transformers")` → `.product(name: "Tokenizers", package: "swift-tokenizers")`
   - Add `.product(name: "SwiftAcervo", package: "SwiftAcervo")`
3. Update `Flux2Core` target deps (`Package.swift:45-46`):
   - **Delete** both `.product(name: "Hub", package: "swift-transformers")` and `.product(name: "Transformers", package: "swift-transformers")` entries (verified unused in source)
   - Add `.product(name: "SwiftAcervo", package: "SwiftAcervo")`
4. Delete `Package.resolved`.
5. Trigger resolution via XcodeBuildMCP `swift_package_build` to regenerate `Package.resolved`.
6. Verify the new `Package.resolved` contains `swift-tokenizers` and `SwiftAcervo`, and contains NO `swift-transformers` reference (`swift-jinja` may remain as a transitive dep of swift-tokenizers — that is expected and acceptable since swift-jinja is not HF code).

**Exit criteria**:
- [ ] `grep -n swift-transformers Package.resolved` returns no matches
- [ ] `grep -n swift-tokenizers Package.resolved` returns at least one match
- [ ] `grep -n SwiftAcervo Package.resolved` returns at least one match
- [ ] `grep -n swift-hf-api Package.resolved` returns no matches
- [ ] `swift_package_build` MAY fail (call sites unchanged) — capture the failure log and confirm the failures are limited to the call-site files in the Codebase Reconnaissance tables (TextEncoderModelDownloader.swift, Flux2Core/ModelDownloader.swift, the tokenizer call sites, ModelRegistry display strings)
- [ ] Working tree commit exists with message `R1 manifest swap; HF excision`

---

### Sortie 8: Tokenizers API rename + applyChatTemplate audit

**Priority**: 19 — restores library-target build. Foundation: unblocks Sorties 8 and 9 (downloader rewrites). Risk: low (mechanical renames, single grep'd call sites). Behavior risk: `applyChatTemplate` default flip from `false` → `true` is a silent semantic change that this sortie locks down.

**Entry criteria**:
- [ ] Sortie 7 commit exists
- [ ] `docs/missions/recon-swift-tokenizers.md` reachable for signature reference

**Tasks**:
1. Rename `decode(tokens:)` → `decode(tokenIds:)` at the 8 source sites: `TekkenTokenizer.swift:398, 475`; `Qwen3Generator.swift:104, 134, 142, 230, 259, 267`.
2. Rewrite the positional-arg call at `FluxTextEncoders.swift:896` (`tokenizer.decode(tokens, skipSpecialTokens: skipSpecialTokens)`) → `tokenizer.decode(tokenIds: tokens, skipSpecialTokens: skipSpecialTokens)`. Same conversion at `TokenizerTests.swift:140`.
3. Rename `AutoTokenizer.from(modelFolder:)` → `AutoTokenizer.from(directory:)` at `FluxTextEncoders.swift:201` and `TekkenTokenizer.swift:232`. Verify no caller passes `hubApi:` or `strict:` parameters (both removed in swift-tokenizers).
4. **`applyChatTemplate` audit (preserve current behavior):** Add explicit `addGenerationPrompt: false` to every default-relying site to lock in pre-migration semantics:
   - `Sources/FluxTextEncoders/Tokenizer/TekkenTokenizer.swift:473, 510`
   - `Tests/FluxTextEncodersTests/TokenizerTests.swift:77, 93, 108`
   - `Tests/FluxTextEncodersTests/FluxTextEncodersTests.swift:204`
   The site at `EmbeddingExtractor.swift:192` already passes `false` — leave it.
5. Verify `Sendable` conformance on stored `Tokenizer` properties (`TekkenTokenizer.swift:42`, `KleinEmbeddingExtractor.swift:23`) compiles cleanly under Swift 6.2.
6. After edits: `swift_package_build` for `FluxTextEncoders` and `Flux2Core` library targets must succeed. The two downloader files remain broken (Sorties 8 and 9).

**Exit criteria**:
- [ ] `grep -rn 'decode(tokens:' Sources Tests` returns no matches
- [ ] `grep -rEn 'tokenizer\.decode\([a-zA-Z_]' Sources Tests` (positional first-arg form, BSD-grep portable) returns no matches in `Sources/FluxTextEncoders/` or `Tests/`
- [ ] `grep -rn 'from(modelFolder:' Sources Tests` returns no matches
- [ ] `grep -rn 'applyChatTemplate(messages:' Sources Tests | grep -v 'addGenerationPrompt'` returns no matches
- [ ] `swift_package_build` succeeds for `FluxTextEncoders` and `Flux2Core` library targets (test/CLI/App MAY still fail due to downloaders)
- [ ] No new `Sendable` warnings
- [ ] Commit message: `R2.1-R2.4, R3.1; tokenizers API rename`

---

### Sortie 9: Rewrite TextEncoderModelDownloader against Acervo

**Priority**: 18.25 — first runtime HF excision. Risk: high (replaces Hub-based download paths; adds `notProvisionedOnCDN` error case; deletes the `tekken.json` fallback). Complexity: high (one large file rewrite + error-case design). Foundation: establishes the Acervo integration pattern reused by Sortie 10.

**Entry criteria**:
- [ ] Sortie 8 commit exists
- [ ] WU1 manifests for all `lmstudio-community/Qwen3-*` and `mistralai/Mistral-Small-3.2-...` repos verified live

**Tasks**:
1. In `Sources/FluxTextEncoders/Loading/TextEncoderModelDownloader.swift`:
   - Replace `import Hub` with `import SwiftAcervo`.
   - Delete the legacy HF cache path resolution (lines 87-110, the `~/.cache/huggingface/hub/...` block).
   - Replace `hubApi: HubApi` static + `makeHubApi()` + `reconfigureHubApi()` with `Acervo` static-API calls. `customModelsDirectory` semantics map to Acervo's `sharedModelsDirectory` — verify whether SwiftAcervo exposes a way to override the storage root; if not, document a deviation in `customModelsDirectory`'s behavior in the doc comment.
   - Replace the `download(model:progress:)` body that uses `hubApi.snapshot(from:matching:)` with `Acervo.ensureAvailable(modelInfo.repoId, files: <explicit file list>, progress: ...)`. The file list per model comes from the manifest written in WU1 — for Mistral MLX quants the list includes `tekken.json`, `tokenizer.json`, `tokenizer_config.json`, the safetensors shards, and config files (verified present in each MLX quant repo ahead of plan).
   - **Delete `ensureTekkenJson(at:progress:)` entirely** (lines 252-283 plus the call sites that invoke it at lines 210, 242, 264). This method is dead code: the hardcoded `huggingface.co/.../tekken.json` fetch was a defensive fallback for the case where downloaded model dirs lacked `tekken.json` — but every lmstudio-community MLX quant ships `tekken.json` directly (verified by HF API probe across 4-bit, 6-bit, 8-bit). With explicit-file-list shipping via Acervo, `tekken.json` is always present.
   - Add handling for `ModelVariant.bf16` (text encoder): when called for the cut variant, throw `TextEncoderModelDownloaderError.notProvisionedOnCDN(variant: .bf16)` with a message naming the variant and stating it will be re-enabled in a follow-up CDN mission.
   - Replace `findModelPath(for:)` with `Acervo.modelDirectory(for: model.repoId)` + `Acervo.isModelAvailable(_:)` checks. Drop `verifyShardedModel(at:)` if Acervo's manifest verification covers it; otherwise keep and call after `ensureAvailable`.
   - Same migration for `downloadQwen3(_:progress:)` and `findQwen3ModelPath(for:)` (the parallel Qwen3 path starting around line 296).
2. Public API of `TextEncoderModelDownloader` should remain stable: `download(_:progress:)`, `downloadQwen3(_:progress:)`, `findModelPath(for:)`, `findQwen3ModelPath(for:)`, `verifyShardedModel(at:)`, `modelsDirectory`, `customModelsDirectory`. If any caller-facing signature must change, document the break in the commit message.
3. `swift_package_build` for `FluxTextEncoders` (full target) must succeed.

**Exit criteria**:
- [ ] `grep -n 'import Hub' Sources/FluxTextEncoders/Loading/TextEncoderModelDownloader.swift` returns no matches
- [ ] `grep -in 'huggingface' Sources/FluxTextEncoders/Loading/TextEncoderModelDownloader.swift` returns no matches
- [ ] `grep -nE 'hubApi|HubApi|snapshot\(from:' Sources/FluxTextEncoders/Loading/TextEncoderModelDownloader.swift` returns no matches
- [ ] `swift_package_build` succeeds for `FluxTextEncoders` (full target including the `FluxEncodersCLI` and `Flux2App` consumers, if they compile after this sortie)
- [ ] Commit message: `R2.5; TextEncoderModelDownloader → SwiftAcervo`

---

### Sortie 10: Rewrite Flux2Core/ModelDownloader against Acervo

**Priority**: 15.5 — largest single rewrite in the mission (deletes the hand-rolled HF API client). Risk: high (touches transformer + VAE + klein download paths; introduces `notProvisionedOnCDN` for 4 cut variants; updates 8 enumerated callers). Complexity: high. Could be split into 9a (downloader rewrite + new error type) and 9b (caller updates) if execution shows context pressure — kept atomic because the new error type and caller updates must land together to keep the build green.

**Entry criteria**:
- [ ] Sortie 9 commit exists
- [ ] WU1 manifests for all transformer + VAE + klein repos verified live

**Tasks**:
1. In `Sources/Flux2Core/Loading/ModelDownloader.swift`:
   - **Delete** the entire hand-rolled HF API client: `fetchFileList(repoId:subfolder:)` (around line 419), the `download(filePath:repoId:...)` per-file download (around line 475), and any helper that hits `huggingface.co`.
   - Replace the `download(_:progress:)` method body with `Acervo.ensureAvailable(repoId, files: <explicit list>, progress: ...)`. For repos with `huggingFaceSubfolder` (e.g., `qint8` → `flux-2-dev/transformer/qint8/`), use the file list resolution decided in WU1 Sortie 2 task 1 (subfolder-only ship vs. whole-repo-ship-with-runtime-filter).
   - Replace `findModelPath(for:)` with `Acervo.modelDirectory(for: repoId)` lookup.
   - Replace `repoId(for:)` with the same logic but document that the returned string is now an Acervo model ID (slugified internally by SwiftAcervo).
   - Keep `verifyModel(at:)` — this is local-side shard completeness verification, complementary to Acervo's manifest verification.
2. **Add `notProvisionedOnCDN` handling for cut variants.** Define a new error case (e.g., `Flux2DownloadError.notProvisionedOnCDN(variant: TransformerVariant)`) and throw it from `download(_:progress:)` when called with any of:
   - `TransformerVariant.bf16` (was `black-forest-labs/FLUX.2-dev`)
   - `TransformerVariant.klein4B_base_bf16` (was `black-forest-labs/FLUX.2-klein-base-4B`)
   - `TransformerVariant.klein9B_base_bf16` (was `black-forest-labs/FLUX.2-klein-base-9B`)
   - `TransformerVariant.klein9B_kv_bf16` (was `black-forest-labs/FLUX.2-klein-9b-kv`)
   The error message should name the variant and state: "Variant not yet provisioned on Acervo CDN. Track follow-up CDN provisioning mission in docs/missions/."
3. Add a `var isProvisionedOnCDN: Bool` property on `TransformerVariant` (returns `false` for the four cut variants, `true` for the rest) so UI iteration over `allCases` can gray them out.
4. Update the concrete `Flux2ModelDownloader` callers — re-grep at sortie start with `grep -rn 'Flux2ModelDownloader(' Sources/` to get the current list (Sortie 5 deleted `Sources/Flux2CLI/`, so the callers from there are gone). Expected post-cleanup callers:
   - `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift:141`
   - `Sources/Flux2Core/Training/LoRATrainingHelper.swift:470`
   - `Sources/Flux2App/ViewModels/ModelManager.swift:332, 374`
   Each must handle the new error gracefully — surfacing "not yet available" rather than crashing. Also audit `Tests/Flux2CoreTests/Flux2CoreTests.swift:406` (which references `TransformerVariant.bf16`) — adjust to skip download or assert the not-provisioned error.
5. Goal: no public API change to `Flux2ModelDownloader.download(_:progress:)` other than the new error case. Internal swap.
6. `swift_package_build` must succeed for ALL remaining targets in `Package.swift` (libraries + the `Flux2App` target + test targets; CLIs were deleted in Sortie 5).

**Exit criteria**:
- [ ] `grep -rin 'huggingface\.co' Sources/` returns no matches — every runtime HF URL is gone from `Sources/`
- [ ] `grep -rin 'huggingface\.co/api' Sources/` returns no matches
- [ ] `grep -rn 'fetchFileList' Sources/` returns no matches
- [ ] `swift_package_build` succeeds for ALL targets
- [ ] Commit message: `Flux2Core/ModelDownloader → SwiftAcervo; remove hand-rolled HF API client`

---

### Sortie 11: Strip HF runtime strings; add README acknowledgments

**Priority**: 8 — completes the runtime HF excision (removes display strings + accessor renames) and lands the credits surface. Risk: low (mechanical renames + README authoring). Verification: `grep -rn 'huggingface' Sources/` returns only origin/license attribution comments.

> **Note**: GUI About-panel work is explicitly out of scope for this mission per user direction (deferred to a separate workstream). This sortie's credits work is README-only.

**Entry criteria**:
- [ ] Sortie 10 commit exists; full project builds

**Tasks**:
1. In `Sources/Flux2Core/Configuration/ModelRegistry.swift`:
   - **Remove** the `huggingFaceURL` accessors at lines 115, 268, 329 (no runtime HF URLs).
   - **Rename** `huggingFaceRepo` → `repoId` and `huggingFaceSubfolder` → `repoSubfolder` (or similar HF-free names). Update all callers in `Sources/`. Doc comments may attribute model weight origin to HuggingFace (e.g., "Acervo model ID; weights originally from <hf-org>/<hf-repo>") — this is allowed under the credits policy.
   - Where the prior `huggingFaceURL` was consumed by UI code, remove the link or replace with an Acervo CDN manifest URL if SwiftAcervo exposes one.
2. Audit `Sources/FluxTextEncoders/Configuration/TextEncoderModelRegistry.swift` similarly — the `huggingFaceURL` accessors at lines 84, 166, 208, 247 must go. The `repoId` field stays.
3. **Update README.md with an `## Acknowledgments` section** containing:
   - Open-source dependency list with licenses: `mlx-swift` (MIT), `swift-argument-parser` (Apache-2.0), `swift-tokenizers` (Apache-2.0), `SwiftAcervo` (verify its license), `universal`/YAML (verify).
   - HuggingFace credit line: "Model weights for the variants distributed via this project's Acervo CDN were originally published on HuggingFace by Mistral AI, the Black Forest Labs team, the LM Studio community, and individual contributors (`VincentGOURBIN`, `aydin99`). Intrusive Memory mirrors these weights via Cloudflare R2 for distribution to Flux2Swift users."
   - Per-model attribution table mapping each Acervo-provisioned model to its HuggingFace origin and license.

**Exit criteria**:
- [ ] `grep -rin 'huggingface' Sources/ | grep -vE '^\s*//|^\s*\*|/\*|\*/' ` returns no matches — every remaining `huggingface` mention in `Sources/` lives on a comment line (line starts with `//`, ` *`, or contains `/*`/`*/`)
- [ ] `grep -rnE 'huggingFaceURL|huggingFaceRepo|huggingFaceSubfolder' Sources/` returns no matches (renamed)
- [ ] README.md contains the `## Acknowledgments` section with the dep list, the HuggingFace credit line, and the per-model attribution table
- [ ] `swift_package_build` succeeds for all targets
- [ ] Commit message: `Strip HF runtime strings; README acknowledgments`

---

### Sortie 12: Test sweep + tokenizer parity verification

**Priority**: 5 — proves the migration didn't regress tokenizer behavior. Risk: medium (hidden parity gaps surface here; weakening assertions to "pass tests" is forbidden). Foundation: gates the CI verification in Sortie 13.

**Entry criteria**:
- [ ] Sortie 11 commit exists; full project builds clean
- [ ] Sortie 6 golden fixtures present at `Tests/Fixtures/TokenizerParity/`

**Tasks**:
1. Update `Tests/FluxTextEncodersTests/TokenizerTests.swift` for the `decode` rename (if not already done in Sortie 8 task 2). Re-run; assert green.
2. Add `Tests/FluxTextEncodersTests/TokenizerParityTests.swift`:
   - Loads each Sortie 6 fixture JSON.
   - Asserts new-package `encode(text:)` output equals recorded `expected_token_ids`.
   - Asserts new-package `decode(tokenIds:)` output equals recorded `expected_decoded_text`.
   - Includes a printout of prompt + diff on any mismatch.
3. Run `swift_package_test` for `FluxTextEncodersTests`. All assertions must pass.
4. Run `swift_package_test` for `Flux2CoreTests` and `Flux2GPUTests`. Investigate any new failures, especially around offline/cache-only paths. Acervo's `isModelAvailable` semantics differ from the old `Hub` cache — fix tests to use Acervo APIs, do not weaken assertions.
5. Audit any test helper that calls out to `huggingface.co` or expects `HF_TOKEN`. Replace with Acervo-based fixtures or skip with a clear reason.
6. Mark `TokenizerParityTests.swift` with a comment indicating Sortie 13 will delete it after CI green.

**Exit criteria**:
- [ ] `swift_package_test` succeeds for `FluxTextEncodersTests`, `Flux2CoreTests`, `Flux2GPUTests`
- [ ] Parity test exists and passes
- [ ] No `huggingface.co` fetches in any test code path (`grep -rn 'huggingface' Tests/` returns only credits/comment references)
- [ ] No assertions weakened relative to pre-migration (verify via `git diff Tests/`)
- [ ] Commit message: `R4.1, R4.2, R4.3; test sweep + parity`

---

### Sortie 13: CI verification + parity test cleanup

**Priority**: 1.75 — terminal sortie. Risk: low (CI may surface OS-version or simulator drift, but local green is a strong predictor). Cleanup: removes the parity oracle so it does not rot in-tree.

**Entry criteria**:
- [ ] Sortie 12 commit exists; full local test suite green
- [ ] Mission branch pushed to remote

**Tasks**:
1. Open or update a draft PR for the mission branch.
2. **Verify CI does NOT have `HF_TOKEN` configured** (per the "huggingface-free" goal). If any GitHub Actions secret named `HF_TOKEN` exists in the repo, remove it and confirm the workflow still passes. CI's runtime requirement reduces to: macOS 26 runner, Swift 6.2+, internet access to `R2_PUBLIC_URL` (read-only, no auth).
3. Wait for the GitHub Actions matrix to complete via `gh pr checks --watch`.
4. If CI fails, triage and fix on the same branch. Do **not** mark this sortie complete with red CI.
5. Once CI is fully green, delete `TokenizerParityTests.swift` and `Tests/Fixtures/TokenizerParity/`. Re-run `swift_package_test` to confirm deletion does not break anything else. Push.
6. Verify final `Package.resolved` is committed cleanly (no untracked changes after a fresh `swift package resolve`).
7. Update the PR description with: (a) link to `recon-swift-tokenizers.md`, (b) link to `cdn-ship-log.md`, (c) summary of API changes, (d) note that the parity test was added and removed within this PR, (e) link to the new credits surface.

**Exit criteria**:
- [ ] `gh pr checks <PR>` shows all checks green
- [ ] No `HF_TOKEN` secret configured at the repo level (verify via `gh secret list`)
- [ ] `Tests/FluxTextEncodersTests/TokenizerParityTests.swift` does not exist
- [ ] `Tests/Fixtures/TokenizerParity/` does not exist
- [ ] `swift_package_test` still passes after parity test deletion
- [ ] PR description includes the mission summary
- [ ] No untracked changes after a fresh `swift package resolve`

---

## Parallelism Structure

**Critical path** (length 11 sorties on the longest chain): S1 → S2 → S3 → S4 → S7 → S8 → S9 → S10 → S11 → S12 → S13. S5 (cleanup) and S6 (fixtures) sit before S7 but neither depends on WU1, so they can begin as soon as the operator starts the mission — they do not extend wall-clock beyond WU1.

**Parallel execution groups**:

- **Group 1 — WU1 sequential (CDN provisioning)**: S1 → S2 → S3 → S4. No intra-group parallelism (each ship is a precondition for the next stage's verification or list completeness). Operator-credentialed work; supervising agent only.
- **Group 2 — Pre-swap WU2 prep (overlapping WU1)**: S5 (library cleanup) and S6 (fixture capture) both have zero file overlap with WU1 sorties (different directories: WU1 writes to `docs/missions/` + the operator's R2 bucket; S5 deletes `Sources/Flux2CLI/` + tests; S6 writes to `Tests/Fixtures/`). Both can dispatch concurrently with any WU1 sortie. Order between S5 and S6 is independent functionally but they share the SwiftPM build directory, so the supervisor serializes them: **S5 → S6**, with the pair running alongside WU1's S2/S3/S4. **Supervising agent only** (both run `swift_package_build`/`swift_package_test`; sub-agents must not build).
- **Group 3 — WU2 sequential (code migration)**: S7 → S8 → S9 → S10 → S11 → S12 → S13. Each sortie modifies code that the next depends on (Package.swift swap, then call-site rewrites, then downloaders, then registry strings, then tests, then CI). No intra-group parallelism. **Hard gate**: S7 cannot start until WU1 S4 reports success AND S5+S6 are committed.

**Agent constraints**:
- **Supervising agent**: handles every sortie in this mission. All WU2 sorties run `swift_package_build` or `swift_package_test`. WU1 sorties run operator-credentialed CLI tools (`acervo ship`, `hf`, `curl`, R2 access) that should not be delegated.
- **Sub-agents (up to 4)**: not used. The mission shape is heavily sequential and build-bound. Allocating sub-agents here would add coordination overhead without wall-clock benefit.

**Maximum parallelism**: 2 simultaneous sorties at any moment (one WU1 ship/verify in flight + one WU2 prep sortie from {S5, S6}).

**Missed opportunities**: none material. The dependency graph is linear by construction:
- CDN must be provisioned before any code change can be tested against the new path (WU1 → WU2 S7 layer gate).
- Within WU2 post-swap, each sortie's exit state (Package.swift swap, then call-site rewrites, then downloader rewrites, then registry strings) is the next sortie's entry precondition.
- Splitting S10 into 10a (downloader rewrite) + 10b (caller updates) was considered and rejected: the new error case must land with its callers to keep the build green.

---

## Open Questions & Missing Documentation

These items have documented fallbacks and do NOT block execution start. Each is resolved inside the sortie that needs it; if the fallback fires, the decision is captured in the sortie's commit message and recon docs.

| Sortie | Issue Type | Description | Resolution Plan |
|---|---|---|---|
| WU2 S9 (task 1) | Open question | Does Acervo's manifest verification cover what the legacy `verifyShardedModel(at:)` checks (per-shard SHA + `.safetensors.index.json` presence + total byte count)? | Read `SwiftAcervo` 0.8.3+ source. If covered, drop `verifyShardedModel(at:)`. If partial, keep it and call it after `Acervo.ensureAvailable`. Decision goes in the sortie's commit message. |
| WU2 S11 (task 3) | Vague criterion | "verify its license" for `SwiftAcervo`, `universal`/YAML | Auto-resolved at sortie start: `cat .build/checkouts/SwiftAcervo/LICENSE` (and equivalent for the YAML dep). Record the license name in the README acknowledgments table. Not a research task — mechanical lookup. |

**Vague criteria scan**: clean — every exit criterion in the plan is grep- or build-verifiable.

**Missing documentation**: none. `docs/missions/recon-swift-tokenizers.md` (existing) covers the swift-tokenizers API surface. `docs/missions/recon-cdn-inventory.md` is created by Sortie 1 (per its task list) and consumed by all later sorties — its absence at start is expected.

**Requires manual review before execution**: 0 issues. Plan is unblocked.

---

## Summary

| Metric | Value |
|---|---|
| Work units | 2 |
| Total sorties | 13 (4 in WU1, 9 in WU2) |
| Dependency structure | Layered: WU1 (CDN provisioning) → WU2 post-swap (S7+). WU2 pre-swap sorties (S5 cleanup, S6 fixtures) are independent of WU1 and can run alongside it. |
| Critical path length | 11 sorties (S1→S2→S3→S4→S7→S8→S9→S10→S11→S12→S13; S5+S6 overlap with WU1) |
| Maximum simultaneous sorties | 2 (one WU1 sortie + one of {S5, S6}) |
| Models to mirror to CDN | 11 (~137 GB) |
| Variants kept in registry but not provisioned | 5 (1 text-encoder BF16 + 4 transformer variants) |
| Gated repos requiring license click-through | 1 (`black-forest-labs/FLUX.2-klein-9B`) |
| Source requirements doc | `docs/missions/tokenizer-migration.md` (original; expanded by user direction during plan iteration) |
| Requirements covered | R1.1–R1.4, R2.1–R2.6, R3.1, R4.1–R4.3, R5.1, R6.1–R6.2, R7 + full HF excision |
| Open questions blocking execution | 0 (4 in-sortie questions with documented fallbacks; see above) |

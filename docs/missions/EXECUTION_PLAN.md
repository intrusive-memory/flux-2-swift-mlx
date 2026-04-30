---
feature_name: OPERATION FAREWELL EMBRACE
starting_point_commit: 72a3eb6f3fe2b295e0eec5c4a8eb02ea1865e55f
mission_branch: mission/farewell-embrace/01
iteration: 1
---

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
| About / credits surface | **README only for this mission.** The app's GUI About panel work is deferred to a separate workstream. Sortie 20 adds a `## Acknowledgments` section to the project README crediting HuggingFace as the origin of mirrored model weights and noting Intrusive Memory's R2 redistribution. |
| `acervo ship` subfolder support | **Confirmed: `acervo` treats the path as a normalized string and accepts paths multiple directories deep.** WU1 Sortie 11 ships `VincentGOURBIN/flux_qint_8bit` as `acervo ship VincentGOURBIN/flux_qint_8bit flux-2-dev/transformer/qint8/`. No runtime filter needed. |
| Package shape | **Library-only.** Both `Sources/Flux2CLI/` and `Sources/FluxEncodersCLI/` are deleted in Sortie 14. The package's executable products go away; only `Flux2Core` and `FluxTextEncoders` libraries remain. End-to-end CLI exercise is delegated to the sibling `../SwiftVinetas` repo. |
| `customModelsDirectory` | **Removed entirely in Sortie 14.** Sandboxed apps cannot use a custom storage directory (container-bound), so the override was misleading API. Storage location is `Acervo.sharedModelsDirectory` post-migration; no override hook is exposed. The two test files that exclusively covered this surface (`Tests/FluxTextEncodersTests/TextEncoderModelDirectoryTests.swift`, `Tests/Flux2CoreTests/ModelDirectoryTests.swift`) are deleted. |
| WU1 ship granularity | **One sortie per repo ship.** Each `acervo ship <model-id>` is its own agent dispatch with a binary pass/fail outcome. Ships execute strictly sequentially (operator can only run one ship at a time; staging dir + HF bandwidth are shared). Smallest payload first to fail cheap. |

## Codebase Reconnaissance (verified ahead of plan)

### Tokenization surface

| Concern | Confirmed locations |
|---|---|
| `import Tokenizers` | `FluxTextEncoders.swift:14`, `Tokenizer/TekkenTokenizer.swift:14`, `Embeddings/KleinEmbeddingExtractor.swift:16`, `Generation/Qwen3Generator.swift:12` |
| `decode(tokens:)` keyword form (rename to `decode(tokenIds:)`) | `TekkenTokenizer.swift:398, 475`; `Qwen3Generator.swift:104, 134, 142, 230, 259, 267` (8 sites) |
| `decode(_:skipSpecialTokens:)` positional form (must rewrite — no positional overload in swift-tokenizers 0.4.2) | `FluxTextEncoders.swift:896`; `Tests/FluxTextEncodersTests/TokenizerTests.swift:140` |
| `from(modelFolder:)` (rename to `from(directory:)`; new signature drops `hubApi:` and `strict:` parameters) | `FluxTextEncoders.swift:201`, `Tokenizer/TekkenTokenizer.swift:232` |
| `applyChatTemplate(messages:)` call sites (audit `addGenerationPrompt` default flip) | Explicit `false`: `Embeddings/EmbeddingExtractor.swift:193`. Default-relying (must be made explicit): `TekkenTokenizer.swift:473, 510`; `TokenizerTests.swift:77, 93, 108`; `FluxTextEncodersTests.swift:204` |

### HuggingFace runtime surface (full elimination targets)

| Concern | Confirmed locations |
|---|---|
| `import Hub` | `FluxTextEncoders/Loading/TextEncoderModelDownloader.swift:7` (only `import`-level site) |
| `HubApi.snapshot(from:matching:)` calls | `TextEncoderModelDownloader.swift:224` (Mistral) and `:320` (Qwen3) |
| `HubApi(downloadBase:)` initializer | `TextEncoderModelDownloader.swift:29` |
| Hardcoded `huggingface.co` URL (URLSession direct fetch) | `TextEncoderModelDownloader.swift:267` — `https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/resolve/main/tekken.json`. **The entire `ensureTekkenJson(at:progress:)` method (lines 252-283) is deleted, not migrated** — `tekken.json` ships in every lmstudio-community MLX quant directly (verified by HF API probe), so the fallback is never needed. |
| Hand-rolled HF API client (NOT using `Hub`) | `Flux2Core/Loading/ModelDownloader.swift:419-480` — hits `huggingface.co/api/models/<repo>/tree/main` and `huggingface.co/<repo>/resolve/main/<file>` directly via URLSession. Largest single rewrite in the mission. |
| `huggingFaceRepo` / `huggingFaceURL` / `huggingFaceSubfolder` accessors (display strings) | `Flux2Core/Configuration/ModelRegistry.swift:34, 117, 257, 273, 321, 334` |
| Legacy HF cache path resolution | `TextEncoderModelDownloader.swift:87-110` (`~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/...`) — **delete entirely** per locked decision 7. |
| `repoId` accessor/field on `ModelInfo` / `Qwen3ModelInfo` | `TextEncoderModelRegistry.swift` (verified: `repoId` accessor at lines 69, 155; stored property at lines 198, 236). **Kept** — Acervo's `slugify(_:)` converts `org/repo` → `org_repo` transparently. |
| `repoId` on `TransformerVariant` | **Does NOT exist yet** in `Flux2Core/Configuration/ModelRegistry.swift` (verified: only `huggingFaceRepo` accessors at lines 34, 257, 321). Sortie 20 introduces `repoId` here by renaming `huggingFaceRepo` — no naming collision. |

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

- **`acervo ship` requires `HF_TOKEN` plus operator license acceptance for the one gated repo (`black-forest-labs/FLUX.2-klein-9B`).** Acceptance URL: `https://huggingface.co/black-forest-labs/FLUX.2-klein-9B`. Failed CHECK 1 in Sortie 12 is recoverable — accept the license, re-run the sortie.
- **HF download cost.** Sorties 2–12 of WU1 download ~137 GB from huggingface.co before re-uploading to R2. This is the operator's bandwidth/storage. Plan for ≥3 hours of wall clock on a typical broadband uplink, plus R2 upload time. Each ship is its own sortie so partial completion is preserved across operator sessions — failure of, say, Sortie 10 (Mistral 8-bit, 25 GB) does not invalidate the prior 8 successful ships.
- **Module-name collision (swift-transformers `Tokenizers` vs swift-tokenizers `Tokenizers`)** precludes side-by-side parity testing within one SPM resolution. Tokenizer parity uses captured golden fixtures: WU2 Sortie 15 captures encode/decode outputs against the OLD package, WU2 Sortie 21 asserts equality against the NEW package.
- **Positional `decode` overload absent in swift-tokenizers 0.4.2.** `FluxTextEncoders.swift:896` and `TokenizerTests.swift:140` currently rely on a positional `[Int]` overload that swift-transformers 1.x exposed but swift-tokenizers does not. Must be rewritten to `decode(tokenIds:skipSpecialTokens:)` during WU2 Sortie 17.
- **`applyChatTemplate` `addGenerationPrompt` default flipped `false` → `true`.** Five default-relying call sites will silently change behavior at the moment of dep swap. WU2 Sortie 17 makes every call explicit to lock in current `false` semantics.
- **Branch coordination.** `Sources/Flux2Core/Loading/ModelDownloader.swift` is dirty per `git status` on `mission/marching-relay/1`. Mission must start from a clean working tree on `main` (or another agreed merge base). Do **not** start until marching-relay is merged or rebased.
- **No swift-tokenizers version bump available.** Latest is `0.4.2` (released 2026-04-24). The pin is current.
- **Test impact.** `Flux2CoreTests`, `Flux2GPUTests`, and `FluxTextEncodersTests` may include cases that hit the old HF paths; some may require `HF_TOKEN` env setup. Post-migration, the suite must run with no `HF_TOKEN` and only `R2_PUBLIC_URL` (read-only CDN) reachable. Tests that were skipped under "no HF token" should be re-evaluated under "no R2" instead.

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|---|---|---|---|---|
| Acervo CDN Provisioning | `/Users/stovak/Projects/flux-2-swift-mlx` (operator-side `acervo ship` runs) | 13 | 0 | none |
| HF Excision + Code Migration | `/Users/stovak/Projects/flux-2-swift-mlx` | 9 | 1 | WU1 complete (all 11 manifests verified on CDN) for Sortie 16 onwards. Sortie 14 (library cleanup) and Sortie 15 (fixture capture) have no WU1 dependency and may run in parallel with WU1. |

---

## Work Unit 1 — Acervo CDN Provisioning

> **Granularity**: One sortie per repo ship (per locked decision). Each `acervo ship <model-id>` is its own agent dispatch with binary pass/fail. Sorties 2–12 execute strictly sequentially (operator can only run one ship at a time; staging dir and HF bandwidth are shared). Order: smallest non-gated payload first (fail cheap), then gated klein-9B last, then smoke test.

### Sortie 1: Inventory lock + license verification probe

**Priority**: 41 — highest. Blocks all 21 downstream sorties. Foundation: locks the canonical 11-model inventory + license posture used by every later sortie. Risk: medium (operator may need to click HF license).

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

### Per-ship sortie template (applied to Sorties 2–12)

Each ship sortie shares this structure. Fields that vary per repo (model ID, size, df threshold, special notes) are listed in each individual sortie below.

**Tasks** (every ship sortie):
1. **Preflight df check** — verify ≥ `<df_threshold>` GB free in `$HOME` via `df -k "$HOME" | awk 'NR==2 {exit ($4 < <df_threshold>*1024*1024)}'`. Record the `df -k "$HOME"` output to `docs/missions/cdn-ship-log.md` under a per-sortie heading before shipping.
2. **Ship the repo** via `acervo ship <model-id>` (or for the qint8 subfolder ship, the two-arg form). Capture stdout/stderr verbatim to the ship log under the per-sortie heading.
3. **Verify manifest is live**: `curl -fsS -o /tmp/manifest-<slug>.json -w "%{http_code}\n" "$R2_PUBLIC_URL/<slugified-repo>/manifest.json"`. Record exit code and HTTP code to the ship log.
4. **Verify expected tokenizer artifacts are in the manifest** (per-repo: `tekken.json` for Mistral; `tokenizer.json` for Qwen3; n/a for transformer/VAE).

**Exit criteria** (every ship sortie):
- [ ] One new `CHECK 6 passed` entry in `docs/missions/cdn-ship-log.md` under this sortie's heading (verifiable: `awk '/^## Sortie <N>/,/^## Sortie/' docs/missions/cdn-ship-log.md | grep -c "CHECK 6 passed"` returns ≥ 1)
- [ ] `curl -fsS -o /dev/null -w "%{http_code}" "$R2_PUBLIC_URL/<slug>/manifest.json"` returns `200`
- [ ] Tokenizer-artifact grep over the captured manifest passes (per-repo specifics in the sortie)
- [ ] `git status -s -- Sources/ Tests/` returns empty (no source/test edits by this sortie)

---

### Sortie 2: Ship `lmstudio-community/Qwen3-4B-MLX-4bit` (2 GB)

**Priority**: 38 — first ship; validates the shipping pipeline on the smallest non-gated payload. Risk: low (smallest payload, no gating, ungated org).

**Entry criteria**:
- [ ] Sortie 1 complete; license probe + env vars + `--dry-run` recorded
- [ ] `df_threshold = 6` (≥6 GB free in `$HOME`; ~2 GB ship + 2× headroom)

**Repo-specific**:
- Model ID: `lmstudio-community/Qwen3-4B-MLX-4bit`
- Tokenizer-artifact assertion: manifest contains `"tokenizer.json"`, `"tokenizer_config.json"`, `"special_tokens_map.json"`, `"added_tokens.json"`

---

### Sortie 3: Ship `lmstudio-community/Qwen3-8B-MLX-4bit` (4 GB)

**Priority**: 37.5 — second ship; second-smallest. Risk: low.

**Entry criteria**:
- [ ] Sortie 2 commit/log complete; previous manifest verified live
- [ ] `df_threshold = 10`

**Repo-specific**:
- Model ID: `lmstudio-community/Qwen3-8B-MLX-4bit`
- Tokenizer-artifact assertion: manifest contains `"tokenizer.json"`

---

### Sortie 4: Ship `lmstudio-community/Qwen3-4B-MLX-8bit` (4 GB)

**Priority**: 37 — third ship. Risk: low.

**Entry criteria**:
- [ ] Sortie 3 commit/log complete; previous manifest verified live
- [ ] `df_threshold = 10`

**Repo-specific**:
- Model ID: `lmstudio-community/Qwen3-4B-MLX-8bit`
- Tokenizer-artifact assertion: manifest contains `"tokenizer.json"`

---

### Sortie 5: Ship `aydin99/FLUX.2-klein-4B-int8` (4 GB)

**Priority**: 36.5 — fourth ship; first transformer-family ship. Risk: low (community ungated).

**Entry criteria**:
- [ ] Sortie 4 commit/log complete; previous manifest verified live
- [ ] `df_threshold = 10`

**Repo-specific**:
- Model ID: `aydin99/FLUX.2-klein-4B-int8`
- Tokenizer-artifact assertion: n/a (transformer; no tokenizer expected). Skip the artifact grep step but record "no-tokenizer-expected" in the ship log.

---

### Sortie 6: Ship `lmstudio-community/Qwen3-8B-MLX-8bit` (8 GB)

**Priority**: 36 — fifth ship. Risk: low.

**Entry criteria**:
- [ ] Sortie 5 commit/log complete; previous manifest verified live
- [ ] `df_threshold = 18`

**Repo-specific**:
- Model ID: `lmstudio-community/Qwen3-8B-MLX-8bit`
- Tokenizer-artifact assertion: manifest contains `"tokenizer.json"`

---

### Sortie 7: Ship `black-forest-labs/FLUX.2-klein-4B` (8 GB)

**Priority**: 35.5 — sixth ship; this is also the canonical VAE source for ALL Flux.2 variants. Risk: low (Apache-2.0, ungated).

**Entry criteria**:
- [ ] Sortie 6 commit/log complete; previous manifest verified live
- [ ] `df_threshold = 18`

**Repo-specific**:
- Model ID: `black-forest-labs/FLUX.2-klein-4B`
- Tokenizer-artifact assertion: n/a (transformer/VAE). Record "no-tokenizer-expected; this manifest is the VAE source for downstream Flux.2 variants" in the ship log.

---

### Sortie 8: Ship `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit` (13 GB)

**Priority**: 35 — seventh ship; first Mistral. Risk: low (ungated).

**Entry criteria**:
- [ ] Sortie 7 commit/log complete; previous manifest verified live
- [ ] `df_threshold = 28`

**Repo-specific**:
- Model ID: `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit`
- Tokenizer-artifact assertion: manifest contains `"tekken.json"` AND `"tokenizer.json"` AND `"tokenizer_config.json"` (this is the keystone verification that makes deletion of `ensureTekkenJson(at:)` in Sortie 18 safe — Mistral 4-bit is the smallest of the three Mistral quants and the first chance to confirm the assumption).

---

### Sortie 9: Ship `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-6bit` (19 GB)

**Priority**: 34.5 — eighth ship. Risk: low.

**Entry criteria**:
- [ ] Sortie 8 commit/log complete; previous manifest verified live
- [ ] `df_threshold = 40`

**Repo-specific**:
- Model ID: `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-6bit`
- Tokenizer-artifact assertion: manifest contains `"tekken.json"` AND `"tokenizer.json"`

---

### Sortie 10: Ship `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit` (25 GB)

**Priority**: 34 — ninth ship. Risk: medium (largest single Mistral, ~25 GB upload).

**Entry criteria**:
- [ ] Sortie 9 commit/log complete; previous manifest verified live
- [ ] `df_threshold = 50`

**Repo-specific**:
- Model ID: `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit`
- Tokenizer-artifact assertion: manifest contains `"tekken.json"` AND `"tokenizer.json"`

---

### Sortie 11: Ship `VincentGOURBIN/flux_qint_8bit` (32 GB) — subfolder ship

**Priority**: 33.5 — tenth ship; largest payload in the mission. Risk: medium (subfolder syntax, largest single upload). Foundation: this is the canonical qint8 transformer for production use.

**Entry criteria**:
- [ ] Sortie 10 commit/log complete; previous manifest verified live
- [ ] `df_threshold = 65`

**Repo-specific**:
- Model ID: `VincentGOURBIN/flux_qint_8bit`
- Ship invocation: `acervo ship VincentGOURBIN/flux_qint_8bit flux-2-dev/transformer/qint8/` (two-arg form; subfolder-only ship). The shipped manifest contains only the qint8 files; downstream `Acervo.ensureAvailable(...)` calls in Sortie 19 do not need a runtime filter. Record the exact command line in the ship log.
- Tokenizer-artifact assertion: n/a (transformer). Record "no-tokenizer-expected" + the subfolder decision (`acervo ship` two-arg form invoked) in the ship log.

---

### Sortie 12: Ship `black-forest-labs/FLUX.2-klein-9B` (18 GB) — gated

**Priority**: 33 — eleventh ship; the only gated repo. Risk: high (license-acceptance failure is the most likely failure mode in WU1; isolated as its own sortie for clean retry semantics on license-related failures). Foundation: completes the WU1 ship phase; required for `TransformerVariant.klein9B_bf16` runtime path in Sortie 19.

**Entry criteria**:
- [ ] Sortie 11 commit/log complete
- [ ] License verification log from Sortie 1 shows `black-forest-labs/FLUX.2-klein-9B` accepted (HTTP 200 from probe). If not, **stop**: operator must visit `https://huggingface.co/black-forest-labs/FLUX.2-klein-9B`, click "Agree and access repository", and re-run the Sortie 1 probe before this sortie can dispatch.
- [ ] `df_threshold = 40`

**Repo-specific**:
- Model ID: `black-forest-labs/FLUX.2-klein-9B`
- Tokenizer-artifact assertion: n/a (transformer). Record "no-tokenizer-expected; gated repo; license-accepted at Sortie 1" in the ship log.

---

### Sortie 13: CDN read-side smoke test from clean machine

**Priority**: 32 — high. WU1 gate: confirms the read path Acervo will use at runtime works against the freshly provisioned manifests. Foundation: validates the slugify mapping (`org/repo` → `org_repo`) the runtime depends on. May run in parallel with Sortie 15 (no shared dependency).

**Entry criteria**:
- [ ] Sortie 12 complete; 11 manifests live on R2

**Tasks**:
1. From a clean directory with no Flux2Swift state, write a minimal one-shot Swift script (`scripts/cdn-smoke-test.swift`, deleted at end of sortie) that imports `SwiftAcervo` and calls `Acervo.fetchManifest(for:)` for each of the 11 model IDs. Do not depend on `acervo` CLI read-side commands — call the Swift API directly so this exercises the same code path the runtime will use.
2. For each model: verify the returned manifest's file count and total byte count match the ship log from Sorties 2-12.
3. Verify Acervo's `slugify(_:)` produces the expected slug for each `org/repo` (e.g., `lmstudio-community/Mistral-Small-...` → `lmstudio-community_Mistral-Small-...`). Document any surprises.
4. Pick one small model (`lmstudio-community/Qwen3-4B-MLX-4bit`) and run `Acervo.ensureAvailable(_:files:progress:)` end-to-end. Confirm files land in `Acervo.modelDirectory(for:)` and SHA-256 manifests verify.
5. Tear down the test directory AND delete `scripts/cdn-smoke-test.swift` (the script is one-shot recon and must not be committed).

**Exit criteria**:
- [ ] All 11 `fetchManifest` calls return success (logged with model id + file count + total bytes)
- [ ] Manifest file counts match the ship log from Sorties 2-12 for all 11 models
- [ ] One end-to-end `ensureAvailable(_:files:progress:)` (small model: `lmstudio-community/Qwen3-4B-MLX-4bit`) succeeds and SHA-256 verification passes
- [ ] `scripts/cdn-smoke-test.swift` does not exist; no untracked files outside `Acervo.sharedModelsDirectory`
- [ ] Working tree commit message includes the literal string `WU1 complete; CDN provisioned`

---

## Work Unit 2 — HF Excision + Code Migration

> **Gate**: WU2 Sortie 16 does not begin until WU1 Sortie 13 reports success AND Sorties 14 + 15 are committed. Sorties 14 (library cleanup) and 15 (fixture capture) have no WU1 dependency and may run in parallel with any WU1 sortie; the supervisor serializes them as Sortie 14 → Sortie 15 because they share the SwiftPM build directory.

### Sortie 14: Library-only cleanup — remove CLI targets and customModelsDirectory

**Priority**: 26.5 — high foundation for WU2. Reduces the surface every subsequent sortie has to migrate (fewer call sites in S17/S19, fewer registry strings in S20, fewer tests in S21). Risk: low (mechanical deletion). Independent of WU1 (no CDN, no HF, no Acervo touch). Decided ahead of execution: this package is library-only; downstream end-to-end testing happens via `../SwiftVinetas` (sibling repo). Sandboxed apps cannot use a custom storage directory anyway, so `customModelsDirectory` is a misleading API surface that is deleted now rather than migrated.

**Entry criteria**:
- [ ] Working tree clean and on the mission base branch
- [ ] `Package.resolved` still pins `huggingface/swift-transformers` (cleanup operates on the OLD dependency graph; the swap happens in Sortie 16)
- [ ] No source edits made by Sortie 15 onwards yet

**Tasks**:

1. **Delete CLI targets and source**:
   - Remove the directories: `Sources/Flux2CLI/` and `Sources/FluxEncodersCLI/`.
   - In `Package.swift`: remove the `Flux2CLI` and `FluxEncodersCLI` `.executableTarget(...)` entries AND their corresponding `.executable(name:targets:)` product entries. The library products (`Flux2Core`, `FluxTextEncoders`) and their targets remain.
2. **Delete `customModelsDirectory` from runtime code** (do this AFTER task 1 so the deleted CLI doesn't dangle a reference):
   - `Sources/Flux2Core/Configuration/ModelRegistry.swift`: remove the `static var customModelsDirectory: URL?` declaration (around line 415) and the `if let custom = customModelsDirectory { ... }` branch in the directory accessor (around lines 433–435). The accessor returns the unconditional default for now; Sortie 19 will replace the default with `Acervo.modelDirectory(for:)`.
   - `Sources/FluxTextEncoders/Loading/TextEncoderModelDownloader.swift`: remove the `static var customModelsDirectory: URL?` (around line 20), simplify the body of `makeHubApi()` so it no longer consults `customModelsDirectory` (the function itself remains — full deletion of `makeHubApi()` + `reconfigureHubApi()` happens in Sortie 18 when Acervo replaces `hubApi`), and remove the `if let custom = customModelsDirectory { ... }` branches in the Mistral and Qwen3 directory accessors (around lines 40 and 51). After this sortie: `makeHubApi()` and `reconfigureHubApi()` still exist; `customModelsDirectory` does not.
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

### Sortie 15: Capture golden tokenizer fixtures from the old package

**Priority**: 24.75 — must run BEFORE Sortie 16 (Package.swift swap) because it depends on the old swift-transformers `Tokenizers` module being resolved. Parallelizable with WU1 Sortie 13 (no shared file or dependency). Foundation: produces the parity oracle used by Sortie 21.

**Entry criteria**:
- [ ] Sortie 14 commit exists (cleanup landed; build clean against old package)
- [ ] `Package.resolved` still pins `huggingface/swift-transformers`
- [ ] Working tree clean

**Tasks**:
1. Identify the project's tokenizer fixtures (search `Tests/FluxTextEncodersTests/` and `Tests/Fixtures/` for `tekken.json` / `tokenizer.json` references). Pick at minimum one Tekken tokenizer + one Qwen3 tokenizer that are exercised by current tests.
2. Define a deterministic prompt set (≥10 prompts: ASCII, multilingual, emojis, chat-template messages with role markers, one ≥1024-char document).
3. Add a one-shot fixture-generation test under `Tests/Fixtures/Generators/` that, against the OLD package, writes JSON files of `{prompt, encoded_token_ids[], decoded_text}` per (tokenizer × prompt).
4. Run via XcodeBuildMCP `swift_package_test` (never raw `swift test`). Commit the resulting fixtures under `Tests/Fixtures/TokenizerParity/`.
5. After fixtures are generated, delete the generator (it depends on old-package APIs that won't compile after Sortie 16).

**Exit criteria**:
- [ ] `Tests/Fixtures/TokenizerParity/*.json` exists with one file per (tokenizer × prompt)
- [ ] Generator removed; no remaining references to old-package-specific APIs in test code
- [ ] Project still builds against the old package (`swift_package_build` succeeds)

---

### Sortie 16: Package.swift swap

**Priority**: 22.75 — gate sortie that establishes the new dependency graph. Foundation: every WU2 sortie after this depends on the new module resolution. Risk: medium (the old `Tokenizers` module name is reused by swift-tokenizers, so call sites won't import-break — failures are silent until call-site rewrites in S17). Build is expected to fail at this sortie's exit (call sites unchanged).

**Entry criteria**:
- [ ] WU1 complete (all 11 manifests live, smoke test passed)
- [ ] Sortie 15 fixtures committed

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

### Sortie 17: Tokenizers API rename + applyChatTemplate audit

**Priority**: 19 — restores library-target build. Foundation: unblocks Sorties 18 and 19 (downloader rewrites). Risk: low (mechanical renames, single grep'd call sites). Behavior risk: `applyChatTemplate` default flip from `false` → `true` is a silent semantic change that this sortie locks down.

**Entry criteria**:
- [ ] Sortie 16 commit exists
- [ ] `docs/missions/recon-swift-tokenizers.md` reachable for signature reference

**Tasks**:
1. Rename `decode(tokens:)` → `decode(tokenIds:)` at the 8 source sites: `TekkenTokenizer.swift:398, 475`; `Qwen3Generator.swift:104, 134, 142, 230, 259, 267`.
2. Rewrite the positional-arg call at `FluxTextEncoders.swift:896` (`tokenizer.decode(tokens, skipSpecialTokens: skipSpecialTokens)`) → `tokenizer.decode(tokenIds: tokens, skipSpecialTokens: skipSpecialTokens)`. Same conversion at `TokenizerTests.swift:140`.
3. Rename `AutoTokenizer.from(modelFolder:)` → `AutoTokenizer.from(directory:)` at `FluxTextEncoders.swift:201` and `TekkenTokenizer.swift:232`. Verify no caller passes `hubApi:` or `strict:` parameters (both removed in swift-tokenizers).
4. **`applyChatTemplate` audit (preserve current behavior):** Add explicit `addGenerationPrompt: false` to every default-relying site to lock in pre-migration semantics:
   - `Sources/FluxTextEncoders/Tokenizer/TekkenTokenizer.swift:473, 510`
   - `Tests/FluxTextEncodersTests/TokenizerTests.swift:77, 93, 108`
   - `Tests/FluxTextEncodersTests/FluxTextEncodersTests.swift:204`
   The site at `Embeddings/EmbeddingExtractor.swift:193` already passes `false` — leave it.
5. Verify `Sendable` conformance on stored `Tokenizer` properties (`TekkenTokenizer.swift:42`, `KleinEmbeddingExtractor.swift:23`) compiles cleanly under Swift 6.2.
6. After edits: `swift_package_build` for `FluxTextEncoders` and `Flux2Core` library targets must succeed. The two downloader files remain broken (Sorties 18 and 19).

**Exit criteria**:
- [ ] `grep -rn 'decode(tokens:' Sources Tests` returns no matches
- [ ] No positional `decode` calls remain: `grep -rEn 'tokenizer\.decode\(' Sources Tests` returns matches and EVERY match is followed by `tokenIds:` (verifiable via `grep -rEn 'tokenizer\.decode\(' Sources Tests | grep -v 'tokenIds:'` returning no matches)
- [ ] `grep -rn 'from(modelFolder:' Sources Tests` returns no matches
- [ ] `grep -rn 'applyChatTemplate(messages:' Sources Tests | grep -v 'addGenerationPrompt'` returns no matches
- [ ] `swift_package_build` succeeds for `FluxTextEncoders` and `Flux2Core` library targets (test/CLI/App MAY still fail due to downloaders)
- [ ] No new `Sendable` warnings
- [ ] Commit message: `R2.1-R2.4, R3.1; tokenizers API rename`

---

### Sortie 18: Rewrite TextEncoderModelDownloader against Acervo

**Priority**: 18.25 — first runtime HF excision. Risk: high (replaces Hub-based download paths; adds `notProvisionedOnCDN` error case; deletes the `tekken.json` fallback). Complexity: high (one large file rewrite + error-case design). Foundation: establishes the Acervo integration pattern reused by Sortie 19.

**Entry criteria**:
- [ ] Sortie 17 commit exists
- [ ] WU1 manifests for all `lmstudio-community/Qwen3-*-MLX-*bit` (4 manifests; Sorties 2, 3, 4, 6) and `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-{4bit,6bit,8bit}` (3 manifests; Sorties 8, 9, 10) verified live on R2 — these are the seven repos this sortie's downloader migration depends on. **Note**: the upstream `mistralai/Mistral-Small-3.2-24B-Instruct-2506` repo is NOT shipped to CDN per the locked decision; the `tekken.json` it once provided as fallback ships inside every lmstudio-community MLX quant.

**Tasks**:
1. In `Sources/FluxTextEncoders/Loading/TextEncoderModelDownloader.swift`:
   - Replace `import Hub` with `import SwiftAcervo`.
   - Delete the legacy HF cache path resolution (lines 87-110, the `~/.cache/huggingface/hub/...` block).
   - Delete the `hubApi: HubApi` static, `makeHubApi()`, and `reconfigureHubApi()`. Storage location is `Acervo.sharedModelsDirectory` (Acervo manages this internally; no override hook is exposed — this matches the locked decision to remove `customModelsDirectory` entirely).
   - Replace the `download(model:progress:)` body that uses `hubApi.snapshot(from:matching:)` with `Acervo.ensureAvailable(modelInfo.repoId, files: <explicit file list>, progress: ...)`. The file list per model comes from the manifest written in WU1 — for Mistral MLX quants the list includes `tekken.json`, `tokenizer.json`, `tokenizer_config.json`, the safetensors shards, and config files (verified present in each MLX quant repo ahead of plan and re-confirmed by Sortie 8's manifest grep).
   - **Delete `ensureTekkenJson(at:progress:)` entirely** (lines 252-283 plus the call sites that invoke it at lines 210, 242, 264). This method is dead code: the hardcoded `huggingface.co/.../tekken.json` fetch was a defensive fallback for the case where downloaded model dirs lacked `tekken.json` — but every lmstudio-community MLX quant ships `tekken.json` directly (verified by HF API probe AND by Sortie 8/9/10 manifest greps). With explicit-file-list shipping via Acervo, `tekken.json` is always present.
   - Add handling for `ModelVariant.bf16` (text encoder): when called for the cut variant, throw `TextEncoderModelDownloaderError.notProvisionedOnCDN(variant: .bf16)` with a message naming the variant and stating it will be re-enabled in a follow-up CDN mission.
   - Replace `findModelPath(for:)` with `Acervo.modelDirectory(for: model.repoId)` + `Acervo.isModelAvailable(_:)` checks. Drop `verifyShardedModel(at:)` if Acervo's manifest verification covers it; otherwise keep and call after `ensureAvailable`.
   - Same migration for `downloadQwen3(_:progress:)` and `findQwen3ModelPath(for:)` (the parallel Qwen3 path starting around line 296).
2. Public API of `TextEncoderModelDownloader` should remain stable for the surfaces still present after Sortie 14: `download(_:progress:)`, `downloadQwen3(_:progress:)`, `findModelPath(for:)`, `findQwen3ModelPath(for:)`, `verifyShardedModel(at:)`, `modelsDirectory`. (`customModelsDirectory` is intentionally NOT in this list — it was deleted in Sortie 14.) If any caller-facing signature must change, document the break in the commit message.
3. `swift_package_build` for `FluxTextEncoders` (full target) must succeed.

**Exit criteria**:
- [ ] `grep -n 'import Hub' Sources/FluxTextEncoders/Loading/TextEncoderModelDownloader.swift` returns no matches
- [ ] `grep -n 'huggingface' Sources/FluxTextEncoders/Loading/TextEncoderModelDownloader.swift` returns no matches (case-insensitive)
- [ ] `grep -n 'hubApi\|HubApi\|snapshot(from:' Sources/FluxTextEncoders/Loading/TextEncoderModelDownloader.swift` returns no matches
- [ ] `swift_package_build` succeeds for `FluxTextEncoders` (full target including the consumers, if they compile after this sortie)
- [ ] Commit message: `R2.5; TextEncoderModelDownloader → SwiftAcervo`

---

### Sortie 19: Rewrite Flux2Core/ModelDownloader against Acervo

**Priority**: 15.5 — largest single rewrite in the mission (deletes the hand-rolled HF API client). Risk: high (touches transformer + VAE + klein download paths; introduces `notProvisionedOnCDN` for 4 cut variants; updates 4 enumerated callers post-cleanup). Complexity: high. Could be split into 19a (downloader rewrite + new error type) and 19b (caller updates) if execution shows context pressure — kept atomic because the new error type and caller updates must land together to keep the build green.

**Entry criteria**:
- [ ] Sortie 18 commit exists
- [ ] WU1 manifests for `aydin99/FLUX.2-klein-4B-int8`, `black-forest-labs/FLUX.2-klein-4B`, `VincentGOURBIN/flux_qint_8bit`, and `black-forest-labs/FLUX.2-klein-9B` verified live (Sorties 5, 7, 11, and 12)

**Tasks**:
1. In `Sources/Flux2Core/Loading/ModelDownloader.swift`:
   - **Delete** the entire hand-rolled HF API client: `fetchFileList(repoId:subfolder:)` (around line 419), the `download(filePath:repoId:...)` per-file download (around line 475), and any helper that hits `huggingface.co`.
   - Replace the `download(_:progress:)` method body with `Acervo.ensureAvailable(repoId, files: <explicit list>, progress: ...)`. For repos with `huggingFaceSubfolder` (e.g., `qint8` → `flux-2-dev/transformer/qint8/`), the file list resolution decided in WU1 Sortie 11 (subfolder-only ship) means no runtime filter is needed — Acervo's manifest already only contains the qint8 files.
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
4. Update the concrete `Flux2ModelDownloader` callers — re-grep at sortie start with `grep -rn 'Flux2ModelDownloader(' Sources/` to get the current list (Sortie 14 deleted `Sources/Flux2CLI/`, so the callers from there are gone). Expected post-cleanup callers:
   - `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift:141`
   - `Sources/Flux2Core/Training/LoRATrainingHelper.swift:470`
   - `Sources/Flux2App/ViewModels/ModelManager.swift:332, 374`
   Each must handle the new error gracefully — surfacing "not yet available" rather than crashing. Also audit `Tests/Flux2CoreTests/Flux2CoreTests.swift:406` (which references `TransformerVariant.bf16`) — adjust to skip download or assert the not-provisioned error.
5. Goal: no public API change to `Flux2ModelDownloader.download(_:progress:)` other than the new error case. Internal swap.
6. `swift_package_build` must succeed for ALL remaining targets in `Package.swift` (libraries + the `Flux2App` target + test targets; CLIs were deleted in Sortie 14).

**Exit criteria**:
- [ ] `grep -rn 'huggingface\.co' Sources/` returns no matches (case-insensitive) — every runtime HF URL is gone from `Sources/`
- [ ] `grep -rn 'huggingface\.co/api' Sources/` returns no matches
- [ ] `grep -rn 'fetchFileList' Sources/` returns no matches
- [ ] `swift_package_build` succeeds for ALL targets
- [ ] Commit message: `Flux2Core/ModelDownloader → SwiftAcervo; remove hand-rolled HF API client`

---

### Sortie 20: Strip HF runtime strings; add README acknowledgments

**Priority**: 8 — completes the runtime HF excision (removes display strings + accessor renames) and lands the credits surface. Risk: low (mechanical renames + README authoring). Verification: `grep -rn 'huggingface' Sources/` returns only origin/license attribution comments.

> **Note**: GUI About-panel work is explicitly out of scope for this mission per user direction (deferred to a separate workstream). This sortie's credits work is README-only.

**Entry criteria**:
- [ ] Sortie 19 commit exists; full project builds

**Tasks**:
1. In `Sources/Flux2Core/Configuration/ModelRegistry.swift` (verified: `huggingFaceRepo` at lines 34, 257, 321; `huggingFaceURL` at lines 117, 273, 334; no pre-existing `repoId` accessor on these enums — confirmed via `grep -n 'repoId' Sources/Flux2Core/Configuration/ModelRegistry.swift` returns no matches at sortie start):
   - **Remove** the three `huggingFaceURL` accessors (no runtime HF URLs).
   - **Rename** `huggingFaceRepo` → `repoId` and `huggingFaceSubfolder` → `repoSubfolder`. Because `repoId` does not currently exist on these enums, this is a clean rename, not a delete. Update all callers in `Sources/` (the rename will compile-fail at every old call site; fix each).
   - Doc comments may attribute model weight origin to HuggingFace (e.g., "Acervo model ID; weights originally from <hf-org>/<hf-repo>") — this is allowed under the credits policy.
   - Where the prior `huggingFaceURL` was consumed by UI code, remove the link or replace with an Acervo CDN manifest URL if SwiftAcervo exposes one.
2. Audit `Sources/FluxTextEncoders/Configuration/TextEncoderModelRegistry.swift` (verified: `repoId` accessor at lines 69, 155 and stored property at 198, 236; `huggingFaceURL` accessors at lines 84, 166, 208, 247). **Delete** the four `huggingFaceURL` accessors. The `repoId` accessor/field already exists and stays — no rename needed here.
3. **Update README.md with an `## Acknowledgments` section** containing:
   - Open-source dependency list with licenses: `mlx-swift` (MIT), `swift-argument-parser` (Apache-2.0), `swift-tokenizers` (Apache-2.0), `SwiftAcervo` (verify its license), `universal`/YAML (verify).
   - HuggingFace credit line: "Model weights for the variants distributed via this project's Acervo CDN were originally published on HuggingFace by Mistral AI, the Black Forest Labs team, the LM Studio community, and individual contributors (`VincentGOURBIN`, `aydin99`). Intrusive Memory mirrors these weights via Cloudflare R2 for distribution to Flux2Swift users."
   - Per-model attribution table mapping each Acervo-provisioned model to its HuggingFace origin and license.

**Exit criteria**:
- [ ] `grep -rn 'huggingface' Sources/` returns matches ONLY in code comments documenting model origin (no runtime URLs, no `huggingFaceURL` accessors)
- [ ] `grep -rn 'huggingFaceURL\|huggingFaceRepo\|huggingFaceSubfolder' Sources/` returns no matches (renamed)
- [ ] README.md contains the `## Acknowledgments` section with the dep list, the HuggingFace credit line, and the per-model attribution table
- [ ] `swift_package_build` succeeds for all targets
- [ ] Commit message: `Strip HF runtime strings; README acknowledgments`

---

### Sortie 21: Test sweep + tokenizer parity verification

**Priority**: 5 — proves the migration didn't regress tokenizer behavior. Risk: medium (hidden parity gaps surface here; weakening assertions to "pass tests" is forbidden). Foundation: gates the CI verification in Sortie 22.

**Entry criteria**:
- [ ] Sortie 20 commit exists; full project builds clean
- [ ] Sortie 15 golden fixtures present at `Tests/Fixtures/TokenizerParity/`

**Tasks**:
1. Update `Tests/FluxTextEncodersTests/TokenizerTests.swift` for the `decode` rename (if not already done in Sortie 17 task 2). Re-run; assert green.
2. Add `Tests/FluxTextEncodersTests/TokenizerParityTests.swift`:
   - Loads each Sortie 15 fixture JSON.
   - Asserts new-package `encode(text:)` output equals recorded `expected_token_ids`.
   - Asserts new-package `decode(tokenIds:)` output equals recorded `expected_decoded_text`.
   - Includes a printout of prompt + diff on any mismatch.
3. Run `swift_package_test` for `FluxTextEncodersTests`. All assertions must pass.
4. Run `swift_package_test` for `Flux2CoreTests` and `Flux2GPUTests`. Investigate any new failures, especially around offline/cache-only paths. Acervo's `isModelAvailable` semantics differ from the old `Hub` cache — fix tests to use Acervo APIs, do not weaken assertions.
5. Audit any test helper that calls out to `huggingface.co` or expects `HF_TOKEN`. Replace with Acervo-based fixtures or skip with a clear reason.
6. Mark `TokenizerParityTests.swift` with a comment indicating Sortie 22 will delete it after CI green.

**Exit criteria**:
- [ ] `swift_package_test` succeeds for `FluxTextEncodersTests`, `Flux2CoreTests`, `Flux2GPUTests`
- [ ] Parity test exists and passes
- [ ] No `huggingface.co` fetches in any test code path (`grep -rn 'huggingface' Tests/` returns only credits/comment references)
- [ ] No assertions weakened relative to pre-migration (verify via `git diff Tests/`)
- [ ] `TokenizerParityTests.swift` contains a `// Sortie 22 will delete this file after CI green` comment marker
- [ ] Commit message: `R4.1, R4.2, R4.3; test sweep + parity`

---

### Sortie 22: CI verification + parity test cleanup

**Priority**: 1.75 — terminal sortie. Risk: low (CI may surface OS-version or simulator drift, but local green is a strong predictor). Cleanup: removes the parity oracle so it does not rot in-tree.

**Entry criteria**:
- [ ] Sortie 21 commit exists; full local test suite green
- [ ] Mission branch pushed to remote

**Tasks**:
1. Open or update a draft PR for the mission branch.
2. **Verify CI does NOT have `HF_TOKEN` configured** (per the "huggingface-free" goal). If any GitHub Actions secret named `HF_TOKEN` exists in the repo, remove it and confirm the workflow still passes. CI's runtime requirement reduces to: macOS 26 runner, Swift 6.2+, internet access to `R2_PUBLIC_URL` (read-only, no auth).
3. Wait for the GitHub Actions matrix to complete via `gh pr checks --watch`.
4. If CI fails, triage and fix on the same branch. Do **not** mark this sortie complete with red CI.
5. Once CI is fully green, delete `TokenizerParityTests.swift` and `Tests/Fixtures/TokenizerParity/`. Re-run `swift_package_test` to confirm deletion does not break anything else. Push.
6. Verify final `Package.resolved` is committed cleanly (no untracked changes after a fresh `swift package resolve`).
7. Update the PR description with: (a) link to `recon-swift-tokenizers.md`, (b) link to `cdn-ship-log.md`, (c) link to `recon-cdn-inventory.md`, (d) summary of API changes, (e) note that the parity test was added and removed within this PR, (f) link to the new credits surface.

**Exit criteria**:
- [ ] `gh pr checks <PR>` shows all checks green
- [ ] No `HF_TOKEN` secret configured at the repo level (verify via `gh secret list`)
- [ ] `Tests/FluxTextEncodersTests/TokenizerParityTests.swift` does not exist
- [ ] `Tests/Fixtures/TokenizerParity/` does not exist
- [ ] `swift_package_test` still passes after parity test deletion
- [ ] PR description includes the mission summary (links to recon docs, ship log, summary of API changes, credits surface)
- [ ] No untracked changes after a fresh `swift package resolve`

---

## Parallelism Structure

**Critical path** (length 20 sorties on the longest chain): Sortie 1 → Sortie 2 → Sortie 3 → Sortie 4 → Sortie 5 → Sortie 6 → Sortie 7 → Sortie 8 → Sortie 9 → Sortie 10 → Sortie 11 → Sortie 12 → Sortie 13 → Sortie 16 → Sortie 17 → Sortie 18 → Sortie 19 → Sortie 20 → Sortie 21 → Sortie 22. Sortie 14 (cleanup) and Sortie 15 (fixtures) sit before Sortie 16 but neither depends on WU1, so they can begin as soon as the operator starts the mission — they do not extend wall-clock beyond WU1.

**Parallel execution groups**:

- **Group 1 — WU1 sequential (CDN provisioning, 13 sorties)**: Sortie 1 → Sortie 2 → Sortie 3 → Sortie 4 → Sortie 5 → Sortie 6 → Sortie 7 → Sortie 8 → Sortie 9 → Sortie 10 → Sortie 11 → Sortie 12 → Sortie 13. **No intra-group parallelism** (per locked decision: ships are strictly sequential — operator can only run one `acervo ship` at a time, and they share staging dir + HF download bandwidth). Operator-credentialed work; supervising agent only. Each ship is a single-goal binary pass/fail dispatch.
- **Group 2 — Pre-swap WU2 prep (overlapping WU1)**: Sortie 14 (library cleanup) and Sortie 15 (fixture capture) both have zero file overlap with WU1 sorties (different directories: WU1 writes to `docs/missions/` + the operator's R2 bucket; Sortie 14 deletes `Sources/Flux2CLI/` + tests; Sortie 15 writes to `Tests/Fixtures/`). Both can dispatch concurrently with any WU1 sortie. Order between Sortie 14 and Sortie 15 is independent functionally but they share the SwiftPM build directory, so the supervisor serializes them: **Sortie 14 → Sortie 15**, with the pair running alongside any WU1 sortie. **Supervising agent only** (both run `swift_package_build`/`swift_package_test`; sub-agents must not build).
- **Group 3 — WU2 sequential (code migration, 7 sorties)**: Sortie 16 → Sortie 17 → Sortie 18 → Sortie 19 → Sortie 20 → Sortie 21 → Sortie 22. Each sortie modifies code that the next depends on (Package.swift swap, then call-site rewrites, then downloaders, then registry strings, then tests, then CI). No intra-group parallelism. **Hard gate**: Sortie 16 cannot start until WU1 Sortie 13 reports success AND Sorties 14 + 15 are committed.

**Agent constraints**:
- **Supervising agent**: handles every sortie in this mission. All WU2 sorties run `swift_package_build` or `swift_package_test`. WU1 sorties run operator-credentialed CLI tools (`acervo ship`, `hf`, `curl`, R2 access) that should not be delegated.
- **Sub-agents (up to 4)**: not used. The mission shape is heavily sequential and build-bound. Allocating sub-agents here would add coordination overhead without wall-clock benefit.

**Maximum parallelism**: 2 simultaneous sorties at any moment (one WU1 ship/verify in flight + one WU2 prep sortie from {Sortie 14, Sortie 15}).

**Missed opportunities**: none material. The dependency graph is linear by construction:
- CDN must be provisioned before any code change can be tested against the new path (WU1 → WU2 Sortie 16 layer gate).
- Within WU1, ships are strictly sequential per locked decision (no concurrent `acervo ship` invocations).
- Within WU2 post-swap, each sortie's exit state (Package.swift swap, then call-site rewrites, then downloader rewrites, then registry strings) is the next sortie's entry precondition.
- Splitting Sortie 19 into 19a (downloader rewrite) + 19b (caller updates) was considered and rejected: the new error case must land with its callers to keep the build green.

---

## Open Questions & Missing Documentation

These items have documented fallbacks and do NOT block execution start. Each is resolved inside the sortie that needs it; if the fallback fires, the decision is captured in the sortie's commit message and recon docs.

| Sortie | Issue Type | Description | Resolution Plan |
|---|---|---|---|
| WU2 Sortie 18 (task 1) | Open question | Does Acervo's manifest verification cover what the legacy `verifyShardedModel(at:)` checks (per-shard SHA + `.safetensors.index.json` presence + total byte count)? | Read `SwiftAcervo` 0.8.3+ source. If covered, drop `verifyShardedModel(at:)`. If partial, keep it and call it after `Acervo.ensureAvailable`. Decision goes in the sortie's commit message. |
| WU2 Sortie 20 (task 3) | Vague criterion | "verify its license" for `SwiftAcervo`, `universal`/YAML | Auto-resolved at sortie start: `cat .build/checkouts/SwiftAcervo/LICENSE` (and equivalent for the YAML dep). Record the license name in the README acknowledgments table. Not a research task — mechanical lookup. |

**Vague criteria scan**: clean — every exit criterion in the plan is grep- or build-verifiable.

**Missing documentation**: none. `docs/missions/recon-swift-tokenizers.md` (existing) covers the swift-tokenizers API surface. `docs/missions/recon-cdn-inventory.md` is created by Sortie 1 (per its task list) and consumed by all later sorties — its absence at start is expected.

**Requires manual review before execution**: 0 issues. Plan is unblocked.

---

## Requirements Coverage (verified against `docs/missions/tokenizer-migration.md`)

| Requirement | Source Description | Covered By |
|---|---|---|
| **R1.1** | Replace `swift-transformers` dep with new deps | Sortie 16. **Modified per locked decision**: `swift-tokenizers` + `SwiftAcervo` (NOT `swift-hf-api`, which is rejected in favor of Acervo). |
| **R1.2** | Update target deps in FluxTextEncoders + Flux2Core | Sortie 16 tasks 2-3 |
| **R1.3** | Delete `Package.resolved`, re-resolve, verify no swift-transformers transitives | Sortie 16 tasks 4-6 |
| **R1.4** | Backend trait: default Swift, only opt into Rust if benchmarked | Locked decision: default Swift. Implemented in Sortie 16 (no opt-in). |
| **R2.1** | `FluxTextEncoders.swift:14` import unchanged | Sortie 17 task 2 (positional `decode` rewrite at line 896) |
| **R2.2** | TekkenTokenizer.swift `decode`, `from(modelFolder:)` renames | Sortie 17 tasks 1, 3 |
| **R2.3** | KleinEmbeddingExtractor.swift audit | Sortie 17 task 5 (Sendable verification at line 23); applyChatTemplate in EmbeddingExtractor.swift:193 already explicit `false` per recon |
| **R2.4** | Qwen3Generator.swift audit | Sortie 17 task 1 (6 sites at lines 104, 134, 142, 230, 259, 267) |
| **R2.5** | TextEncoderModelDownloader import + API | Sortie 18. **Modified per locked decision**: replace with SwiftAcervo, not swift-hf-api. Adds `notProvisionedOnCDN` for `ModelVariant.bf16`. Deletes `ensureTekkenJson(at:)`. |
| **R2.6** | Flux2Core/ModelDownloader.swift audit for Hub usage | Sortie 19. **Expanded per scope**: full hand-rolled HF API client deletion, not just `import` changes. |
| **R3.1** | Tokenizer Sendable conformance compile-clean | Sortie 17 task 5 |
| **R4.1** | TokenizerTests.swift `decode` rename | Sortie 17 task 2 (line 140) + Sortie 21 task 1 (re-verify) |
| **R4.2** | TextEncoderModelDirectoryTests + FluxTextEncodersTests + Flux2CoreTests rerun | TextEncoderModelDirectoryTests + ModelDirectoryTests **deleted** in Sortie 14 (per locked decision: `customModelsDirectory` removed entirely; sandboxed apps can't use a custom storage directory). Remaining tests rerun in Sortie 21. |
| **R4.3** | Tokenizer parity test (added + dropped within mission) | Sortie 15 captures fixtures from OLD package, Sortie 21 asserts equality, Sortie 22 deletes. |
| **R5.1** | No data file changes | Out of scope; verified by inspection — `tokenizer.json`, `tekken.json`, `tokenizer_config.json`, safetensors layouts unchanged. |
| **R6.1** | XcodeBuildMCP `swift_package_build`/`_test` locally; GitHub Actions green | Throughout WU2 (no `swift build` / `swift test`); CI gate at Sortie 22. |
| **R6.2** | Minimum platforms compatibility | Verified in Sortie 16 build (SwiftAcervo 0.8.3+ supports `macOS(.v26)` / `iOS(.v26)`; failure here aborts the sortie). |
| **R7** | Risks: `swift-hf-api` parity, `decode` rename, trait selection, dirty ModelDownloader | Pre-flight Risk Notes addresses each. `swift-hf-api` parity moot per locked decision (not introduced). `decode` rename addressed in Sortie 17 with grep verification in exit criteria. Trait selection: default Swift (locked). Dirty ModelDownloader: pre-flight note instructs not to start until marching-relay merged or rebased. |

**Additional scope coverage** (beyond the original tokenizer-migration.md):

| Scope item | Source | Covered By |
|---|---|---|
| Hand-rolled HF API client deletion | Plan iteration scope expansion | Sortie 19 |
| HF runtime URL elimination (`huggingface.co` strings, `huggingFaceURL` accessors) | Plan iteration scope expansion | Sortie 19 (URL grep) + Sortie 20 (accessor renames) |
| Acervo CDN provisioning (11 models, ~137 GB) | Plan iteration scope expansion | WU1 (Sorties 1–13) |
| `notProvisionedOnCDN` error case for 5 cut variants | Locked decision | Sortie 18 (text encoder bf16) + Sortie 19 (4 transformer variants) |
| README acknowledgments | Locked decision (About panel deferred) | Sortie 20 task 3 |
| Library-only package shape (CLI deletion) | Locked decision | Sortie 14 |
| `customModelsDirectory` removal | Locked decision | Sortie 14 |

**Coverage verdict**: ✓ All R1.1–R7 requirements covered, with two requirements modified per locked decisions (R1.1 swaps `swift-hf-api` for `SwiftAcervo`; R4.2 drops two test files because their underlying API is deleted). All scope expansions covered.

---

## Summary

| Metric | Value |
|---|---|
| Work units | 2 |
| Total sorties | 22 (13 in WU1, 9 in WU2) |
| Dependency structure | Layered: WU1 (CDN provisioning) → WU2 post-swap (Sortie 16+). WU2 pre-swap sorties (Sortie 14 cleanup, Sortie 15 fixtures) are independent of WU1 and can run alongside it. |
| Critical path length | 20 sorties (Sortie 1 → Sortie 13 sequential, then Sortie 16 → Sortie 22 sequential; Sorties 14+15 overlap WU1 and don't extend wall-clock) |
| Maximum simultaneous sorties | 2 (one WU1 sortie + one of {Sortie 14, Sortie 15}) |
| Models to mirror to CDN | 11 (~137 GB; one ship per sortie, Sorties 2–12) |
| Variants kept in registry but not provisioned | 5 (1 text-encoder BF16 + 4 transformer variants) |
| Gated repos requiring license click-through | 1 (`black-forest-labs/FLUX.2-klein-9B`; isolated as Sortie 12) |
| Source requirements doc | `docs/missions/tokenizer-migration.md` (original; expanded by user direction during plan iteration) |
| Requirements covered | R1.1–R1.4, R2.1–R2.6, R3.1, R4.1–R4.3, R5.1, R6.1–R6.2, R7 + full HF excision |
| Open questions blocking execution | 0 (2 in-sortie questions with documented fallbacks; see above) |

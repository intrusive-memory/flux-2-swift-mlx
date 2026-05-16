# TODO

## 2026-05-15 — Telemetry coverage gaps (from SwiftVinetas `--telemetry` diagnosis)

Reproducer: `vinetas generate "<prompt>" --model klein4b --seed 42 --telemetry`. Trace files cited below live in `~/Library/Caches/vinetas/telemetry/`. Compared head-to-head against PixArt on the same prompt/seed.

A full Flux2 Klein-4B run currently emits only **13 events** end-to-end (vs PixArt's 174 for an equivalent run via SwiftTuberia). The events that *are* present are well-formed; the gaps below are missing instrumentation, not broken instrumentation.

### Status — 2026-05-16

Items 1 and 2 are done in this repo on `development`. New event cases live in
`Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` and the emit sites are
in `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift`. The Vinetas-side
`Flux2TelemetryAdapter` needs new sink-phase strings added before any of these
events will surface in the JSONL trace. See "Vinetas-side follow-up" below.

### Done

1. ✅ **Per-step `denoiseStep*` events.** Added `denoiseStepStart(variant, stepIndex, totalSteps, t, latentShape, latentDtype)` and `denoiseStepComplete(variant, stepIndex, totalSteps, t, latentStat, durationSeconds)` to all four denoise paths (textToImage, imageToImageKVExtractStep0, imageToImageKVCached, imageToImageFullRecompute). `latentStat` reuses `TuberiaTensorStat.sample` and is gated by an `if let telemetry = currentTelemetry()` so a `nil` reporter still pays zero MLX-reduction cost. The KV-extract step reuses the loop-end stat to avoid a second reduction on the same latent.
2. ✅ **`weightLoadComplete` for the transformer + `quantizationComplete`.** `loadTransformer` now captures duration + `paramCount` (sum of `weights.values.size` before `removeAll`) and emits `weightLoadComplete(.transformer, …)`. A new `quantizationComplete(component, bits, groupSize, durationSeconds)` event fires around the on-the-fly quant block, so 4-bit/8-bit quant time is attributable.

### Skipped

3. ⏸ **Per-step transformer forward stats (`transformerForwardStart/Complete`).** Skipped as redundant: `denoiseStepComplete` already carries the post-step latent stat and `durationSeconds`, and the transformer forward dominates step time (scheduler.step is effectively free). Adding a second pair per step would double emissions without surfacing new information. Revisit if cross-engine apples-to-apples symmetry with PixArt's `backboneForward*` becomes load-bearing for an analysis tool.

### Vinetas-side follow-up

`Flux2TelemetryAdapter` in SwiftVinetas needs new exhaustive-switch arms for:
- `denoiseStepStart` → sink phase `flux_denoise_step_start`
- `denoiseStepComplete` → sink phase `flux_denoise_step_complete` (memory snapshot? Probably not — fires N×28 per generation.)
- `quantizationComplete` → sink phase `flux_quantization_complete_<component>` (memory snapshot, mirrors `weightLoadComplete`)

Until that adapter is updated, the new events emit from the pipeline but get dropped at the adapter boundary.

### Out of scope here

Per-step latent stat *schema* (mean/std/min/max/hasNaN/hasInf) matches what `vaeDecodeComplete.pixelStat` already uses in this library — no new struct needed, just instantiate it per step.

---
feature_name: OPERATION UNIVERSAL SOLVENT
starting_point_commit: c921c9f475d41662a0087764cc7ec8353cd462f0
mission_branch: mission/universal-solvent/01
iteration: 1
---

# EXECUTION_PLAN.md — Flux2Swift: Replace Yams with Universal

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase).

---

## Source Requirements

- **REQUIREMENTS.md**: Remove the Yams dependency and replace with [universal](https://github.com/marcprux/universal)

## Codebase Analysis

**Current Yams footprint:**
- `Package.swift:21` — Yams package declaration (`"https://github.com/jpsim/Yams"`, from `"5.1.0"`)
- `Package.swift:62` — Flux2CLI target depends on `.product(name: "Yams", package: "Yams")`
- `Sources/Flux2CLI/TrainingConfigYAML.swift:5` — `import Yams`
- `Sources/Flux2CLI/TrainingConfigYAML.swift:299-300` — `YAMLDecoder()` and `.decode(TrainingConfigYAML.self, from: yamlString)`
- `Sources/Flux2CLI/TrainLoRACommand.swift:9` — `import Yams` (unused directly; only calls `YAMLConfigParser`)

**Replacement library (`universal` v5.3.0):**
- Package URL: `https://github.com/marcprux/universal`
- Latest version: `5.3.0`
- Product name: `"YAML"` from package `"universal"`
- YAML module provides `YAML.parse(yaml: String)` → `YAML` value
- `YAML` conforms to `Encodable` (NOT `Decodable`)
- Decode pattern: `YAML.parse(yaml:)` → `JSONEncoder().encode(yamlValue)` → `JSONDecoder().decode(T.self, from: jsonData)`

---

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|-------------|
| Flux2CLI-Yams-Replacement | Sources/Flux2CLI | 2 | 1 | none |

---

## Flux2CLI-Yams-Replacement

### Sortie 1: Update Package.swift Dependencies

**Priority**: 7.25 — Foundation sortie; blocks all subsequent work. Must resolve before source changes.

**Entry criteria**:
- [ ] First sortie — no prerequisites

**Tasks**:
1. In `Package.swift`, replace the Yams package dependency (line 21) `.package(url: "https://github.com/jpsim/Yams", from: "5.1.0")` with `.package(url: "https://github.com/marcprux/universal", from: "5.3.0")`.
2. In `Package.swift`, update the Flux2CLI target dependency (line 62) from `.product(name: "Yams", package: "Yams")` to `.product(name: "YAML", package: "universal")`.
3. Run `swift package resolve` to verify the dependency graph resolves successfully.

**Exit criteria**:
- [ ] `grep -c "Yams" Package.swift` returns `0`
- [ ] `grep -c 'marcprux/universal' Package.swift` returns `1`
- [ ] `grep -c '"YAML"' Package.swift` returns `1` (Flux2CLI target dependency)
- [ ] `swift package resolve` exits with code 0

---

### Sortie 2: Replace Yams API Usage and Verify Build

**Priority**: 2.5 — Final sortie; no downstream dependencies. Moderate risk (new API pattern).

**Entry criteria**:
- [ ] Sortie 1 exit criteria met (`Package.swift` updated, `swift package resolve` succeeds)

**Tasks**:
1. In `Sources/Flux2CLI/TrainingConfigYAML.swift`, replace `import Yams` (line 5) with `import YAML`.
2. In `Sources/Flux2CLI/TrainingConfigYAML.swift`, replace the `YAMLDecoder`-based parsing block (lines 298-301):
   ```swift
   // BEFORE (Yams):
   let decoder = YAMLDecoder()
   let config = try decoder.decode(TrainingConfigYAML.self, from: yamlString)
   return config

   // AFTER (universal):
   let yamlValue = try YAML.parse(yaml: yamlString)
   let jsonData = try JSONEncoder().encode(yamlValue)
   let config = try JSONDecoder().decode(TrainingConfigYAML.self, from: jsonData)
   return config
   ```
3. In `Sources/Flux2CLI/TrainLoRACommand.swift`, remove `import Yams` (line 9). This file does not use any YAML types directly.
4. Build the project: `xcodebuild build -scheme Flux2Swift-Package -destination 'platform=macOS'`.
5. Run existing tests: `xcodebuild test -scheme Flux2Swift-Package -destination 'platform=macOS'`.

**Exit criteria**:
- [ ] `grep -r 'import Yams' Sources/` returns no matches (exit code 1)
- [ ] `grep -r 'Yams' Package.swift Sources/` returns no matches (exit code 1)
- [ ] `grep -c 'YAML.parse' Sources/Flux2CLI/TrainingConfigYAML.swift` returns `1`
- [ ] `grep -c 'JSONDecoder().decode' Sources/Flux2CLI/TrainingConfigYAML.swift` returns at least `1`
- [ ] `xcodebuild build -scheme Flux2Swift-Package -destination 'platform=macOS'` exits with code 0
- [ ] `xcodebuild test -scheme Flux2Swift-Package -destination 'platform=macOS'` exits with code 0

---

## Parallelism Structure

**Critical Path**: Sortie 1 → Sortie 2 (length: 2 sorties)

**Parallel Execution Groups**: None — single work unit with 2 sequential sorties.

**Agent Constraints**:
- **Supervising agent**: Handles both sorties (both have build/resolve steps)
- **Sub-agents**: None needed (mission is too small to benefit from parallelism)

---

## Open Questions & Missing Documentation

No blocking issues identified.

| Sortie | Issue Type | Status | Description |
|--------|-----------|--------|-------------|
| Sortie 2 | External API | Resolved | `universal` YAML module's `YAML` type conforms to `Encodable` — confirmed via source review. The YAML→JSON→Decodable pattern is viable. |
| Sortie 1 | Version | Resolved | Latest `universal` version is `5.3.0` — confirmed via `gh api repos/marcprux/universal/tags`. |

---

## Summary

| Metric | Value |
|--------|-------|
| Work units | 1 |
| Total sorties | 2 |
| Dependency structure | sequential |
| Critical path length | 2 sorties |
| Parallelism | 1 supervising agent, 0 sub-agents |
| Average sortie size | ~17 turns (budget: 50) |

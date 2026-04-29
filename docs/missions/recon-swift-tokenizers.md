# swift-tokenizers Reference (DePasqualeOrg/swift-tokenizers)

## Latest Version

- **Latest released tag**: `0.4.2`
- **Release date**: 2026-04-24 (22:29 UTC)
- **URL**: https://github.com/DePasqualeOrg/swift-tokenizers/releases/tag/0.4.2
- **Recommended pin for Flux2Swift**: `from: "0.4.2"` — bump from `0.4.2` baseline. The release notes say "Fix Package.swift and update readme" only (PR #25), so the API surface is identical to `0.4.1`. There is no `0.4.x` major-API churn risk; pinning `from: "0.4.2"` will accept `0.4.3+` patch fixes (typical SemVer minor under 1.0 with this maintainer is conservative — see `0.4.0 → 0.4.1` was packaging-only). If the migration is risk-averse, use `.upToNextMinor(from: "0.4.2")` to refuse a `0.5.x` bump until reviewed.

> Note: the requirements doc that referenced `0.4.2` was not stale — `0.4.2` IS the latest as of 2026-04-27. There is no `0.5.x` available. The associated Rust XCFramework artifact is published separately at tag `tokenizers-rust-0.4.1` (the `0.4.2` Swift release reuses the `0.4.1` Rust binary; see `Package.swift:11`).

---

## Platform / Toolchain Floor

Source: [`Package.swift`](https://github.com/DePasqualeOrg/swift-tokenizers/blob/0.4.2/Package.swift)

- `// swift-tools-version: 6.1` (Package.swift:1)
- Platforms: `[.iOS(.v17), .macOS(.v14)]` (Package.swift:192)
- No tvOS / watchOS / visionOS declared — these will fall back to whatever Swift 6.1 / SwiftPM allows by default.
- Swift language mode: implied Swift 6 (tools-version 6.1). Strict concurrency: the `Tokenizer` protocol is declared `Sendable` (Tokenizer.swift:269) and `PreTrainedTokenizer` is `@unchecked Sendable, Tokenizer` (Tokenizer.swift:348). `TokenizerExecutionBackend` is `Sendable` (Tokenizer.swift:244). Type aliases `Message = [String: any Sendable]` and `ToolSpec = [String: any Sendable]` (Tokenizer.swift:29, 32). Vocabulary types use `@unchecked Sendable` because the underlying `NSDictionary`/`NSArray` is treated as immutable post-construction (Tokenizer.swift:9, 19).

**Compatibility with Flux2Swift's floor (Swift 6.2, macOS 26 / iOS 26)**: clean — Flux2Swift's floor is *higher* than swift-tokenizers' floor on every axis. Flux2Swift compiles under tools-version 6.2 against `Tokenizer`'s `Sendable` requirement without issue.

---

## Public API Surface (Tokenizers module)

> All citations are at tag `0.4.2`. The umbrella product is `Tokenizers`, defined in `Package.swift:194` and assembled by `Sources/TokenizersFacade/Exports.swift` which `@_exported import`s `TokenizersCore` (and `TokenizersSwiftBackend` when the `Swift` trait is on). The Rust backend's symbols are intentionally NOT re-exported — only `AutoTokenizer.from(directory:)` is the public entry.

### `protocol Tokenizer` (Tokenizer.swift:269–286)

```swift
public protocol Tokenizer: Sendable {
    func tokenize(text: String) -> [String]
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertIdToToken(_ id: Int) -> String?
    var bosToken: String? { get }
    var eosToken: String? { get }
    var unknownToken: String? { get }
    var hasChatTemplate: Bool { get }
    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int]
}
```

Default implementations on the protocol extension (Tokenizer.swift:288–323):

```swift
extension Tokenizer {
    public var hasChatTemplate: Bool { false }                                      // line 288

    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument? = nil,
        addGenerationPrompt: Bool = true,
        truncation: Bool = false,
        maxLength: Int? = nil,
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [Int]                                                                // line 290

    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: String,
        addGenerationPrompt: Bool = true,
        truncation: Bool = false,
        maxLength: Int? = nil,
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [Int]                                                                // line 307
}
```

Public extension on `Tokenizer` (Tokenizer.swift:325–343):

```swift
public extension Tokenizer {
    func encode(text: String) -> [Int]                                               // line 325
    func callAsFunction(_ text: String, addSpecialTokens: Bool = true) -> [Int]      // line 329
    func decode(tokenIds: [Int]) -> String                                           // line 332
    func convertTokensToIds(_ tokens: [String]) -> [Int?]                            // line 335
    func convertIdsToTokens(_ ids: [Int]) -> [String?]                               // line 338
    var bosTokenId: Int? { bosToken.flatMap { convertTokenToId($0) } }               // line 341
    var eosTokenId: Int? { eosToken.flatMap { convertTokenToId($0) } }               // line 342
    var unknownTokenId: Int? { unknownToken.flatMap { convertTokenToId($0) } }       // line 343
}
```

### `enum AutoTokenizer` (Tokenizer.swift:459 + Sources/TokenizersFacade/AutoTokenizerDirectoryLoader.swift)

```swift
public enum AutoTokenizer {}                                                         // Tokenizer.swift:459

public extension AutoTokenizer {
    static func from(directory: URL) async throws -> Tokenizer                       // AutoTokenizerDirectoryLoader.swift:25
}
```

**That is the ONLY public factory.** There is NO `from(pretrained:)`, NO `from(modelFolder:)`, NO `from(tokenizerConfig:tokenizerData:)`, NO `HubApi`-based loader. The package is deliberately decoupled from `huggingface/swift-transformers`'s `Hub` dependency — the README explicitly says: "without Hugging Face Hub dependencies."

### `decode` overloads

- Required: `func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String` — Tokenizer.swift:271
- Convenience: `func decode(tokenIds: [Int]) -> String` (defaults `skipSpecialTokens: false`) — Tokenizer.swift:332
- **There is NO positional `decode(_ tokens: [Int], skipSpecialTokens:)` overload.** All decode entry points use the label `tokenIds:`.

### `encode` overloads

- Required: `func encode(text: String, addSpecialTokens: Bool) -> [Int]` — Tokenizer.swift:270
- Convenience: `func encode(text: String) -> [Int]` (defaults `addSpecialTokens: true`) — Tokenizer.swift:325
- Convenience: `func callAsFunction(_ text: String, addSpecialTokens: Bool = true) -> [Int]` — Tokenizer.swift:329

### `applyChatTemplate` family

The protocol requirement is the fully-explicit 7-arg form (Tokenizer.swift:278–285). Two default overloads provide convenience:

1. Same shape as protocol but with default values for every arg after `messages:` (Tokenizer.swift:290).
2. `chatTemplate: String` shortcut that wraps it in `.literal(...)` (Tokenizer.swift:307).

The shorter swift-transformers 1.x convenience overloads (`messages:`, `messages:tools:`, `messages:tools:additionalContext:`, `messages:chatTemplate:ChatTemplateArgument`) **do not exist** in 0.4.2 — call sites that used those will need to either use the long form with default args inferred or pass `nil`s explicitly.

### Special-token properties

Returned as **properties** (computed in a `public extension Tokenizer`) — same shape as swift-transformers 1.x:

- `var bosToken: String? { get }` — protocol requirement, Tokenizer.swift:274
- `var eosToken: String? { get }` — protocol requirement, Tokenizer.swift:275
- `var unknownToken: String? { get }` — protocol requirement, Tokenizer.swift:276
- `var bosTokenId: Int? { get }` — extension default, Tokenizer.swift:341
- `var eosTokenId: Int? { get }` — extension default, Tokenizer.swift:342
- `var unknownTokenId: Int? { get }` — extension default, Tokenizer.swift:343

**`padToken` / `padTokenId` are NOT on the `Tokenizer` protocol.** `padToken: String?` exists only on `TokenizerRuntimeConfiguration` (TokenizerRuntimeConfiguration.swift:49), which is `package`-visible — not public. There is no `padTokenId` anywhere on the public surface (matches swift-transformers 1.x — `padTokenId` was on `GenerationConfig`, not `Tokenizer`).

### `protocol TokenizingModel` (Tokenizer.swift:175–211)

Public lower-level protocol that backends conform to. Includes `tokenize(text:) -> [String]`, `convertTokenToId(_:) -> Int?`, `convertIdToToken(_:) -> Int?`, plus `bosToken/bosTokenId`/`eosToken/eosTokenId`/`unknownToken/unknownTokenId`/`fuseUnknownTokens: Bool`. Most callers should target `Tokenizer`, not `TokenizingModel`.

### `class PreTrainedTokenizer` (Tokenizer.swift:348–457)

Concrete `Tokenizer` implementation. Constructor (`init(model:runtimeConfiguration:backend:)`) is `package`-visible — callers cannot construct directly; they must go through `AutoTokenizer.from(directory:)`. All public methods on the class match the protocol surface above. The class is `@unchecked Sendable`.

### Other types worth knowing

- `public typealias Message = [String: any Sendable]` (Tokenizer.swift:29)
- `public typealias ToolSpec = [String: any Sendable]` (Tokenizer.swift:32)
- `public enum ChatTemplateArgument { case literal(String); case name(String) }` (Tokenizer.swift:256–264)
- `public enum TokenizerError: LocalizedError, Equatable` (Tokenizer.swift:35–74) — case names changed: `unsupportedModelType(String)` (was `unsupportedTokenizer`); added `missingUnknownToken`, `unsupportedComponent`, `missingConfigField`. Removed: `missingTokenizerClassInConfig`.

---

## Migration Diff: swift-transformers 1.x → swift-tokenizers

| swift-transformers 1.3.0 signature | swift-tokenizers 0.4.2 signature | Verdict |
|---|---|---|
| `tokenizer.decode(tokens: [Int]) -> String` | `tokenizer.decode(tokenIds: [Int]) -> String` | **renamed** (parameter label `tokens:` → `tokenIds:`) |
| `tokenizer.decode(tokens: [Int], skipSpecialTokens: Bool) -> String` | `tokenizer.decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String` | **renamed** (parameter label `tokens:` → `tokenIds:`) |
| `tokenizer.decode(_ tokens: [Int], skipSpecialTokens: Bool)` (positional) | — | **never existed** in either package; if Flux2Swift currently calls this it's a typo or shim |
| `AutoTokenizer.from(modelFolder: URL, hubApi: HubApi = .shared, strict: Bool = true) async throws -> Tokenizer` | `AutoTokenizer.from(directory: URL) async throws -> Tokenizer` | **renamed + signature-changed** (label `modelFolder:` → `directory:`, `hubApi:` and `strict:` removed) |
| `AutoTokenizer.from(pretrained: String, hubApi: HubApi = .shared, strict: Bool = true) async throws -> Tokenizer` | — | **REMOVED**. Caller must download files (e.g. via separate `swift-hf-api`) to a local directory first, then call `from(directory:)`. README explicitly documents this migration. |
| `AutoTokenizer.from(tokenizerConfig: Config, tokenizerData: Config, strict: Bool = true) throws -> Tokenizer` | — | **REMOVED**. There is no public `Config`-based factory. |
| `tokenizer.encode(text: String) -> [Int]` | identical | **identical** |
| `tokenizer.encode(text: String, addSpecialTokens: Bool) -> [Int]` | identical | **identical** |
| `tokenizer.tokenize(text: String) -> [String]` | identical | **identical** |
| `tokenizer.callAsFunction(_:addSpecialTokens:)` | identical | **identical** |
| `tokenizer.applyChatTemplate(messages:) throws -> [Int]` | — (use long form with defaults) | **signature-changed** (convenience overload removed; long form has default args, so `tokenizer.applyChatTemplate(messages: msgs)` still compiles) |
| `tokenizer.applyChatTemplate(messages:tools:) throws -> [Int]` | — | **removed** (use long form: `applyChatTemplate(messages: msgs, tools: t)`) |
| `tokenizer.applyChatTemplate(messages:tools:additionalContext:) throws -> [Int]` | — | **removed** (use long form) |
| `tokenizer.applyChatTemplate(messages:chatTemplate: ChatTemplateArgument) throws -> [Int]` | — | **removed** (use long form) |
| `tokenizer.applyChatTemplate(messages:chatTemplate: String) throws -> [Int]` | identical (Tokenizer.swift:307) | **identical** |
| `tokenizer.applyChatTemplate(messages:chatTemplate:addGenerationPrompt:truncation:maxLength:tools:) throws -> [Int]` | replaced by 7-arg form (with `additionalContext`) | **signature-changed** (extra `additionalContext:` arg required, but has default `nil` in extension overload — so existing call sites compile if they used keyword args) |
| `tokenizer.applyChatTemplate(messages:chatTemplate:addGenerationPrompt:truncation:maxLength:tools:additionalContext:) throws -> [Int]` | identical shape | **signature-changed**: `addGenerationPrompt` default flipped from `false` → `true` in 0.4.2 (Tokenizer.swift:292). **This is a behavior change** — call sites that relied on the default will start emitting a generation prompt. |
| `tokenizer.bosToken: String?` / `eosToken` / `unknownToken` | identical | **identical** |
| `tokenizer.bosTokenId: Int?` / `eosTokenId` / `unknownTokenId` | identical (still optional `Int?`, still computed via `convertTokenToId`) | **identical** |
| `tokenizer.padTokenId: Int?` | — | **never existed on `Tokenizer` protocol in either package**; lives on `GenerationConfig` in swift-transformers, not exposed in swift-tokenizers at all |
| `tokenizer.hasChatTemplate: Bool` | identical | **identical** |
| `tokenizer.convertTokenToId(_:)` / `convertIdToToken(_:)` | identical | **identical** |
| `tokenizer.convertTokensToIds(_:)` / `convertIdsToTokens(_:)` | identical | **identical** |
| `import Hub` (HubApi, HubApi.shared, snapshot APIs) | — | **removed** — swift-tokenizers has no `Hub` module. Replace with `swift-hf-api` (`https://github.com/DePasqualeOrg/swift-hf-api`) or any HF-compatible downloader. |
| `import Generation`, `import Models`, `import Transformers` | — | **out of scope** — swift-tokenizers covers only the `Tokenizers` surface. Generation/Models from swift-transformers must be replaced separately if Flux2Swift uses them (Flux2Swift currently imports `Transformers` in two targets — verify usage). |

### Verdict on the six specific questions

1. **`tokenizer.decode(tokens: [Int])`**: renamed to `decode(tokenIds: [Int])`. Mechanical rename.
2. **`tokenizer.decode(_ tokens: [Int], skipSpecialTokens: Bool)` (positional)**: never existed in swift-transformers 1.3.0 (only labeled `decode(tokens:skipSpecialTokens:)`). swift-tokenizers does NOT add a positional overload. Verdict: **does not exist in either package**.
3. **`AutoTokenizer.from(modelFolder: URL)`**: confirmed renamed to `from(directory: URL)`. Also lost the `hubApi:` and `strict:` parameters.
4. **`AutoTokenizer.from(pretrained: String)`**: **REMOVED**. No replacement in the package itself; download-then-load is the new pattern.
5. **`applyChatTemplate(messages:)`**: same call site still compiles thanks to default args, BUT the default value of `addGenerationPrompt` changed from `false` → `true`. Audit every existing call.
6. **`bosTokenId` / `eosTokenId`**: still optional `Int?` properties on `Tokenizer`. Identical. Same for `unknownTokenId`. (`padTokenId` does not exist on either.)

---

## Backend Trait

Defined in `Package.swift:195–199`:

```swift
traits: [
    .default(enabledTraits: [defaultBackendTrait]),  // "Swift" or "Rust" via env
    .trait(name: "Swift"),
    .trait(name: "Rust"),
],
```

- **Available traits**: `Swift`, `Rust`. Mutually exclusive — `BackendSelection.swift` `#error`s if both or neither are set (except during DocC builds via `TOKENIZERS_DOCS_BUILD`).
- **Default**: `Swift`.
- **Pure-Swift backend** (`Swift` trait): pulls `swift-jinja` (Jinja product) + `yyjson` (Package.swift:108–109).
- **Rust backend** (`Rust` trait): pulls `TokenizersRust` binary `.binaryTarget` — an XCFramework hosted at the GitHub release asset URL `tokenizers-rust-0.4.1/TokenizersRust-0.4.1.xcframework.zip` with checksum `9b403e6053eefdcbb3a5aac62577467a4c1ae970e84df0ac28bd22c624bc0832` (Package.swift:11–13). **Yes, the Rust trait pulls a binary XCFramework dep.**

### Opting into the Rust backend

**Under SPM (Package.swift consumer)**: pass `traits:` in the dependency declaration:
```swift
.package(
    url: "https://github.com/DePasqualeOrg/swift-tokenizers.git",
    from: "0.4.2",
    traits: ["Rust"]
)
```

**Under Xcode**: `File → Packages → Package Traits…`, toggle `Rust` on and `Swift` off.

**Under raw `xcodebuild` (CI)**: `xcodebuild` has no `--traits` flag (Package.swift:50–52 comment confirms this). The package supports an env var workaround that flips the *default* trait at manifest-eval time:
```bash
TOKENIZERS_BACKEND=Rust xcodebuild build -scheme YourScheme -destination 'platform=macOS,arch=arm64'
```
This must be set both at resolve time and build time — it's read in `Package.swift` itself (line 52), so the manifest sees a different default trait set.

**Under `swift build` / `swift test`**: use `--traits Rust` (SwiftPM 6.1+ supports `--traits` directly). However, per global instructions, do NOT use `swift build`/`swift test` in this project — go through XcodeBuildMCP or raw `xcodebuild`.

---

## Transitive Dependencies

From `Package.swift:58–62`:

| Dep | Pin | When pulled |
|---|---|---|
| `https://github.com/huggingface/swift-jinja.git` | `from: "2.0.0"` | always declared; only linked when `Swift` trait is on (Package.swift:108) |
| `https://github.com/ibireme/yyjson.git` | `exact: "0.12.0"` | always declared; only linked when `Swift` trait is on (Package.swift:109) |
| `https://github.com/DePasqualeOrg/swift-hf-api.git` | `from: "0.2.0"` | always declared; **only consumed by the `TokenizersTests` target**, not by the public `Tokenizers` library (Package.swift:158). Downstream consumers do NOT link this product but it WILL appear in their `Package.resolved`. |

Optional, gated by env vars (won't normally appear in a downstream's resolved file unless they set the env var at resolve time):

- `https://github.com/ml-explore/mlx-swift-lm.git` `from: "3.31.3"` — only when `TOKENIZERS_ENABLE_BENCHMARKS=1` AND macOS.
- `https://github.com/swiftlang/swift-docc-plugin` `from: "1.4.0"` — only when `TOKENIZERS_ENABLE_DOCS=1`.

**Inherited transitive deps for Flux2Swift's `Package.resolved` (Swift trait, normal build)**:
- `swift-jinja` (already in Flux2Swift via swift-transformers 1.x — same package, expect a unification of the resolved version)
- `yyjson` (already in Flux2Swift via swift-transformers — pinned at `0.12.0` exact; matches swift-transformers 1.3.0's pin, no conflict)
- `swift-hf-api` (NEW — declared but only used by the test target, will still appear in the resolved file)

If using the Rust trait, additionally the binary XCFramework `TokenizersRust-0.4.1.xcframework.zip` (~ tens of MB; downloaded on resolve; cached in `~/.swiftpm/security`).

**Net dep count vs swift-transformers 1.3.0**: swift-transformers 1.x pulls `swift-jinja`, `swift-huggingface`, `swift-collections`, `swift-crypto`, `yyjson` (5 deps). swift-tokenizers 0.4.2 pulls `swift-jinja`, `yyjson`, `swift-hf-api` (3 deps). **Net reduction**: removes `swift-huggingface`, `swift-collections`, `swift-crypto` (the entire Hub stack). This is a real win on resolve time and binary size.

---

## Known Issues / Open PRs Worth Flagging

- **0 open issues** on https://github.com/DePasqualeOrg/swift-tokenizers (`gh api repos/DePasqualeOrg/swift-tokenizers/issues?state=open` returned empty as of 2026-04-27).
- **0 open PRs** as of 2026-04-27.
- The most recent merged PR (#25, in `0.4.2`) was a packaging fix only.
- The `0.4.1` release notes mention PR #22 "Align with Python Transformers v5 and Rust Tokenizers" — this is a deliberate breaking realignment vs. swift-transformers 1.x semantics. Behavior parity with HuggingFace Python `transformers v5` is the design goal, not parity with `huggingface/swift-transformers 1.x`. Flag for downstream behavior-diff testing especially around chat templates.
- **Behavior change to flag**: `applyChatTemplate(...)`'s `addGenerationPrompt` default switched from `false` (swift-transformers 1.x) to `true` (swift-tokenizers 0.4.2). Any Flux2Swift call site that omitted this argument will start emitting a generation prompt. Audit every `applyChatTemplate` call before merging.
- **No `from(pretrained:)`**: this is the largest migration friction point. If Flux2Swift loads tokenizers by HF model id today (e.g., `google/flan-t5-xxl` / `openai/clip-vit-large-patch14`), the migration MUST also introduce a download step. Two options: (a) add `swift-hf-api` as a direct dep and call `HFAPI.snapshot(...)` then `AutoTokenizer.from(directory:)`; (b) keep `huggingface/swift-transformers`'s `Hub` module just for downloads while moving tokenizing to swift-tokenizers (bigger dep graph, transitional).
- **Sendable**: `Tokenizer` is `Sendable`, `PreTrainedTokenizer` is `@unchecked Sendable`. Under Swift 6.2 strict concurrency, `@unchecked` will be tolerated but flagged by some linters. No strict-concurrency *errors* expected.
- **Rust trait at-rest size**: the binary XCFramework is downloaded and cached during package resolution. CI without internet access at resolve time will fail — verify your CI cache strategy.
- **Tokenizers module name collision**: BOTH packages export a product named `Tokenizers`. If Flux2Swift keeps `huggingface/swift-transformers` in the `Package.swift` while migrating gradually, SwiftPM will error with "multiple products named 'Tokenizers'". Migration must be a single atomic swap, or one of the imports must be aliased — and SPM doesn't easily alias product names. Practical answer: complete the swap in one PR.

---

## Recommended Adoption Path for Flux2Swift

Pin `from: "0.4.2"` (or `.upToNextMinor(from: "0.4.2")` for extra caution). Use the default `Swift` trait — Flux2Swift already accepts `swift-jinja` and `yyjson` transitively from swift-transformers, so the dep graph stays simple, and the Swift backend is the documented default with no XCFramework download to manage in CI. Defer the `Rust` trait until a measured tokenization hotspot demands it. Migration touches `FluxTextEncoders` and `Flux2Core` (both currently `import Transformers` from swift-transformers): swap `import Tokenizers` (single-product replacement), rename every `decode(tokens:)` call to `decode(tokenIds:)`, replace any `AutoTokenizer.from(modelFolder:)` with `from(directory:)`, replace any `AutoTokenizer.from(pretrained:)` with a download-then-`from(directory:)` flow (introduce `swift-hf-api` as a direct dep — it's already a transitive dep so no resolve change), and audit every `applyChatTemplate(...)` call site for the silent `addGenerationPrompt` default flip from `false` → `true`. Drop `import Hub` (no replacement — `HubApi.shared` is gone; use `HFAPI` from `swift-hf-api`). Flux2Swift's Swift 6.2 / macOS 26 / iOS 26 floors are all *higher* than swift-tokenizers' Swift 6.1 / macOS 14 / iOS 17 floors, so no toolchain-floor conflicts. Do the swap in a single PR — both packages export a product named `Tokenizers` and SwiftPM cannot resolve them simultaneously. Generation/Models from swift-transformers (if used) are out of scope for this package and need a separate migration plan.

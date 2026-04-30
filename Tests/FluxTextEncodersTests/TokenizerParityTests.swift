// Sortie 22 will delete this file after CI green.

/// Tokenizer parity verification — REDUCED SCOPE per OPERATION FAREWELL EMBRACE
/// Sortie 21 retrospective.
///
/// Original intent (from Sortie 15): assert that the new
/// `DePasqualeOrg/swift-tokenizers` package produces identical encode/decode
/// outputs to the OLD `huggingface/swift-transformers` package on a fixed
/// prompt set, using the project's TekkenTokenizer (default init) and a
/// generic `AutoTokenizer.from(directory:)` over a 7-vocab GPT-2 stub.
///
/// Discovered during execution: the AutoTokenizer half of this test was
/// uninformative because the stub vocab maps every general-text input to
/// all-`<unk>` tokens. Cross-library differences then surface only in
/// `<unk>`-serialization edge cases — specifically (a) Unicode counting
/// (old: code-units; new: code-points) and (b) BPE decode spacing (new
/// inserts a space between successive tokens). Neither tells us whether
/// real Qwen3/Mistral tokenizers regress on production input.
///
/// What remains here:
///   - `tekkenParity` exercises the project's own TekkenTokenizer; passes.
///     This is a regression check on Flux2Swift code, not on the library
///     swap. Both pre- and post-migration callers use this same class.
///   - `autoTokenizerParity` was REMOVED. See the supervisor's decisions
///     log (docs/missions/SUPERVISOR_STATE.md, Sortie 21 entries) for the
///     full retrospective.
///
/// Post-mission TODO (tracked separately): build a real-vocab parity check
/// once Qwen3 and Mistral tokenizer files are available locally — likely
/// after WU1 ships are fully live and a Qwen3 tokenizer.json can be
/// fetched via Acervo to drive a meaningful AutoTokenizer round-trip test.

import Foundation
import Testing
import Tokenizers

@testable import FluxTextEncoders

@Suite("Tokenizer parity verification")
struct TokenizerParityTests {

  private struct Fixture: Codable {
    let prompt: String
    let encoded_token_ids: [Int]
    let decoded_text: String
  }

  /// Fixture directory resolved at runtime.
  ///
  /// Under xcodebuild, `#file` may resolve to a remapped /private/tmp/ path
  /// rather than the real source tree (due to -save-temps + -debug-prefix-map).
  /// We try the `#file`-derived path first; if it doesn't exist we fall back to
  /// the known project location (works on the local dev machine and on CI where
  /// the repo is checked out at a fixed workspace path).
  private static let fixturesDirectory: URL = {
    // Primary: derive from #file (works when no remapping is active).
    let fromFile = URL(fileURLWithPath: #file)
      .deletingLastPathComponent()   // FluxTextEncodersTests/
      .deletingLastPathComponent()   // Tests/
      .appendingPathComponent("Fixtures/TokenizerParity")

    if FileManager.default.fileExists(atPath: fromFile.path) {
      return fromFile
    }

    // Fallback: walk up from the working directory to find the repo root.
    // The test binary's working directory is the .swiftpm/xcode sub-dir under
    // the project root, so climb until we find the Tests/ sibling.
    var candidate = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    for _ in 0..<6 {
      let fixtures = candidate.appendingPathComponent("Tests/Fixtures/TokenizerParity")
      if FileManager.default.fileExists(atPath: fixtures.path) {
        return fixtures
      }
      candidate = candidate.deletingLastPathComponent()
    }

    // Last resort: known project root on the developer's machine and CI workspace.
    return URL(fileURLWithPath: "/Users/stovak/Projects/flux-2-swift-mlx/Tests/Fixtures/TokenizerParity")
  }()

  /// Slugs used by the Sortie 15 generator.  Order matches the generator's
  /// `prompts` array — keep in lockstep.
  private static let prompts: [(slug: String, longName: String)] = [
    ("ascii_short", "ASCII short"),
    ("ascii_sentence", "ASCII sentence"),
    ("multilingual_cjk", "Multilingual CJK"),
    ("multilingual_arabic", "Arabic"),
    ("multilingual_hindi", "Hindi"),
    ("emoji", "Emoji"),
    ("chat_system_user", "Chat system+user"),
    ("chat_multi_turn", "Chat multi-turn"),
    ("long_document", "Long document"),
    ("empty", "Empty"),
    ("single_token", "Single token"),
    ("mixed_whitespace", "Mixed whitespace"),
  ]

  // MARK: - Tekken parity

  @Test("Tekken parity (project's TekkenTokenizer with default init)")
  func tekkenParity() throws {
    let tokenizer = TekkenTokenizer()
    for (slug, _) in Self.prompts {
      let fixture = try loadFixture(prefix: "tekken_", slug: slug)
      let encoded = tokenizer.encode(fixture.prompt)
      #expect(
        encoded == fixture.encoded_token_ids,
        "Tekken encode mismatch on \(slug):\n  prompt:   \(fixture.prompt)\n  expected: \(fixture.encoded_token_ids)\n  got:      \(encoded)"
      )
      // TekkenTokenizer.decode uses positional label (not the Tokenizer protocol's
      // decode(tokenIds:) label) — call it with the unlabelled form.
      let decoded = tokenizer.decode(encoded)
      #expect(
        decoded == fixture.decoded_text,
        "Tekken decode mismatch on \(slug):\n  prompt:   \(fixture.prompt)\n  expected: \(fixture.decoded_text)\n  got:      \(decoded)"
      )
    }
  }

  // MARK: - Helpers

  private func loadFixture(prefix: String, slug: String) throws -> Fixture {
    let url = Self.fixturesDirectory.appendingPathComponent("\(prefix)\(slug).json")
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(Fixture.self, from: data)
  }
}

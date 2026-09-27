// CaptionParserTests.swift - Pure parsing tests for training dataset captions.
//
// CaptionParser turns user-supplied .txt / metadata.jsonl files into training
// captions. Malformed input here flows straight into LoRA training, so these
// tests pin the parsing, field-precedence, trigger-word and validation rules.

import Foundation
import Testing

@testable import Flux2Core

@Suite final class CaptionParserTests {

  let tempDir: URL

  init() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("CaptionParserTests_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  deinit {
    try? FileManager.default.removeItem(at: tempDir)
  }

  private func write(_ contents: String, to name: String) throws {
    try contents.write(
      to: tempDir.appendingPathComponent(name), atomically: true, encoding: .utf8)
  }

  /// Image files are only checked by extension, so empty placeholders suffice.
  private func touch(_ name: String) {
    FileManager.default.createFile(
      atPath: tempDir.appendingPathComponent(name).path, contents: Data())
  }

  // MARK: - parseJSONLine

  @Test func jsonLinePrefersPromptOverCaptionOverText() throws {
    let parser = CaptionParser()

    let all = try parser.parseJSONLine(
      #"{"file_name": "a.png", "prompt": "P", "caption": "C", "text": "T"}"#)
    #expect(all.caption == "P")

    let captionAndText = try parser.parseJSONLine(
      #"{"file_name": "a.png", "caption": "C", "text": "T"}"#)
    #expect(captionAndText.caption == "C")

    let textOnly = try parser.parseJSONLine(#"{"file_name": "a.png", "text": "T"}"#)
    #expect(textOnly.caption == "T")
    #expect(textOnly.filename == "a.png")
  }

  @Test func jsonLineCollectsStringAndNumericMetadataOnly() throws {
    let parser = CaptionParser()
    let parsed = try parser.parseJSONLine(
      #"{"file_name": "a.png", "prompt": "P", "style": "ink", "weight": 2, "tags": ["x"]}"#)

    // Arrays are dropped; strings and numbers are stringified; caption/file keys excluded.
    #expect(parsed.metadata == ["style": "ink", "weight": "2"])
  }

  @Test func jsonLineWithoutExtraFieldsHasNilMetadata() throws {
    let parsed = try CaptionParser().parseJSONLine(#"{"file_name": "a.png", "caption": "C"}"#)
    #expect(parsed.metadata == nil)
  }

  @Test func jsonLineMissingFileNameThrows() {
    #expect {
      try CaptionParser().parseJSONLine(#"{"prompt": "P"}"#)
    } throws: { error in
      guard case CaptionParserError.missingField(let field) = error else { return false }
      return field == "file_name"
    }
  }

  @Test func jsonLineMissingCaptionThrows() {
    #expect {
      try CaptionParser().parseJSONLine(#"{"file_name": "a.png", "other": "x"}"#)
    } throws: { error in
      guard case CaptionParserError.missingField(let field) = error else { return false }
      return field == "prompt/caption/text"
    }
  }

  @Test func jsonLineThatIsNotAnObjectThrowsInvalidJSON() {
    #expect {
      try CaptionParser().parseJSONLine(#"["a.png", "caption"]"#)
    } throws: { error in
      guard case CaptionParserError.invalidJSON = error else { return false }
      return true
    }
  }

  @Test func jsonLineThatIsNotJSONThrows() {
    #expect(throws: (any Error).self) {
      try CaptionParser().parseJSONLine("not json at all")
    }
  }

  // MARK: - Trigger word / caption processing

  @Test(arguments: ["[trigger]", "[TRIGGER]", "{trigger}"])
  func triggerPlaceholdersAreReplaced(placeholder: String) throws {
    let parser = CaptionParser(triggerWord: "sks")
    let parsed = try parser.parseJSONLine(
      #"{"file_name": "a.png", "prompt": "a photo of \#(placeholder) dog"}"#)
    #expect(parsed.caption == "a photo of sks dog")
  }

  @Test func placeholderIsLeftVerbatimWithoutTriggerWord() throws {
    let parsed = try CaptionParser().parseJSONLine(
      #"{"file_name": "a.png", "prompt": "a photo of [trigger] dog"}"#)
    #expect(parsed.caption == "a photo of [trigger] dog")
  }

  @Test func captionWhitespaceIsTrimmedAndDoubleSpacesCollapsed() throws {
    let parsed = try CaptionParser().parseJSONLine(
      #"{"file_name": "a.png", "prompt": "  a  red   car \n"}"#)
    // processCaption performs a single "  " → " " pass, so a run of three
    // spaces collapses to two. Pinning current behavior.
    #expect(parsed.caption == "a red  car")
  }

  @Test func parseTextFileTrimsAndInjectsTrigger() throws {
    try write("\n  portrait of [trigger]  \n", to: "cap.txt")
    let caption = try CaptionParser(triggerWord: "ohwx")
      .parseTextFile(tempDir.appendingPathComponent("cap.txt"))
    #expect(caption == "portrait of ohwx")
  }

  // MARK: - parseDataset

  @Test func textDatasetPairsImagesWithCaptionsAndSkipsUncaptioned() throws {
    touch("one.png")
    touch("two.JPG")
    touch("orphan.webp")  // no caption → skipped
    touch("notes.md")  // not an image → ignored
    try write("first [trigger]", to: "one.txt")
    try write("second", to: "two.txt")

    let results = try CaptionParser(triggerWord: "tok")
      .parseDataset(at: tempDir, extension: "TXT")
      .sorted { $0.filename < $1.filename }

    #expect(results.map(\.filename) == ["one.png", "two.JPG"])
    #expect(results.map(\.caption) == ["first tok", "second"])
  }

  @Test(arguments: ["jsonl", "json"])
  func jsonlDatasetSkipsBlankAndMalformedLines(ext: String) throws {
    try write(
      """
      {"file_name": "a.png", "prompt": "alpha"}

      not json
      {"file_name": "b.png"}

      {"file_name": "c.png", "text": "gamma"}
      """,
      to: "metadata.jsonl")

    let results = try CaptionParser().parseDataset(at: tempDir, extension: ext)
    #expect(results.map(\.filename) == ["a.png", "c.png"])
    #expect(results.map(\.caption) == ["alpha", "gamma"])
  }

  @Test func jsonlDatasetWithoutMetadataFileThrows() {
    #expect {
      try CaptionParser().parseDataset(at: tempDir, extension: "jsonl")
    } throws: { error in
      guard case CaptionParserError.metadataFileNotFound(let url) = error else { return false }
      return url.lastPathComponent == "metadata.jsonl"
    }
  }

  @Test func unsupportedExtensionThrows() {
    #expect {
      try CaptionParser().parseDataset(at: tempDir, extension: "csv")
    } throws: { error in
      guard case CaptionParserError.unsupportedFormat(let fmt) = error else { return false }
      return fmt == "csv"
    }
  }

  // MARK: - validateDataset

  @Test func validateMissingDirectoryIsInvalid() {
    let missing = tempDir.appendingPathComponent("does-not-exist")
    let result = CaptionParser().validateDataset(at: missing, extension: "txt")

    #expect(!result.isValid)
    #expect(result.imageCount == 0)
    #expect(result.errors.count == 1)
    #expect(result.errors[0].hasPrefix("Dataset directory not found"))
  }

  @Test func validateTextDatasetWarnsOnMissingCaptionAndSmallDataset() {
    touch("a.png")
    touch("b.png")
    try? write("cap", to: "a.txt")

    let result = CaptionParser().validateDataset(at: tempDir, extension: "txt")

    #expect(result.isValid)  // warnings only
    #expect(result.imageCount == 2)
    #expect(result.errors.isEmpty)
    #expect(result.warnings.contains("Missing caption: b.png"))
    #expect(result.warnings.contains { $0.hasPrefix("Very small dataset (2 images)") })
    #expect(result.warnings.contains("2 images but 1 captions"))
  }

  @Test func validateLargeFullyCaptionedTextDatasetHasNoWarnings() throws {
    for i in 0..<5 {
      touch("img\(i).png")
      try write("caption \(i)", to: "img\(i).txt")
    }
    let result = CaptionParser().validateDataset(at: tempDir, extension: "txt")
    #expect(result.isValid)
    #expect(result.imageCount == 5)
    #expect(result.warnings.isEmpty)
  }

  @Test func validateJSONLDatasetWithoutMetadataIsInvalid() {
    touch("a.png")
    let result = CaptionParser().validateDataset(at: tempDir, extension: "jsonl")
    #expect(!result.isValid)
    #expect(result.errors == ["metadata.jsonl not found"])
    #expect(result.imageCount == 1)
  }

  @Test func validateJSONLDatasetCountsParsedCaptions() throws {
    touch("a.png")
    touch("b.png")
    try write(
      """
      {"file_name": "a.png", "prompt": "alpha"}
      garbage
      """,
      to: "metadata.jsonl")

    let result = CaptionParser().validateDataset(at: tempDir, extension: "jsonl")
    #expect(result.isValid)
    #expect(result.imageCount == 2)
    #expect(result.warnings.contains("2 images but 1 captions"))
  }

  // MARK: - DatasetValidationResult.summary

  @Test func summaryListsErrorsWarningsAndStatus() {
    let invalid = DatasetValidationResult(
      imageCount: 3, errors: ["boom"], warnings: ["careful"])
    #expect(!invalid.isValid)
    #expect(
      invalid.summary == """
        Dataset Validation:
          Images: 3
          Errors:
            - boom
          Warnings:
            - careful
          Status: Invalid
        """)

    let valid = DatasetValidationResult(imageCount: 10, errors: [], warnings: [])
    #expect(valid.isValid)
    #expect(
      valid.summary == """
        Dataset Validation:
          Images: 10
          Status: Valid
        """)
  }

  @Test func zeroImagesIsInvalidEvenWithoutErrors() {
    let result = DatasetValidationResult(imageCount: 0, errors: [], warnings: [])
    #expect(!result.isValid)
    #expect(result.summary.hasSuffix("Status: Invalid"))
  }
}

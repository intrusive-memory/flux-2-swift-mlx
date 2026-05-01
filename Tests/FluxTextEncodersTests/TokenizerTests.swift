/**
 * TokenizerTests.swift
 * Unit tests for TekkenTokenizer
 */

import Testing

@testable import FluxTextEncoders

@Suite("TokenizerTests")
struct TokenizerTests {

  let tokenizer: TekkenTokenizer

  init() {
    // Use default tokenizer (no model path)
    tokenizer = TekkenTokenizer()
  }

  // MARK: - Basic Encoding Tests

  @Test func encodeEmptyString() {
    let tokens = tokenizer.encode("")
    #expect(tokens.isEmpty, "Empty string should produce empty token array")
  }

  @Test func encodeSimpleText() {
    let tokens = tokenizer.encode("Hello")
    #expect(!tokens.isEmpty, "Non-empty string should produce tokens")
  }

  @Test func encodeWithSpecialTokens() {
    let tokensWithoutSpecial = tokenizer.encode("Hello", addSpecialTokens: false)
    let tokensWithSpecial = tokenizer.encode("Hello", addSpecialTokens: true)

    // With special tokens should have BOS at start and EOS at end
    #expect(
      tokensWithSpecial.count > tokensWithoutSpecial.count,
      "Adding special tokens should increase token count")

    // First token should be BOS
    if !tokensWithSpecial.isEmpty {
      #expect(
        tokensWithSpecial.first == tokenizer.bosToken,
        "First token with special tokens should be BOS")
    }

    // Last token should be EOS
    if !tokensWithSpecial.isEmpty {
      #expect(
        tokensWithSpecial.last == tokenizer.eosToken,
        "Last token with special tokens should be EOS")
    }
  }

  // MARK: - Special Token Properties

  @Test func specialTokenProperties() {
    // BOS token should be 1 by default
    #expect(tokenizer.bosToken == 1, "BOS token should be 1")

    // EOS token should be 2 by default
    #expect(tokenizer.eosToken == 2, "EOS token should be 2")

    // PAD token should be 11 by default
    #expect(tokenizer.padToken == 11, "PAD token should be 11")
  }

  @Test func vocabSize() {
    let vocabSize = tokenizer.vocabSize
    #expect(vocabSize > 0, "Vocab size should be positive")
  }

  // MARK: - Chat Template Tests

  @Test func applyChatTemplateSimple() {
    let messages: [[String: String]] = [
      ["role": "user", "content": "Hello"]
    ]

    let prompt = tokenizer.applyChatTemplate(messages: messages, addGenerationPrompt: false)

    // Should contain the user message
    #expect(prompt.contains("Hello"), "Chat template should contain user message")

    // Should contain instruction markers
    #expect(
      prompt.contains("[INST]") || prompt.contains("user"),
      "Chat template should contain instruction markers")
  }

  @Test func applyChatTemplateWithSystem() {
    let messages: [[String: String]] = [
      ["role": "system", "content": "You are helpful"],
      ["role": "user", "content": "Hi"],
    ]

    let prompt = tokenizer.applyChatTemplate(messages: messages, addGenerationPrompt: false)

    // Should contain both system and user content
    #expect(
      prompt.contains("You are helpful") || prompt.contains("helpful"),
      "Chat template should contain system message")
    #expect(prompt.contains("Hi"), "Chat template should contain user message")
  }

  @Test func applyChatTemplateMultiTurn() {
    let messages: [[String: String]] = [
      ["role": "user", "content": "Hello"],
      ["role": "assistant", "content": "Hi there!"],
      ["role": "user", "content": "How are you?"],
    ]

    let prompt = tokenizer.applyChatTemplate(messages: messages, addGenerationPrompt: false)

    // Should contain all messages
    #expect(prompt.contains("Hello"), "Should contain first user message")
    #expect(prompt.contains("Hi there!"), "Should contain assistant message")
    #expect(prompt.contains("How are you?"), "Should contain second user message")
  }

  @Test func encodeChatMessagesProducesTokens() {
    let messages: [[String: String]] = [
      ["role": "user", "content": "Hello world"]
    ]

    let tokens = tokenizer.encodeChatMessages(messages: messages)

    #expect(!tokens.isEmpty, "Chat messages should produce tokens")

    // First token should be BOS
    #expect(
      tokens.first == tokenizer.bosToken,
      "First token should be BOS")
  }

  // MARK: - Decoding Tests

  @Test func decodeEmptyArray() {
    let text = tokenizer.decode([])
    #expect(text.isEmpty, "Empty token array should decode to empty string")
  }

  @Test func decodeSkipsSpecialTokensByDefault() {
    // Decode with special token IDs
    let tokens = [tokenizer.bosToken, tokenizer.eosToken]
    let text = tokenizer.decode(tokens, skipSpecialTokens: true)

    // Should not contain special token strings
    #expect(!text.contains("<s>"), "Should skip BOS token")
    #expect(!text.contains("</s>"), "Should skip EOS token")
  }

  @Test func batchDecode() {
    let tokenLists = [
      [tokenizer.bosToken],
      [tokenizer.eosToken],
    ]

    let decoded = tokenizer.batchDecode(tokenLists)

    #expect(decoded.count == 2, "Batch decode should return same number of strings")
  }

  // MARK: - Edge Cases

  @Test func encodeUnicodeText() {
    let tokens = tokenizer.encode("Hello 世界 🌍")
    // Should not crash and should produce some tokens
    #expect(!tokens.isEmpty, "Unicode text should produce tokens")
  }

  @Test func encodeLongText() {
    let longText = String(repeating: "Hello world. ", count: 100)
    let tokens = tokenizer.encode(longText)

    #expect(!tokens.isEmpty, "Long text should produce tokens")
    #expect(tokens.count > 10, "Long text should produce many tokens")
  }

  @Test func encodeSpecialCharacters() {
    let specialText = "Hello\nWorld\tTab\"Quote'Single"
    let tokens = tokenizer.encode(specialText)

    #expect(!tokens.isEmpty, "Text with special characters should produce tokens")
  }

  // MARK: - MLXArray Support
  // Note: Tests that require Metal/MLX GPU access are not included here.
  // They would need to be run in an environment with full MLX support.
  // The tokenizer's MLXArray functionality can be tested via integration tests
  // or directly in Xcode with proper Metal environment.
}

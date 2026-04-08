/**
 * GenerationResultTests.swift
 * Unit tests for GenerationResult
 */

import Testing
@testable import FluxTextEncoders

@Suite("GenerationResultTests")
struct GenerationResultTests {

    // MARK: - Initialization Tests

    @Test func generationResultInit() {
        let result = GenerationResult(
            text: "Hello world",
            tokens: [100, 200, 300],
            promptTokens: 10,
            generatedTokens: 3,
            totalTime: 1.5,
            tokensPerSecond: 2.0
        )

        #expect(result.text == "Hello world")
        #expect(result.tokens == [100, 200, 300])
        #expect(result.promptTokens == 10)
        #expect(result.generatedTokens == 3)
        #expect(result.totalTime == 1.5)
        #expect(result.tokensPerSecond == 2.0)
    }

    @Test func generationResultEmptyText() {
        let result = GenerationResult(
            text: "",
            tokens: [],
            promptTokens: 5,
            generatedTokens: 0,
            totalTime: 0.1,
            tokensPerSecond: 0.0
        )

        #expect(result.text.isEmpty)
        #expect(result.tokens.isEmpty)
        #expect(result.generatedTokens == 0)
    }

    // MARK: - Summary Tests

    @Test func summaryFormat() {
        let result = GenerationResult(
            text: "Test output",
            tokens: [1, 2, 3, 4, 5],
            promptTokens: 100,
            generatedTokens: 5,
            totalTime: 2.0,
            tokensPerSecond: 2.5
        )

        let summary = result.summary()

        #expect(summary.contains("Prompt: 100 tokens"), "Should show prompt tokens")
        #expect(summary.contains("Generated: 5 tokens"), "Should show generated tokens")
        #expect(summary.contains("2.5 tokens/s"), "Should show tokens per second")
        #expect(summary.contains("2.00s"), "Should show time")
    }

    @Test func summaryWithZeroTokens() {
        let result = GenerationResult(
            text: "",
            tokens: [],
            promptTokens: 0,
            generatedTokens: 0,
            totalTime: 0.0,
            tokensPerSecond: 0.0
        )

        let summary = result.summary()

        #expect(summary.contains("Prompt: 0 tokens"))
        #expect(summary.contains("Generated: 0 tokens"))
    }

    @Test func summaryWithLargeNumbers() {
        let result = GenerationResult(
            text: String(repeating: "a", count: 1000),
            tokens: Array(0..<1000),
            promptTokens: 5000,
            generatedTokens: 1000,
            totalTime: 50.0,
            tokensPerSecond: 20.0
        )

        let summary = result.summary()

        #expect(summary.contains("5000 tokens"))
        #expect(summary.contains("1000 tokens"))
        #expect(summary.contains("20.0 tokens/s"))
    }

    // MARK: - Token Count Consistency Tests

    @Test func tokenCountMatchesArray() {
        let tokens = [10, 20, 30, 40, 50]
        let result = GenerationResult(
            text: "test",
            tokens: tokens,
            promptTokens: 10,
            generatedTokens: tokens.count,
            totalTime: 1.0,
            tokensPerSecond: 5.0
        )

        #expect(result.tokens.count == result.generatedTokens, "Token count should match tokens array length")
    }

    // MARK: - Performance Metrics Tests

    @Test func highPerformanceMetrics() {
        let result = GenerationResult(
            text: "Fast generation",
            tokens: Array(0..<100),
            promptTokens: 50,
            generatedTokens: 100,
            totalTime: 1.0,
            tokensPerSecond: 100.0
        )

        #expect(result.tokensPerSecond == 100.0)

        let summary = result.summary()
        #expect(summary.contains("100.0 tokens/s"))
    }

    @Test func lowPerformanceMetrics() {
        let result = GenerationResult(
            text: "Slow generation",
            tokens: [1],
            promptTokens: 1000,
            generatedTokens: 1,
            totalTime: 10.0,
            tokensPerSecond: 0.1
        )

        let summary = result.summary()
        #expect(summary.contains("0.1 tokens/s"))
    }

    // MARK: - Sendable Conformance

    @Test func generationResultIsSendable() {
        let result = GenerationResult(
            text: "test",
            tokens: [1],
            promptTokens: 1,
            generatedTokens: 1,
            totalTime: 1.0,
            tokensPerSecond: 1.0
        )
        // Verify Sendable conformance by using it as Sendable
        let _: any Sendable = result
        #expect(result.text == "test")
    }

    // MARK: - Edge Cases

    @Test func veryLongText() {
        let longText = String(repeating: "Hello world. ", count: 10000)
        let result = GenerationResult(
            text: longText,
            tokens: Array(0..<10000),
            promptTokens: 100,
            generatedTokens: 10000,
            totalTime: 100.0,
            tokensPerSecond: 100.0
        )

        #expect(result.text.count == longText.count)
        #expect(result.tokens.count == 10000)
    }

    @Test func unicodeText() {
        let unicodeText = "Hello 世界 🌍 مرحبا"
        let result = GenerationResult(
            text: unicodeText,
            tokens: [1, 2, 3, 4, 5],
            promptTokens: 5,
            generatedTokens: 5,
            totalTime: 0.5,
            tokensPerSecond: 10.0
        )

        #expect(result.text == unicodeText)
    }

    @Test func specialCharactersInText() {
        let specialText = "Line1\nLine2\tTab\"Quote'Single"
        let result = GenerationResult(
            text: specialText,
            tokens: [1, 2, 3],
            promptTokens: 3,
            generatedTokens: 3,
            totalTime: 0.1,
            tokensPerSecond: 30.0
        )

        #expect(result.text == specialText)
    }

    // MARK: - Time Formatting Tests

    @Test func timeFormattingInSummary() {
        // Test various time values
        let shortResult = GenerationResult(
            text: "quick",
            tokens: [1],
            promptTokens: 1,
            generatedTokens: 1,
            totalTime: 0.01,
            tokensPerSecond: 100.0
        )

        let summary = shortResult.summary()
        #expect(summary.contains("0.01s"), "Should format short time correctly")
    }

    @Test func decimalPrecisionInSummary() {
        let result = GenerationResult(
            text: "test",
            tokens: [1, 2, 3],
            promptTokens: 10,
            generatedTokens: 3,
            totalTime: 1.23456,
            tokensPerSecond: 2.43902
        )

        let summary = result.summary()
        // Should be formatted with limited decimal places
        #expect(summary.contains("1.23s") || summary.contains("1.2s"), "Time should be formatted with limited decimals")
    }
}

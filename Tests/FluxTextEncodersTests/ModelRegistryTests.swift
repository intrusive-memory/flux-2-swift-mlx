/**
 * TextEncoderModelRegistryTests.swift
 * Unit tests for TextEncoderModelRegistry and ModelVariant
 */

import Testing
@testable import FluxTextEncoders

@Suite("TextEncoderModelRegistryTests")
@MainActor
struct TextEncoderModelRegistryTests {

    // MARK: - ModelVariant Tests

    @Test @MainActor func modelVariantRawValues() {
        #expect(ModelVariant.bf16.rawValue == "bf16")
        #expect(ModelVariant.mlx8bit.rawValue == "8bit")
        #expect(ModelVariant.mlx6bit.rawValue == "6bit")
        #expect(ModelVariant.mlx4bit.rawValue == "4bit")
    }

    @Test @MainActor func modelVariantDisplayNames() {
        #expect(ModelVariant.bf16.displayName == "Full Precision (BF16)")
        #expect(ModelVariant.mlx8bit.displayName == "8-bit Quantized")
        #expect(ModelVariant.mlx6bit.displayName == "6-bit Quantized")
        #expect(ModelVariant.mlx4bit.displayName == "4-bit Quantized")
    }

    @Test @MainActor func modelVariantShortNames() {
        #expect(ModelVariant.bf16.shortName == "BF16")
        #expect(ModelVariant.mlx8bit.shortName == "8-bit")
        #expect(ModelVariant.mlx6bit.shortName == "6-bit")
        #expect(ModelVariant.mlx4bit.shortName == "4-bit")
    }

    @Test @MainActor func modelVariantEstimatedSizes() {
        #expect(ModelVariant.bf16.estimatedSize == "~48GB")
        #expect(ModelVariant.mlx8bit.estimatedSize == "~25GB")
        #expect(ModelVariant.mlx6bit.estimatedSize == "~19GB")
        #expect(ModelVariant.mlx4bit.estimatedSize == "~14GB")
    }

    @Test @MainActor func modelVariantCaseIterable() {
        let allCases = ModelVariant.allCases
        #expect(allCases.count == 4, "Should have 4 model variants")
        #expect(allCases.contains(.bf16))
        #expect(allCases.contains(.mlx8bit))
        #expect(allCases.contains(.mlx6bit))
        #expect(allCases.contains(.mlx4bit))
    }

    // MARK: - ModelInfo Tests

    @Test @MainActor func modelInfoInit() {
        let model = ModelInfo(
            id: "test-model",
            repoId: "org/test-model",
            name: "Test Model",
            description: "A test model",
            variant: .mlx8bit,
            parameters: "24B"
        )

        #expect(model.id == "test-model")
        #expect(model.repoId == "org/test-model")
        #expect(model.name == "Test Model")
        #expect(model.description == "A test model")
        #expect(model.variant == .mlx8bit)
        #expect(model.parameters == "24B")
    }

    // MARK: - TextEncoderModelRegistry Tests

    @Test @MainActor func textEncoderModelRegistrySharedInstance() {
        let registry1 = TextEncoderModelRegistry.shared
        let registry2 = TextEncoderModelRegistry.shared
        #expect(registry1 === registry2, "Shared instance should be singleton")
    }

    @Test @MainActor func textEncoderModelRegistryAllModels() {
        let models = TextEncoderModelRegistry.shared.allModels()

        #expect(models.count >= 3,
                                    "Should have at least 3 models (8bit, 6bit, 4bit)")
    }

    @Test @MainActor func textEncoderModelRegistryContainsExpectedVariants() {
        let models = TextEncoderModelRegistry.shared.allModels()
        let variants = models.map { $0.variant }

        #expect(variants.contains(.mlx8bit), "Should have 8-bit model")
        #expect(variants.contains(.mlx4bit), "Should have 4-bit model")
    }

    @Test @MainActor func textEncoderModelRegistryDefaultModel() {
        let defaultModel = TextEncoderModelRegistry.shared.defaultModel()

        #expect(defaultModel.variant == .mlx8bit,
                      "Default model should be 8-bit")
        #expect(!defaultModel.id.isEmpty, "Default model should have ID")
        #expect(!defaultModel.repoId.isEmpty, "Default model should have repo ID")
    }

    @Test @MainActor func textEncoderModelRegistryFindByVariant() {
        let model8bit = TextEncoderModelRegistry.shared.model(withVariant: .mlx8bit)
        let model4bit = TextEncoderModelRegistry.shared.model(withVariant: .mlx4bit)

        #expect(model8bit != nil, "Should find 8-bit model")
        #expect(model4bit != nil, "Should find 4-bit model")

        #expect(model8bit?.variant == .mlx8bit)
        #expect(model4bit?.variant == .mlx4bit)
    }

    @Test @MainActor func textEncoderModelRegistryFindById() {
        let models = TextEncoderModelRegistry.shared.allModels()
        guard let firstModel = models.first else {
            Issue.record("Should have at least one model")
            return
        }

        let foundModel = TextEncoderModelRegistry.shared.model(withId: firstModel.id)

        #expect(foundModel != nil, "Should find model by ID")
        #expect(foundModel?.id == firstModel.id)
    }

    @Test @MainActor func textEncoderModelRegistryFindByIdNotFound() {
        let model = TextEncoderModelRegistry.shared.model(withId: "non-existent-model-id")
        #expect(model == nil, "Should return nil for non-existent ID")
    }

    @Test @MainActor func textEncoderModelRegistryFindByVariantNotFound() {
        // All variants should exist, but test the lookup mechanism
        let model = TextEncoderModelRegistry.shared.model(withVariant: .bf16)
        // bf16 may or may not be registered depending on implementation
        // Just verify the lookup doesn't crash
        _ = model
    }

    // MARK: - Model Metadata Tests

    @Test @MainActor func registeredModelsHaveValidMetadata() {
        let models = TextEncoderModelRegistry.shared.allModels()

        for model in models {
            #expect(!model.id.isEmpty, "Model ID should not be empty")
            #expect(!model.repoId.isEmpty, "Repo ID should not be empty")
            #expect(!model.name.isEmpty, "Name should not be empty")
            #expect(!model.description.isEmpty, "Description should not be empty")
            #expect(model.parameters == "24B", "Parameters should be 24B")
        }
    }

    @Test @MainActor func registeredModelsHaveHuggingFaceRepos() {
        let models = TextEncoderModelRegistry.shared.allModels()

        for model in models {
            // Repo IDs should follow HuggingFace format: org/repo
            #expect(model.repoId.contains("/"),
                         "Repo ID '\(model.repoId)' should be in HuggingFace format")
        }
    }

    // MARK: - Tekken JSON URL Test

    @Test @MainActor func tekkenJsonURLIsValid() {
        let url = TextEncoderModelRegistry.tekkenJsonURL

        #expect(url.hasPrefix("https://"),
                     "Tekken JSON URL should use HTTPS")
        #expect(url.contains("huggingface.co"),
                     "Tekken JSON URL should be on HuggingFace")
        #expect(url.contains("tekken.json"),
                     "URL should point to tekken.json")
    }

    // MARK: - Sendable Conformance

    @Test @MainActor func modelVariantIsSendable() {
        let variant = ModelVariant.mlx8bit
        let _: any Sendable = variant
        #expect(variant.rawValue == "8bit")
    }

    @Test @MainActor func modelInfoIsSendable() {
        let model = ModelInfo(
            id: "test",
            repoId: "test/test",
            name: "Test",
            description: "Test",
            variant: .mlx8bit,
            parameters: "24B"
        )
        let _: any Sendable = model
        #expect(model.id == "test")
    }

    // MARK: - Gated Status Tests

    @Test @MainActor func modelVariantIsGated() {
        // bf16 from mistralai is gated
        #expect(ModelVariant.bf16.isGated)

        // Quantized versions from lmstudio-community are NOT gated
        #expect(!ModelVariant.mlx8bit.isGated)
        #expect(!ModelVariant.mlx6bit.isGated)
        #expect(!ModelVariant.mlx4bit.isGated)
    }

    @Test @MainActor func modelInfoIsGated() {
        let models = TextEncoderModelRegistry.shared.allModels()

        // Check bf16 model is gated
        if let bf16Model = models.first(where: { $0.variant == .bf16 }) {
            #expect(bf16Model.isGated)
        }

        // Check quantized models are NOT gated
        if let mlx8bit = models.first(where: { $0.variant == .mlx8bit }) {
            #expect(!mlx8bit.isGated)
        }
    }

    @Test @MainActor func qwen3VariantIsGated() {
        // All Qwen3 models from lmstudio-community are NOT gated
        #expect(!Qwen3Variant.qwen3_4B_8bit.isGated)
        #expect(!Qwen3Variant.qwen3_4B_4bit.isGated)
        #expect(!Qwen3Variant.qwen3_8B_8bit.isGated)
        #expect(!Qwen3Variant.qwen3_8B_4bit.isGated)
    }

    // MARK: - HuggingFace URL Tests

    @Test @MainActor func modelVariantHuggingFaceURL() {
        for variant in ModelVariant.allCases {
            #expect(variant.huggingFaceURL.starts(with: "https://huggingface.co/"))
            #expect(variant.huggingFaceURL.contains(variant.repoId))
        }
    }

    @Test @MainActor func modelInfoHuggingFaceURL() {
        let models = TextEncoderModelRegistry.shared.allModels()

        for model in models {
            #expect(model.huggingFaceURL.starts(with: "https://huggingface.co/"))
            #expect(model.huggingFaceURL.contains(model.repoId))
        }
    }

    @Test @MainActor func qwen3VariantHuggingFaceURL() {
        for variant in Qwen3Variant.allCases {
            #expect(variant.huggingFaceURL.starts(with: "https://huggingface.co/"))
            #expect(variant.huggingFaceURL.contains(variant.repoId))
        }
    }

    @Test @MainActor func modelVariantRepoIdValues() {
        // bf16 should be from mistralai
        #expect(ModelVariant.bf16.repoId.contains("mistralai"))

        // Quantized should be from lmstudio-community
        #expect(ModelVariant.mlx8bit.repoId.contains("lmstudio-community"))
        #expect(ModelVariant.mlx6bit.repoId.contains("lmstudio-community"))
        #expect(ModelVariant.mlx4bit.repoId.contains("lmstudio-community"))
    }

    @Test @MainActor func qwen3VariantRepoIdValues() {
        // All Qwen3 should be from lmstudio-community
        for variant in Qwen3Variant.allCases {
            #expect(variant.repoId.contains("lmstudio-community"))
        }
    }

    @Test @MainActor func modelVariantEstimatedSizeGB() {
        #expect(ModelVariant.bf16.estimatedSizeGB == 48)
        #expect(ModelVariant.mlx8bit.estimatedSizeGB == 25)
        #expect(ModelVariant.mlx6bit.estimatedSizeGB == 19)
        #expect(ModelVariant.mlx4bit.estimatedSizeGB == 14)
    }

    @Test @MainActor func qwen3VariantEstimatedSizeGB() {
        #expect(Qwen3Variant.qwen3_4B_8bit.estimatedSizeGB == 4)
        #expect(Qwen3Variant.qwen3_4B_4bit.estimatedSizeGB == 2)
        #expect(Qwen3Variant.qwen3_8B_8bit.estimatedSizeGB == 8)
        #expect(Qwen3Variant.qwen3_8B_4bit.estimatedSizeGB == 4)
    }

    // MARK: - License Tests

    @Test @MainActor func modelVariantLicense() {
        for variant in ModelVariant.allCases {
            #expect(variant.license.contains("Apache"))
            #expect(variant.isCommercialUseAllowed)
        }
    }

    @Test @MainActor func qwen3VariantLicense() {
        for variant in Qwen3Variant.allCases {
            #expect(variant.license.contains("Apache"))
            #expect(variant.isCommercialUseAllowed)
        }
    }
}

# flux-2-swift-mlx Makefile
# Build, test, and install the Flux2 CLIs/app with full Metal shader support.
#
# Apple Silicon (arm64) only — MLX has no x86_64 path.

PACKAGE_SCHEME = Flux2Swift-Package
BINARIES = FluxEncodersCLI Flux2CLI Flux2App
BIN_DIR = ./bin
SPM_DIR = .spm

DESTINATION_MAC = platform=macOS,arch=arm64
DESTINATION_IOS = platform=iOS Simulator,name=iPhone 17,OS=26.1

DERIVED_DATA = $(HOME)/Library/Developer/Xcode/DerivedData

# Canonical xcodebuild flags from TESTING_REQUIREMENTS.md §2.
# ARCHS=arm64 + ONLY_ACTIVE_ARCH=YES are non-negotiable: MLX is arm64-only.
XCODEBUILD_FLAGS = \
	-skipPackagePluginValidation \
	ARCHS=arm64 \
	ONLY_ACTIVE_ARCH=YES \
	COMPILER_INDEX_STORE_ENABLE=NO \
	-clonedSourcePackagesDirPath $(SPM_DIR)

.PHONY: all build build-ios release install resolve \
	test test-fte test-core test-gpu \
	lint lint-check \
	clean help

all: install

# Resolve all SPM package dependencies via xcodebuild
resolve:
	xcodebuild -resolvePackageDependencies \
		-scheme $(PACKAGE_SCHEME) \
		-clonedSourcePackagesDirPath $(SPM_DIR)
	@echo "Package dependencies resolved."

# Development build (macOS arm64, debug, no copy)
build: resolve
	xcodebuild build \
		-scheme $(PACKAGE_SCHEME) \
		-destination '$(DESTINATION_MAC)' \
		$(XCODEBUILD_FLAGS) \
		CODE_SIGNING_ALLOWED=NO

# iOS Simulator build (iPhone 17, iOS 26.1) — verifies the package compiles for iOS.
# OS=latest does not work in this environment; pin the exact version.
build-ios: resolve
	xcodebuild build \
		-scheme $(PACKAGE_SCHEME) \
		-destination '$(DESTINATION_IOS)' \
		-skipPackagePluginValidation \
		COMPILER_INDEX_STORE_ENABLE=NO \
		-clonedSourcePackagesDirPath $(SPM_DIR) \
		CODE_SIGNING_ALLOWED=NO

# Release build (macOS arm64) + copy CLIs and Metal bundle to ./bin
release: resolve
	xcodebuild build \
		-scheme $(PACKAGE_SCHEME) \
		-destination '$(DESTINATION_MAC)' \
		-configuration Release \
		$(XCODEBUILD_FLAGS) \
		CODE_SIGNING_ALLOWED=NO
	@$(MAKE) --no-print-directory _copy-binaries CONFIG=Release

# Debug build (macOS arm64) + copy CLIs and Metal bundle to ./bin (default)
install: resolve
	xcodebuild build \
		-scheme $(PACKAGE_SCHEME) \
		-destination '$(DESTINATION_MAC)' \
		$(XCODEBUILD_FLAGS) \
		CODE_SIGNING_ALLOWED=NO
	@$(MAKE) --no-print-directory _copy-binaries CONFIG=Debug

# Internal: copy each executable + the MLX Metal bundle out of DerivedData.
# CONFIG must be Debug or Release.
_copy-binaries:
	@mkdir -p $(BIN_DIR)
	@PRODUCT_DIR=$$(find $(DERIVED_DATA)/Flux2Swift-*/Build/Products/$(CONFIG) -maxdepth 1 -type d 2>/dev/null | head -1); \
	if [ -z "$$PRODUCT_DIR" ]; then \
		echo "Error: Could not find $(CONFIG) build products in DerivedData"; \
		exit 1; \
	fi; \
	for bin in $(BINARIES); do \
		if [ -f "$$PRODUCT_DIR/$$bin" ]; then \
			cp "$$PRODUCT_DIR/$$bin" $(BIN_DIR)/; \
			echo "  Installed $$bin ($(CONFIG))"; \
		else \
			echo "  Warning: $$bin not found in $$PRODUCT_DIR"; \
		fi; \
	done; \
	if [ -d "$$PRODUCT_DIR/mlx-swift_Cmlx.bundle" ]; then \
		rm -rf $(BIN_DIR)/mlx-swift_Cmlx.bundle; \
		cp -R "$$PRODUCT_DIR/mlx-swift_Cmlx.bundle" $(BIN_DIR)/; \
		echo "  Installed mlx-swift_Cmlx.bundle ($(CONFIG))"; \
	else \
		echo "  Warning: Metal bundle not found — binaries may not work"; \
	fi

# CI-safe test targets (no GPU, no model downloads). These mirror the two
# required status checks defined in TESTING_REQUIREMENTS.md §2.
test-fte: resolve
	xcodebuild test \
		-scheme $(PACKAGE_SCHEME) \
		-destination '$(DESTINATION_MAC)' \
		$(XCODEBUILD_FLAGS) \
		-only-testing FluxTextEncodersTests

test-core: resolve
	xcodebuild test \
		-scheme $(PACKAGE_SCHEME) \
		-destination '$(DESTINATION_MAC)' \
		$(XCODEBUILD_FLAGS) \
		-only-testing Flux2CoreTests

# GPU tests — local only. Requires Apple Silicon + 16 GB RAM + downloaded models.
# Will fail on CI; do not wire into PR checks.
test-gpu: resolve
	xcodebuild test \
		-scheme $(PACKAGE_SCHEME) \
		-destination '$(DESTINATION_MAC)' \
		$(XCODEBUILD_FLAGS) \
		-only-testing Flux2GPUTests

# Run the two CI-required test suites.
test: test-fte test-core
	@echo "All CI-safe tests complete."

# Lint: format Swift sources in place with swift-format and surface any
# remaining diagnostics. There is no `.swift-format` config in this repo,
# so default rules apply — first run will produce a large diff.
LINT_PATHS = Sources Tests
lint:
	@echo "Formatting Swift sources in: $(LINT_PATHS)"
	swift format -i -r $(LINT_PATHS)
	@echo "Linting (diagnostics only)..."
	@swift format lint -r $(LINT_PATHS) || \
		echo "  (swift format lint reported issues; some rules can't be auto-fixed)"
	@echo "Done."

# Non-mutating check: report issues without rewriting files. CI-friendly.
lint-check:
	swift format lint -r $(LINT_PATHS)

# Clean build artifacts, resolved deps, and DerivedData for this project.
clean:
	rm -rf $(BIN_DIR)
	rm -rf $(SPM_DIR)
	rm -rf $(DERIVED_DATA)/Flux2Swift-*
	@echo "Cleaned bin, .spm, and Flux2Swift-* DerivedData."

help:
	@echo "flux-2-swift-mlx Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Build targets:"
	@echo "  resolve     - Resolve SPM package dependencies"
	@echo "  build       - Debug build for macOS (arm64, no copy)"
	@echo "  build-ios   - Build for iOS Simulator (iPhone 17, OS 26.1)"
	@echo "  install     - Debug build + copy CLIs/Metal bundle to ./bin (default)"
	@echo "  release     - Release build + copy CLIs/Metal bundle to ./bin"
	@echo ""
	@echo "Test targets:"
	@echo "  test        - Run CI-safe tests (FluxTextEncoders + Flux2Core)"
	@echo "  test-fte    - Run FluxTextEncodersTests only"
	@echo "  test-core   - Run Flux2CoreTests only"
	@echo "  test-gpu    - Run Flux2GPUTests (local only — needs GPU + models)"
	@echo ""
	@echo "Lint targets:"
	@echo "  lint        - Format Sources/ and Tests/ in place with swift-format"
	@echo "  lint-check  - Report style issues without rewriting (CI-friendly)"
	@echo ""
	@echo "Other:"
	@echo "  clean       - Remove ./bin, .spm, and Flux2Swift-* DerivedData"
	@echo "  help        - Show this help"
	@echo ""
	@echo "macOS destination: $(DESTINATION_MAC)"
	@echo "iOS destination:   $(DESTINATION_IOS)"

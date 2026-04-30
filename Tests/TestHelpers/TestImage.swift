// TestImage.swift - In-memory CGImage factory for tests
// No bundled resources, no disk I/O.

import CoreGraphics

/// Pure in-memory test image factory.
public enum TestImage {

    /// Create a solid-color CGImage entirely in memory.
    /// - Parameters:
    ///   - width: Pixel width (default 64).
    ///   - height: Pixel height (default 64).
    /// - Returns: A valid `CGImage`.
    public static func make(width: Int = 64, height: Int = 64) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
            .union(.byteOrder32Big)

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            fatalError("TestImage: failed to create CGContext (\(width)x\(height))")
        }

        // Fill with a recognisable mid-grey so the image is not blank.
        context.setFillColor(gray: 0.5, alpha: 1.0)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        guard let image = context.makeImage() else {
            fatalError("TestImage: CGContext.makeImage() returned nil")
        }
        return image
    }
}

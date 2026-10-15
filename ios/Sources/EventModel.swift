import Foundation

/// Which exact model produced an event, as the backend stores it per event
/// (cloud/app.py's `_model_identity`). Foundation-only so it is testable outside
/// Xcode (ios/Tests/EventModelTests.swift).
struct EventModel: Codable, Hashable {
    struct Detector: Codable, Hashable {
        /// The file that actually executed: the .hef on the accelerator, not the
        /// .pt name that was only a cache key.
        let name: String
        /// sha256 of the model file as loaded; identifies it exactly.
        let sha256: String?
        /// "hailo" or "ultralytics": what executed it.
        let backend: String?
    }

    /// The vision module, e.g. "driver_attention".
    let module: String?
    /// Nil for a module that runs no object detector.
    let detector: Detector?

    /// "damoyolo_tinynasL20_T.hef · e69136d3ebf3" — short enough for a row.
    var summary: String? {
        guard let detector else { return module }
        guard let sha = detector.sha256, !sha.isEmpty else { return detector.name }
        return "\(detector.name) · \(sha.prefix(12))"
    }
}

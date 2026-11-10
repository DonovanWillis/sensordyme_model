import Foundation

/// The rules for what can be sent to Jupiter, mirrored from the backend so the
/// Ask Jupiter bar never offers to send something the server will refuse.
///
/// Source of truth: `jupiter/security.py` `sanitise_request` — whitespace is
/// trimmed, an empty request is refused, and a request longer than
/// `MAX_REQUEST_CHARS` (600) is refused outright rather than truncated. Python's
/// `len` counts Unicode code points, so this counts `unicodeScalars`, not
/// grapheme clusters: an emoji family is one Character but several code points.
enum JupiterPrompt {
    static let maxLength = 600
    /// The counter appears as the request approaches the limit, not before —
    /// a number under every one-line request is noise.
    static let countThreshold = 480

    /// The text that will actually be sent, or nil when there is nothing to send.
    static func normalized(_ raw: String) -> String? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Length as the backend measures it (code points, after trimming).
    static func length(_ raw: String) -> Int {
        normalized(raw)?.unicodeScalars.count ?? 0
    }

    static func isOverLimit(_ raw: String) -> Bool {
        length(raw) > maxLength
    }

    static func canSubmit(_ raw: String) -> Bool {
        normalized(raw) != nil && !isOverLimit(raw)
    }

    static func showsCount(_ raw: String) -> Bool {
        length(raw) >= countThreshold
    }
}

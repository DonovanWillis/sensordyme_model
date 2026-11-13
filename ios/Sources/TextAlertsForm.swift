import Foundation

/// The text-message alert form's rules: what the server accepts and whether
/// there is anything to save. Foundation-only and unit-tested
/// (ios/Tests/TextAlertsFormTests.swift).
///
/// The server keeps one number and one switch per account
/// (`POST /v1/account/phone`, cloud/app.py `set_phone`) and cannot send them
/// back (no GET), so the form compares what is typed with what THIS phone last
/// saved, and says so.
enum TextAlertsForm {
    /// The number as it is sent: spaces removed (the server would keep them).
    static func normalized(_ typed: String) -> String {
        typed.trimmingCharacters(in: .whitespacesAndNewlines).replacingOccurrences(of: " ", with: "")
    }

    /// The server's own rule (set_phone): "+" then digits, at least 8
    /// characters in all; empty removes the number.
    static func isValid(_ typed: String) -> Bool {
        let number = normalized(typed)
        guard number.hasPrefix("+"), number.count >= 8 else { return false }
        return number.dropFirst().allSatisfy { $0.isASCII && $0.isNumber }
    }

    /// Why the typed number would be refused, in words; nil when it is fine
    /// (or empty: nothing typed yet).
    static func problem(_ typed: String) -> String? {
        let number = normalized(typed)
        if number.isEmpty || isValid(number) { return nil }
        if !number.hasPrefix("+") { return "Start with + and the country code, e.g. +1 555 123 4567." }
        return "Use digits only after the +, at least 7 of them."
    }

    /// Whether Save would change anything the server holds (as far as this
    /// phone knows) and would be accepted.
    static func canSave(phone: String, textOnSerious: Bool, savedPhone: String, savedTextOnSerious: Bool) -> Bool {
        let number = normalized(phone)
        guard !number.isEmpty, isValid(number) else { return false }
        return number != normalized(savedPhone) || textOnSerious != savedTextOnSerious
    }
}

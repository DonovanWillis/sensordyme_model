import Foundation

/// Deleting the account (M8.4, `POST /v1/account/delete {password, confirm}`),
/// as pure decisions: what is and is not deleted, when the form may be sent,
/// and what a refusal says. Foundation-only (ios/Tests/AccountDeletionTests.swift).
///
/// The wording follows the backend and docs/PRIVACY.md §5 exactly: what the
/// server deletes, what it releases, and what it cannot reach (the copies kept
/// on the device itself).
enum AccountDeletion {
    static let deleted = [
        "Your account: email, password, phone number and push alert channel",
        "Every event your devices recorded on the server",
        "Your order requests",
        "Every signed-in session, on this phone and any other",
    ]
    static let released = [
        "Your devices are released from the account: their names, detection areas, custom rules and "
            + "pending changes are cleared, and camera-only mode is turned off, so they can be added to "
            + "another account",
    ]
    static let notDeleted = [
        "Copies kept on each device itself (its event log, saved rules, areas and camera setting) "
            + "aren't wiped remotely. Reset the device to remove them.",
        "Backups of the server's database made before the deletion.",
        "Alerts already delivered to your phone through ntfy or by text message.",
    ]

    static let confirmTitle = "Delete your account?"
    static let confirmMessage = "This can't be undone. Your devices stay on and keep their own copies until they are reset."
    static let confirmButton = "Delete account"

    /// The form may be sent: a password was entered and nothing is in flight.
    static func canSubmit(password: String, deleting: Bool) -> Bool {
        !password.isEmpty && !deleting
    }

    /// A refusal, in the owner's terms. Every one says nothing was deleted.
    static func message(status: Int, code: String?, detail: String) -> String {
        switch code {
        case "wrong_password": return "That password isn't right. Nothing was deleted."
        case "confirmation_required": return "Deleting needs your confirmation. Nothing was deleted."
        default: break
        }
        switch status {
        case 0: return "Couldn't reach the server, so nothing was deleted. \(detail)"
        case 404 where code == nil: return "This server can't delete accounts yet. Nothing was deleted."
        case 429: return "Too many tries. Wait a few minutes. Nothing was deleted."
        default: return "Nothing was deleted: \(detail)"
        }
    }

    /// What the server reported deleting, for the line shown after sign-out.
    static func summary(devicesReleased: Int?, eventsDeleted: Int?) -> String {
        var parts = ["Your account was deleted."]
        if let eventsDeleted { parts.append("\(eventsDeleted) \(eventsDeleted == 1 ? "event" : "events") deleted.") }
        if let devicesReleased {
            parts.append("\(devicesReleased) \(devicesReleased == 1 ? "device" : "devices") released.")
        }
        return parts.joined(separator: " ")
    }
}

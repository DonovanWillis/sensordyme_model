import Foundation

/// Reduce Motion: decorative motion that repeats on its own (a pulsing dot, a
/// breathing placeholder feed, thinking dots) stops. State changes still show;
/// only the endless loops go.
enum MotionPolicy {
    static func animates(_ requested: Bool, reduceMotion: Bool) -> Bool {
        requested && !reduceMotion
    }
}

/// What a person is told when the server answered but its reply could not be
/// read. Before this the raw Foundation text reached the screen — "The data
/// couldn't be read because it isn't in the correct format." — which says
/// nothing about what happened or what to do.
enum ReplyError {
    static let unreadable = "The server answered, but this app couldn't read its reply — it may be running a "
        + "different version. Nothing was changed. Pull down to try again; if it keeps happening, check "
        + "Settings → Server."

    /// The message for a decoding failure; nil for any other error, which keeps its own.
    static func message(for error: Error) -> String? {
        error is DecodingError ? unreadable : nil
    }

    /// The banner's headline for a failed refresh. "Can't reach the server"
    /// was shown for every failure, including a reply the server DID send but
    /// this build could not read — contradicting the body underneath.
    static func bannerTitle(for message: String) -> String {
        message == unreadable ? "The server's reply couldn't be read" : "Can't reach the server"
    }
}

/// Spoken descriptions where the screen relies on colour.
enum EventAccessibility {
    /// The event dot's colour, in words: loud = red, soft = amber.
    static func severity(_ severity: String?) -> String {
        switch severity {
        case "loud": "Urgent alert"
        case "soft": "Alert"
        case "experimental": "Experimental observation, not an alert"
        default: "Event"
        }
    }

    static func label(title: String, severity: String?, signals: String?, when: String?) -> String {
        var parts = ["\(self.severity(severity)): \(title)"]
        if let signals, !signals.isEmpty { parts.append(signals) }
        if let when, !when.isEmpty { parts.append(when) }
        return parts.joined(separator: ", ")
    }
}

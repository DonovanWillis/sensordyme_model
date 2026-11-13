import Foundation

// Run with ios/Tests/run.sh

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

print("Reduce Motion stops repeating decoration:")
check(MotionPolicy.animates(true, reduceMotion: false), "a pulse runs normally")
check(!MotionPolicy.animates(true, reduceMotion: true), "no pulse under Reduce Motion")
check(!MotionPolicy.animates(false, reduceMotion: false), "nothing animates that did not ask to")

print("\nan unreadable reply gets an actionable message:")
do {
    struct Devices: Decodable { let devices: [String] }
    var thrown: Error?
    do { _ = try JSONDecoder().decode(Devices.self, from: Data("{\"devices\": [ this is not JSON".utf8)) } catch { thrown = error }
    let message = thrown.flatMap(ReplyError.message(for:))
    check(message == ReplyError.unreadable, "invalid JSON maps to the unreadable-reply message")
    check(message?.contains("Nothing was changed") == true, "it says nothing changed")
    check(message?.contains("Settings → Server") == true, "it says where to look")
    check(!(message ?? "").contains("correct format"), "the raw Foundation text is gone")
    do { _ = try JSONDecoder().decode(Devices.self, from: Data("{\"other\": 1}".utf8)) } catch { thrown = error }
    check(thrown.flatMap(ReplyError.message(for:)) == ReplyError.unreadable, "a missing key is unreadable too")
    check(ReplyError.message(for: URLError(.cannotConnectToHost)) == nil, "a network error keeps its own message")
}

print("\nan event's colour is also said:")
check(EventAccessibility.severity("loud") == "Urgent alert", "loud is urgent")
check(EventAccessibility.severity("soft") == "Alert", "soft is an alert")
check(EventAccessibility.severity(nil) == "Event", "no severity is an event")
check(EventAccessibility.severity("experimental") == "Experimental observation, not an alert",
      "an experimental observation is said not to be an alert (M8.4)")
check(EventAccessibility.label(title: "Drowsiness", severity: "loud", signals: "eyes closed 2.1 s", when: "2 min ago")
      == "Urgent alert: Drowsiness, eyes closed 2.1 s, 2 min ago", "the full label")
check(EventAccessibility.label(title: "Phone Use", severity: "soft", signals: "", when: nil)
      == "Alert: Phone Use", "empty parts are left out")

print("\nthe banner's headline agrees with its body:")
check(ReplyError.bannerTitle(for: ReplyError.unreadable) == "The server's reply couldn't be read",
      "an unreadable reply does not say the server is unreachable")
check(ReplyError.bannerTitle(for: "No network connection. Check the phone's Wi-Fi.") == "Can't reach the server",
      "a network failure still says so")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)

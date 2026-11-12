import Foundation
// run.sh deps: DevicePresence

// Run with ios/Tests/run.sh. Presence comes from the device's telemetry and the
// age of the reply carrying it, never from the backend merely answering.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

let now = Date(timeIntervalSince1970: 2_000_000)
func reading(online: Bool, stale: Bool? = false, seenAgo: TimeInterval? = 5, receivedAgo: TimeInterval? = 2)
    -> ConnectionTruth.Reading {
    .init(online: online, stale: stale,
          lastSeen: seenAgo.map { now.timeIntervalSince1970 - $0 },
          receivedAt: receivedAgo.map { now.addingTimeInterval(-$0) })
}
func presence(_ r: ConnectionTruth.Reading?, failed: Bool = false) -> ConnectionTruth.Presence {
    ConnectionTruth.presence(r, lastRefreshFailed: failed, now: now)
}

print("fresh telemetry in a fresh reply is online:")
check(presence(reading(online: true)) == .online, "reported 5 s ago, reply 2 s old")
check(presence(reading(online: true, stale: nil)) == .online, "a backend without status.stale still works")
check(presence(reading(online: true, seenAgo: -20)) == .online,
      "the phone's clock behind the backend's reads as just now, not stale")
check(presence(reading(online: true, seenAgo: 55)) == .online,
      "within the clock tolerance a live device does not flicker to stale")

print("\nold telemetry is stale, whatever `online` says:")
check(presence(reading(online: true, seenAgo: 61)) == .stale(age: 61),
      "a reply saying online with last_seen 61 s ago (a replayed or cached reply) is stale")
check(presence(reading(online: true, seenAgo: 600)) == .stale(age: 600), "ten minutes old is stale")
check(presence(reading(online: true, stale: true, seenAgo: 31)) == .stale(age: 31),
      "status.stale wins over the online flag (the pills can't disagree)")
check(presence(reading(online: true, stale: true, seenAgo: nil)) == .stale(age: ConnectionTruth.liveWindow),
      "stale with no last_seen still says stale")
check(!presence(reading(online: true, seenAgo: 61)).isOnline, "stale is never online")

print("\nthe backend's offline verdict says since when:")
check(presence(reading(online: false, seenAgo: 120)) == .offline(age: 120), "offline two minutes")
check(presence(reading(online: false, seenAgo: nil)) == .offline(age: nil), "never reported")

print("\nthe reply's own age and the refresh outcome come first:")
check(presence(reading(online: true), failed: true) == .unknown(.refreshFailed),
      "a failed refresh never leaves ONLINE standing")
check(presence(reading(online: false), failed: true) == .unknown(.refreshFailed), "nor OFFLINE")
check(presence(reading(online: true, receivedAgo: 31)) == .unknown(.notRefreshed(age: 31)),
      "a reply 31 s old (polling paused, request hanging) no longer speaks for now")
check(presence(reading(online: true, receivedAgo: 29)) == .online, "a reply within the window still does")
check(presence(reading(online: true, receivedAgo: nil)) == .online,
      "a value that was never fetched is judged on its flags alone")
check(presence(nil) == .unknown(.notLoaded), "nothing loaded is unknown")

print("\nthe summary count follows the same rule as the cards:")
let fleet = [reading(online: true), reading(online: true, seenAgo: 90), reading(online: false, seenAgo: 300)]
check(ConnectionTruth.onlineCount(fleet, lastRefreshFailed: false, now: now) == 1,
      "one live, one stale, one offline counts one online (was 2: it counted `online` flags)")
check(ConnectionTruth.onlineCount(fleet, lastRefreshFailed: true, now: now) == 0,
      "after a failed refresh nothing is counted online")

print("\nwords:")
check(ConnectionTruth.label(.online) == "Online", "Online")
check(ConnectionTruth.label(.stale(age: 90)) == "Stale", "Stale is its own word, not Online or Offline")
check(ConnectionTruth.label(.unknown(.refreshFailed)) == "Unknown", "Unknown")
check(ConnectionTruth.cardLine(.online) == nil, "an online card names its mode instead")
check(ConnectionTruth.cardLine(.stale(age: 90)) == "No report for 1 minute — showing its last report",
      "the stale card line says how old and that it is the last report")
check(ConnectionTruth.cardLine(.offline(age: 7200)) == "Last seen 2 hours ago", "offline says since when")
check(ConnectionTruth.cardLine(.offline(age: nil)) == "Not seen yet", "never seen")
check(ConnectionTruth.cardLine(.unknown(.notRefreshed(age: 45))) == "Not updated for 45 seconds",
      "a reply that is too old says so")
check(ConnectionTruth.spoken(.stale(age: 90)).hasPrefix("Stale data"), "VoiceOver hears 'stale data'")
check(ConnectionTruth.spoken(.offline(age: 60)).contains("last seen"), "VoiceOver hears since when")
check(ConnectionTruth.notReportingMessage(.online) == nil, "no banner while online")
check(ConnectionTruth.notReportingMessage(.unknown(.refreshFailed)) == nil,
      "a failed refresh has its own banner (the server's), not this one")
check(ConnectionTruth.notReportingMessage(.stale(age: 120))?.contains("2 minutes old") == true,
      "the stale banner says how old the report is")
check(ConnectionTruth.notReportingMessage(.offline(age: 60))?.contains("apply when it reconnects") == true,
      "offline says changes wait for the device")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)

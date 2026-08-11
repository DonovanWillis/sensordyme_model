import Foundation

// Run with ios/Tests/run.sh

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

print("a single request is always latest:")
do {
    let g = LatestRequestGuard()
    let token = g.begin()
    check(g.isLatest(token), "the only request issued is latest")
}

print("\na second request supersedes the first:")
do {
    let g = LatestRequestGuard()
    let first = g.begin()
    let second = g.begin()
    check(!g.isLatest(first), "the first request is no longer latest")
    check(g.isLatest(second), "the second request is latest")
}

print("\nthe classic race — slow request finishes after a fast newer one:")
do {
    let g = LatestRequestGuard()
    // Simulates: reload() #1 starts (slow network)...
    let slow = g.begin()
    // ...then reload() #2 starts and finishes first (fast network)...
    let fast = g.begin()
    check(g.isLatest(fast), "the fast, later-started request applies its result")
    // ...then reload() #1's response finally arrives.
    check(!g.isLatest(slow), "the slow, earlier-started request's stale result is discarded")
}

print("\nmany overlapping requests — only the last one issued is ever latest:")
do {
    let g = LatestRequestGuard()
    var tokens: [Int] = []
    for _ in 0..<10 {
        tokens.append(g.begin())
    }
    for token in tokens.dropLast() {
        check(!g.isLatest(token), "an earlier request among many is not latest")
    }
    check(g.isLatest(tokens.last!), "only the very last request is latest")
}

print("\nrapid mode-switch taps — a stale failure must not clobber a newer success:")
do {
    // Models setMode()'s own race: tap Driver Safety (fails slowly), then tap
    // Security (succeeds quickly). The Driver Safety failure arriving late
    // must not be shown once Security has already been confirmed.
    let g = LatestRequestGuard()
    let driverSafetyAttempt = g.begin()
    let securityAttempt = g.begin()
    check(g.isLatest(securityAttempt), "the later tap (Security) owns the UI state")
    check(!g.isLatest(driverSafetyAttempt), "the superseded tap's outcome, success or failure, is ignored")
}

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)

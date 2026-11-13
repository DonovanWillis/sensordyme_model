import Foundation

// Run with ios/Tests/run.sh
//
// When the Ask Jupiter dock is the full field and when it is a compact button.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

/// Scroll from `from` to `to` in 4 pt steps, as a drag reports it.
func scroll(_ t: inout JupiterDockBehavior.Tracker, from: Double, to: Double, max: Double = 2000,
            engaged: Bool = false) {
    let step: Double = from < to ? 4 : -4
    var y = from
    while (step > 0 && y < to) || (step < 0 && y > to) {
        y += step
        t.scrolled(offset: y, maxOffset: max, engaged: engaged)
    }
}

print("at the top it is the field:")
var t = JupiterDockBehavior.Tracker()
t.scrolled(offset: 0, maxOffset: 2000, engaged: false)
check(!t.minimised, "at rest at the top")
scroll(&t, from: 0, to: -60)
check(!t.minimised, "in the pull-to-refresh bounce")
scroll(&t, from: -60, to: 50)
check(!t.minimised, "within the top zone, even after scrolling down")

print("\nscrolling down deliberately minimises it:")
t = JupiterDockBehavior.Tracker()
t.scrolled(offset: 0, maxOffset: 2000, engaged: false)
scroll(&t, from: 0, to: 100)
check(!t.minimised, "100 pt from the top: only 44 pt travelled past the top zone… not yet")
scroll(&t, from: 100, to: 140)
check(t.minimised, "48 pt of downward travel past the top: minimised")

print("\na small jitter does not flip it:")
scroll(&t, from: 140, to: 124)
check(t.minimised, "16 pt back up keeps it minimised")
scroll(&t, from: 124, to: 140)
check(t.minimised, "and down again keeps it minimised")

print("\nscrolling up brings it back:")
scroll(&t, from: 140, to: 108)
check(!t.minimised, "28 pt up restores the field")
scroll(&t, from: 108, to: 130)
check(!t.minimised, "a short scroll down after that does not minimise again")
scroll(&t, from: 130, to: 200)
check(t.minimised, "a deliberate one does")

print("\nthe end of the page brings it back:")
scroll(&t, from: 200, to: 1992, max: 2000)
check(!t.minimised, "within 12 pt of the end")

print("\nnever while it is in use:")
t = JupiterDockBehavior.Tracker()
t.scrolled(offset: 0, maxOffset: 2000, engaged: false)
scroll(&t, from: 0, to: 600, engaged: true)
check(!t.minimised, "focused or holding a typed request: stays the field")
scroll(&t, from: 600, to: 700)
check(t.minimised, "released: a deliberate scroll minimises it again")
t.scrolled(offset: 700, maxOffset: 2000, engaged: true)
check(!t.minimised, "focusing restores it at once")

print("\na tap on the compact dock restores it:")
t = JupiterDockBehavior.Tracker()
t.scrolled(offset: 0, maxOffset: 2000, engaged: false)
scroll(&t, from: 0, to: 400)
check(t.minimised, "fixture: minimised")
t.restore()
check(!t.minimised, "restored")
scroll(&t, from: 400, to: 420)
check(!t.minimised, "and stays restored until a deliberate scroll")

print("\nthe change is reported once:")
t = JupiterDockBehavior.Tracker()
t.scrolled(offset: 0, maxOffset: 2000, engaged: false)
var changes = 0
var y: Double = 0
while y < 400 { y += 4; if t.scrolled(offset: y, maxOffset: 2000, engaged: false) { changes += 1 } }
check(changes == 1, "one change across a long scroll (\(changes))")

print("\nform:")
check(JupiterDockBehavior.form(accessibilitySize: false, minimised: false) == .field, "default size, at rest: field")
check(JupiterDockBehavior.form(accessibilitySize: false, minimised: true) == .button, "default size, minimised: button")
check(JupiterDockBehavior.form(accessibilitySize: true, minimised: false) == .button, "accessibility size: always the button")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)

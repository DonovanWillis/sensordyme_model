import Foundation

// Run with ios/Tests/run.sh
// run.sh deps: BackendResolution
//
// A redirect never carries a sign-in to another host or out of https.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

func allows(_ from: String, _ to: String) -> Bool {
    RedirectPolicy.allows(from: URL(string: from), to: URL(string: to))
}

print("same host is followed:")
check(allows("https://api.example.com/v1/login", "https://api.example.com/v1/login/"), "https, same host")
check(allows("http://192.168.1.50:8787/v1/devices", "http://192.168.1.50:8787/v1/devices/"), "LAN http, same host")
check(allows("http://192.168.1.50:8787/v1/x", "https://192.168.1.50/v1/x"), "an upgrade to https on the same host")
check(allows("https://API.example.com/v1/x", "https://api.example.com/v1/y"), "host compared without case")

print("\nanother host is refused:")
check(!allows("https://api.example.com/v1/login", "https://evil.example.net/v1/login"), "https to another host")
check(!allows("http://192.168.1.50:8787/v1/login", "http://192.168.1.51:8787/v1/login"), "LAN to another LAN host")
check(!allows("https://api.example.com/v1/x", "https://api.example.com.evil.net/v1/x"), "a lookalike suffix")

print("\na downgrade is refused:")
check(!allows("https://api.example.com/v1/login", "http://api.example.com/v1/login"), "https to http, same host")
check(!allows("https://192.168.1.50/v1/login", "http://192.168.1.50:8787/v1/login"), "even on a LAN host")

print("\nthe target must itself be allowed:")
check(!allows("http://localhost:8787/v1/x", "ftp://localhost/v1/x"), "not http(s)")
check(!allows("https://api.example.com/v1/x", "/relative"), "no host")
check(!RedirectPolicy.allows(from: nil, to: URL(string: "https://api.example.com")), "no original URL")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)

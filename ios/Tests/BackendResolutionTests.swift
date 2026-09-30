import Foundation

// Standalone, runnable with `swift ios/Tests/BackendResolutionTests.swift` via
// ios/Tests/run.sh (which concatenates the subject file with this one).
//
// What this pins down is the failure that took the app off the air after a move
// between Wi-Fi networks: sign-in trying to reach `http://11.33.6.240:8787`, an
// address nobody typed, which the app had discovered over mDNS on a network it
// was no longer on and stored as if it were configuration.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

// The real value, from the phone that failed. Note it is NOT an RFC 1918
// address — 11/8 is public space being handed out by a local router — which is
// exactly why the classification here is "is it an address" and not "is it private".
let staleLAN = "http://11.33.6.240:8787"
let production = "https://api.example-placeholder.invalid"
let devMac = "http://donovans-macbook-pro-2.local:8787"

// ------------------------------------------------------------- precedence
print("precedence — configured, then the build's default, then discovery:")
check(BackendResolution.effective(configured: production, productionDefault: "https://built-in.invalid", discovered: staleLAN)
      == production,
      "a typed address outranks everything")
check(BackendResolution.effective(configured: "", productionDefault: "https://built-in.invalid", discovered: staleLAN)
      == "https://built-in.invalid",
      "the build's production URL outranks a discovered one")
check(BackendResolution.effective(configured: "", productionDefault: "", discovered: staleLAN)
      == staleLAN,
      "discovery is used when nothing else is known")
check(BackendResolution.effective(configured: "", productionDefault: "", discovered: "") == "",
      "nothing known is reported as nothing, not as a bogus address")
check(BackendResolution.effective(configured: "   ", productionDefault: "", discovered: staleLAN) == staleLAN,
      "whitespace is not a configuration")

// --------------------------------------------- discovery cannot override config
print("\ndiscovery can never replace a pinned backend:")
check(BackendResolution.mayFallBackToDiscovery(configured: production, productionDefault: "") == false,
      "a typed public backend is never abandoned for the local Wi-Fi")
check(BackendResolution.mayFallBackToDiscovery(configured: "", productionDefault: production) == false,
      "nor is the build's production backend")
check(BackendResolution.mayFallBackToDiscovery(configured: "", productionDefault: "") == true,
      "with nothing pinned, searching is the only way to find anything")
check(BackendResolution.mayFallBackToDiscovery(configured: devMac, productionDefault: production) == true,
      "a developer pointed at a Mac on the LAN still gets self-healing discovery")
check(BackendResolution.mayFallBackToDiscovery(configured: "http://192.168.1.20:8787", productionDefault: "") == true,
      "and so does one pointed at a LAN IP")
// The regression this prevents: an internet blip on a phone with a real backend
// configured would otherwise send it looking for a server on the local network
// and sign the user in against a completely different one.

// ------------------------------------------------------------- migration
print("\nmigrating the single old `baseURL` slot:")
check(BackendResolution.migrate(legacy: staleLAN) == .discard,
      "the stale discovered IP is discarded, so the phone stops trying to reach it")
check(BackendResolution.migrate(legacy: "http://192.168.1.20:8787") == .discard,
      "any bare IPv4 is treated as a discovery artifact")
check(BackendResolution.migrate(legacy: "http://[fe80::1%en0]:8787") == .discard,
      "so is a scoped IPv6 literal")
check(BackendResolution.migrate(legacy: "http://localhost:8787") == .discard,
      "localhost is dropped — on a phone it means the phone itself")
check(BackendResolution.migrate(legacy: "http://127.0.0.1:8787") == .discard, "as is loopback by address")
check(BackendResolution.migrate(legacy: "") == .discard, "an empty value carries nothing forward")
check(BackendResolution.migrate(legacy: "not a url at all") == .discard, "neither does an unparseable one")
check(BackendResolution.migrate(legacy: production) == .promoteToConfigured(production),
      "a real hostname was a decision and is kept, as configuration")
check(BackendResolution.migrate(legacy: devMac) == .promoteToConfigured(devMac),
      "a .local name was typed by a person, so it survives the upgrade too")

// ------------------------------------------------ the build-time production URL
print("\nreading SDBackendBaseURL out of Info.plist:")
check(BackendResolution.productionDefault(fromInfoPlist: nil) == "", "absent means unconfigured")
check(BackendResolution.productionDefault(fromInfoPlist: "") == "", "empty means unconfigured")
check(BackendResolution.productionDefault(fromInfoPlist: "$(SENSORDYME_API_BASE_URL)") == "",
      "an unsubstituted build variable is unconfigured, not a hostname to dial")
check(BackendResolution.productionDefault(fromInfoPlist: "api.example.invalid") == "",
      "a bare host with no scheme is rejected rather than half-used")
check(BackendResolution.productionDefault(fromInfoPlist: " \(production) ") == production,
      "a real https URL is accepted, trimmed")
check(BackendResolution.productionDefault(fromInfoPlist: "http://dev-box.invalid:8787") == "http://dev-box.invalid:8787",
      "http is allowed, so a dev build can point at a plain-HTTP server")

// ------------------------------------------------------- host classification
print("\nLAN-scoped or not:")
check(BackendResolution.isLANScoped(staleLAN), "a bare IP is local-only")
check(BackendResolution.isLANScoped(devMac), "a .local name is local-only")
check(BackendResolution.isLANScoped("http://localhost:8787"), "loopback is local-only")
check(BackendResolution.isLANScoped(production) == false, "a real hostname is not")
check(BackendResolution.isLANScoped("11.33.6.240"), "a bare host works as well as a full URL")
check(BackendResolution.isIPv4Literal("11.33.6.240"), "recognises the address that caused all this")
check(BackendResolution.isIPv4Literal("999.1.1.1") == false, "an out-of-range octet is not an address")
check(BackendResolution.isIPv4Literal("1.2.3") == false, "nor are three octets")
check(BackendResolution.isIPv4Literal("api.sensor.example") == false, "nor is a hostname")

// --------------------------------------------------- what the error message says
print("\nerror advice matches the kind of host that failed:")
check(BackendResolution.unreachableAdvice(host: "11.33.6.240").contains("same Wi-Fi"),
      "a local address is explained as a local address")
check(BackendResolution.unreachableAdvice(host: "api.example-placeholder.invalid").contains("same Wi-Fi") == false,
      "a public host does NOT tell the user to check they share Wi-Fi with the server")
check(BackendResolution.unreachableAdvice(host: "api.example-placeholder.invalid").contains("internet"),
      "it points at the internet connection instead")
// The old message said "make sure the phone is on the same Wi-Fi as the server"
// unconditionally, which for a hosted backend describes a requirement the
// product does not have and sends people to fix a working network.

// ------------------------------------------------- what to actually try, in order
print("\nthe candidate list is ordered by preference but gated by reachability:")
check(BackendResolution.candidates(configured: "", productionDefault: "", discovered: "", allowingLoopback: false)
      == [],
      "nothing known produces nothing to try, not a bogus address")
check(BackendResolution.candidates(configured: "", productionDefault: staleLAN, discovered: devMac, allowingLoopback: false)
      == [staleLAN, devMac],
      "a stale build-time LAN address is tried first but the discovered one is still THERE to fall through to")
// This is the specific regression: with only precedence, the build-time
// http://11.22.7.64:8787 was the answer and the healthy discovered Mac was never
// reached, so the app could not connect and could not be pointed anywhere else.
check(BackendResolution.candidates(configured: "", productionDefault: staleLAN, discovered: devMac, allowingLoopback: false).count == 2,
      "a pinned address that has stopped existing does not hide the one that works")
check(BackendResolution.candidates(configured: production, productionDefault: "", discovered: devMac, allowingLoopback: false)
      == [production, devMac],
      "a typed backend is tried before the local network, not instead of it")
check(BackendResolution.candidates(configured: devMac, productionDefault: "", discovered: devMac, allowingLoopback: false)
      == [devMac],
      "the same address discovered and configured is one candidate, not two timeouts")
check(BackendResolution.candidates(configured: "  ", productionDefault: "", discovered: devMac, allowingLoopback: false)
      == [devMac],
      "whitespace is not a candidate")

print("\naddresses a phone cannot use are dropped before they cost a timeout:")
check(BackendResolution.isUsableFromPhone("http://127.0.0.1:8787", allowingLoopback: false) == false,
      "loopback is not a server a phone can reach")
check(BackendResolution.isUsableFromPhone("http://localhost:8787", allowingLoopback: false) == false,
      "nor is localhost by name")
// ...but the simulator shares the Mac's network stack, so there loopback IS the
// development backend. Filtering it out unconditionally broke the one setup
// where pointing at localhost is the correct thing to do.
check(BackendResolution.isUsableFromPhone("http://localhost:8787", allowingLoopback: true),
      "on the simulator, localhost is the dev Mac and is allowed")
check(BackendResolution.candidates(configured: "", productionDefault: "http://localhost:8787", discovered: "",
                                   allowingLoopback: true)
      == ["http://localhost:8787"],
      "so a simulator build still has its default backend to try")
check(BackendResolution.isUsableFromPhone(devMac, allowingLoopback: false),
      "a .local name is fine — that is how the Mac is meant to be found")
check(BackendResolution.isUsableFromPhone(staleLAN, allowingLoopback: false),
      "so is a bare LAN address")
check(BackendResolution.candidates(configured: "", productionDefault: "http://127.0.0.1:8787", discovered: devMac,
                                   allowingLoopback: false)
      == [devMac],
      "on a device, loopback is filtered out rather than tried and timed out")
// macOS publishes 127.0.0.1 in the A record for its own .local hostname
// alongside the real address, so resolving the backend's Bonjour record on a
// phone really can hand back loopback.

// ----------------------------------------------------------------- retry schedule
print("\nreconnection backs off, but stays bounded:")
check(BackendResolution.retryDelay(attempt: 0, jitter: 0.5) == BackendResolution.minimumRetryDelay,
      "the first retry waits the floor, with neutral jitter")
check(BackendResolution.retryDelay(attempt: 3, jitter: 0.5) > BackendResolution.retryDelay(attempt: 1, jitter: 0.5),
      "later attempts wait longer")
check(BackendResolution.retryDelay(attempt: 40, jitter: 0.5) <= BackendResolution.maximumRetryDelay,
      "a phone left disconnected overnight still rechecks within the ceiling")
check(BackendResolution.retryDelay(attempt: 40, jitter: 1.0) <= BackendResolution.maximumRetryDelay * 1.25,
      "jitter cannot push it far past the ceiling")
check(BackendResolution.retryDelay(attempt: 0, jitter: 0) > 0,
      "no jitter value produces a zero delay — that would be a busy loop")
check(BackendResolution.retryDelay(attempt: 5, jitter: -5) > 0,
      "nor does a jitter argument out of range")
check(BackendResolution.retryDelay(attempt: 5, jitter: 0.0) != BackendResolution.retryDelay(attempt: 5, jitter: 1.0),
      "jitter actually varies the delay, so phones do not retry in lockstep")
check(BackendResolution.retryDelay(attempt: -1, jitter: 0.5) == BackendResolution.minimumRetryDelay,
      "a negative attempt count is treated as the first attempt, not as an error")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)

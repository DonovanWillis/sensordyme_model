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

// --------------------------------------- choosing between several backends
print("\nwhich backend, when more than one answers:")
// The health check proves a server is *a* SensorDyme backend, not that it is
// YOURS. On a shared flat, an office or a campus network, taking whichever
// answered first is a coin flip between your Mac and a colleague's.
typealias Found = BackendResolution.FoundBackend
let mine = Found(url: "http://sensordyme-backend.local:8787", id: "bk_mine")
let theirs = Found(url: "http://10.0.0.9:8787", id: "bk_theirs")
let anonymous = Found(url: "http://10.0.0.7:8787", id: nil)

check(BackendResolution.chooseBackend(from: [], remembering: nil) == .none,
      "nothing found is nothing to choose")
check(BackendResolution.chooseBackend(from: [mine], remembering: nil) == .use(mine),
      "one backend and no memory: adopt it — this is what makes first launch need no setup")
check(BackendResolution.chooseBackend(from: [anonymous], remembering: nil) == .use(anonymous),
      "a backend too old to publish an id is still usable when it is the only one")

check(BackendResolution.chooseBackend(from: [theirs, mine], remembering: "bk_mine") == .use(mine),
      "a remembered id wins however the list is ordered")
check(BackendResolution.chooseBackend(from: [mine, theirs], remembering: "bk_theirs") == .use(theirs),
      "and it is the id that decides, not the position")

// The case that matters most.
check(BackendResolution.chooseBackend(from: [mine, theirs], remembering: nil)
      == .ambiguous([mine, theirs]),
      "several backends and no memory: refuse, rather than silently pick a stranger's Mac")
check(BackendResolution.chooseBackend(from: [anonymous, theirs], remembering: nil)
      == .ambiguous([anonymous, theirs]),
      "several that cannot even be told apart is still a refusal, not a free choice")

// Deliberately NOT a rule: "mine is missing, so take another one". A remembered
// backend that has gone quiet is usually a Mac asleep, not a Mac replaced.
check(BackendResolution.chooseBackend(from: [mine, theirs], remembering: "bk_gone")
      == .ambiguous([mine, theirs]),
      "a remembered backend being absent does not license adopting a different one")
check(BackendResolution.chooseBackend(from: [theirs], remembering: "bk_gone") == .ambiguous([theirs]),
      "...nor does a lone stranger: adopting it would hand our token to their server")

check(BackendResolution.chooseBackend(from: [mine], remembering: "") == .use(mine),
      "an empty remembered id is treated as no memory, not as an id to match")

// One backend reaches discovery twice: its stable name and the address that name
// resolves to today. Counting those as two made a fresh install call its only
// backend "ambiguous" and never connect.
let mineByName = Found(url: "http://sensordyme-backend.local:8787", id: "bk_mine")
let mineByAddress = Found(url: "http://11.33.7.22:8787", id: "bk_mine")
check(BackendResolution.chooseBackend(from: [mineByName, mineByAddress], remembering: nil)
      == .use(mineByName),
      "two spellings of one backend are one backend, and the stable name is kept")
let oldByName = Found(url: "http://sensordyme-backend.local:8787", id: nil, service: "srv:A")
let oldByAddress = Found(url: "http://11.33.7.22:8787", id: nil, service: "srv:A")
check(BackendResolution.chooseBackend(from: [oldByName, oldByAddress], remembering: nil)
      == .use(oldByName),
      "without an id, the Bonjour service is what says two URLs are one backend")
check(BackendResolution.chooseBackend(from: [mineByName, mineByAddress, theirs], remembering: nil)
      == .ambiguous([mineByName, theirs]),
      "deduplicating must not turn two real backends into one")


// MARK: - BackendEndpoint: identity versus location
//
// The bug these cover: BackendLocator resolved the Bonjour service by opening a
// connection and reading currentPath.remoteEndpoint -- the ADDRESS the name had
// resolved to -- and persisted that. The backend publishes a stable name
// (sensordyme-backend.local) and a stable id (bk_...) for exactly this reason,
// and the phone was discarding the name and keeping the address.

typealias Endpoint = BackendResolution.BackendEndpoint

let named = Endpoint(host: "sensordyme-backend.local", port: 8787, backendID: "bk_0eb63f4fc338")
let numeric = Endpoint(host: "10.97.118.172", port: 8787, backendID: "bk_0eb63f4fc338")

check(named.baseURL == "http://sensordyme-backend.local:8787", "a name builds the URL it should")
check(named.hostIsStable, "a .local name is resolvable again later, so it is stable")
check(!numeric.hostIsStable, "an address is only true on the network it was read on")
check(named.backendID == numeric.backendID,
      "the same backend at a name and at an address is still ONE backend")

check(Endpoint(host: "sensordyme-backend.local.", port: 8787).host == "sensordyme-backend.local",
      "a trailing dot is valid DNS and breaks URL(string:), so it is stripped")
check(Endpoint(host: "SensorDyme-Backend.LOCAL", port: 8787).host == "sensordyme-backend.local",
      "hosts compare case-insensitively")

check(Endpoint(url: "http://sensordyme-backend.local:8787")?.port == 8787, "the port is parsed")
check(Endpoint(url: "https://api.example.com")?.port == 443, "https defaults to 443")
check(Endpoint(url: "http://sensordyme-backend.local")?.port == 8787,
      "a bare http backend URL defaults to the backend's own port, not 80")
check(Endpoint(url: "not a url") == nil, "garbage is rejected rather than becoming a host")
check(Endpoint(url: "ftp://sensordyme-backend.local:8787") == nil, "only http(s) is a backend")
check(!Endpoint(host: "127.0.0.1", port: 8787).hostIsStable, "loopback is never a stable backend")
check(!Endpoint(host: "localhost", port: 8787).hostIsStable, "...including by name")

// MARK: - What may be written to storage

check(BackendResolution.mayPersistAsLocation(named), "a name is worth remembering")
check(!BackendResolution.mayPersistAsLocation(numeric),
      "an address must never be persisted: that is what made a dead IP authoritative for ever")

// MARK: - Invalidating a stale location, keeping the identity

check(BackendResolution.discoveryCache(
        after: "http://10.97.118.172:8787", keeping: "http://10.97.118.172:8787") == "",
      "the address that just failed is dropped from the cache")
check(BackendResolution.discoveryCache(
        after: "http://10.97.118.172:8787", keeping: "http://sensordyme-backend.local:8787")
      == "http://sensordyme-backend.local:8787",
      "a NAME survives a failure -- a quiet backend is usually a sleeping Mac, not a wrong name")
check(BackendResolution.discoveryCache(after: "", keeping: "http://10.97.118.172:8787") == "",
      "an address with nothing to attribute the failure to is still not worth keeping")
check(BackendResolution.discoveryCache(after: "http://x", keeping: "") == "",
      "an empty cache stays empty")

// MARK: - A name beats an address for the same backend

let ordered = BackendResolution.preferringStableHost([
    numeric,
    Endpoint(host: "fe80::1", port: 8787, backendID: "bk_0eb63f4fc338"),
    named,
])
check(ordered.first == named, "the stable name is tried first, always")
check(ordered.count == 3, "the others are kept as fallbacks, not discarded")
check(!ordered[1].host.contains(":"), "IPv4 before IPv6: a v6 candidate costs a timeout first")

check(BackendResolution.preferringStableHost([]).isEmpty, "no candidates is not a crash")
let twoNames = BackendResolution.preferringStableHost([
    Endpoint(host: "a.local", port: 1), Endpoint(host: "b.local", port: 2),
])
check(twoNames.map(\.host) == ["a.local", "b.local"],
      "equally stable candidates keep discovery order rather than being reshuffled")

// MARK: - The slow-verification race
//
// Candidate A is found, its verification is slow; B is found, verified, and
// becomes active; then A's verification finally returns. A must not win.

check(BackendResolution.verificationIsCurrent(generation: 2, current: 2),
      "a verification from the current generation is acted on")
check(!BackendResolution.verificationIsCurrent(generation: 1, current: 2),
      "a verification from a superseded generation is discarded, not applied")
check(BackendResolution.verificationIsCurrent(generation: 3, current: 2),
      "a newer generation is never rejected as stale")

// MARK: - The scenario from the brief, end to end
//
// The Mac moves from 10.97.118.172 to 10.97.120.45. Identity is unchanged.

do {
    let before = Endpoint(host: "sensordyme-backend.local", port: 8787, backendID: "bk_0eb63f4fc338")
    // Only the name was persisted, so there is no dead address to try.
    check(BackendResolution.mayPersistAsLocation(before), "the name is what got stored")
    let candidates = BackendResolution.candidates(
        configured: "", productionDefault: "", discovered: before.baseURL
    )
    check(candidates == ["http://sensordyme-backend.local:8787"],
          "after the move the app still has exactly one thing to try, and it is the name")
    // mDNS resolves that same name to the new address; identity is unchanged.
    let after = Endpoint(
        host: "sensordyme-backend.local", port: 8787,
        backendID: "bk_0eb63f4fc338", resolvedAddress: "10.97.120.45"
    )
    check(after.baseURL == before.baseURL, "the URL every consumer uses did not change at all")
    check(after.backendID == before.backendID, "and it is recognisably the same backend")
}


// MARK: - Not sending the user to Settings while recovery is still running
//
// The complaint this fixes, verbatim from the app: "Couldn't find a SensorDyme
// server on this network. ... You can also enter the address yourself in
// Settings -> Manual server address." That was shown after the FIRST failed
// resolve, while exponential retry was still running in the background and
// would usually succeed seconds later. The advice arrived ahead of the recovery
// it pre-empted, and taught people SensorDyme needs manual setup when it does not.

let firstFailure = BackendResolution.failureMessage(attempts: 0, triedHost: "10.0.0.5")
check(!firstFailure.contains("Settings"),
      "a first failure must not send the user to Settings -- retry is still running")
check(firstFailure.lowercased().contains("reconnecting"),
      "it should say what is actually happening instead")

check(BackendResolution.isStillRecovering(attempts: 0), "attempt 1 is still recovering")
check(BackendResolution.isStillRecovering(attempts: 1), "so is attempt 2")
check(BackendResolution.isStillRecovering(attempts: 2), "and attempt 3")
check(!BackendResolution.isStillRecovering(
        attempts: BackendResolution.attemptsBeforeSuggestingManualEntry),
      "after the budget is spent the app stops claiming it is about to work")

let givenUp = BackendResolution.failureMessage(
    attempts: BackendResolution.attemptsBeforeSuggestingManualEntry, triedHost: "10.0.0.5")
check(givenUp.contains("Manual server address"),
      "once automatic discovery has genuinely failed, the escape hatch IS the right advice")
check(givenUp.contains("10.0.0.5"), "and it names what was tried")

// A LAN address and a public one need different advice; that distinction already
// existed and must survive being routed through the new message builder.
check(BackendResolution.failureMessage(attempts: 9, triedHost: "10.0.0.5")
        .contains("same Wi-Fi"),
      "a local address means check the network")
check(BackendResolution.failureMessage(attempts: 9, triedHost: "api.example.com")
        .contains("internet connection"),
      "a hosted backend does not: telling someone to check Wi-Fi sends them to fix what works")

// Nothing was even tried -- no candidates at all.
check(!BackendResolution.failureMessage(attempts: 0, triedHost: nil).contains("Settings"),
      "still no Settings nag on the first attempt with nothing cached")
check(BackendResolution.failureMessage(attempts: 0, triedHost: nil).contains("Looking for"),
      "and it reads as a search in progress")
check(BackendResolution.failureMessage(attempts: 9, triedHost: nil).contains("Manual server address"),
      "but a sustained failure with nothing to try still offers the escape")

// Local Network permission is the one cause retrying can never fix, so it is
// said at once rather than after the backoff has run its course.
let denied = "SensorDyme needs Local Network access."
check(BackendResolution.failureMessage(attempts: 0, triedHost: "10.0.0.5",
                                       permissionProblem: denied) == denied,
      "a permission problem is reported immediately, not after three attempts")
check(BackendResolution.failureMessage(attempts: 0, triedHost: "10.0.0.5",
                                       permissionProblem: "") != denied,
      "an empty problem string is not a problem")

// MARK: - Where an endpoint came from is a fact worth carrying

check(BackendResolution.ResolutionSource.cached.rawValue == "cached", "loggable")
check(BackendResolution.ResolutionSource.bonjour.rawValue == "bonjour", "loggable")
check(BackendResolution.ResolutionSource.manual.rawValue == "manual", "loggable")
check(BackendResolution.ResolutionSource.cached != .bonjour,
      "\"the cache answered\" and \"Bonjour just found it\" are the same URL and different facts")

// MARK: - Where credentials may travel without encryption

print("\ncleartext is for the local network only:")
check(BackendResolution.cleartextPermitted("https://api.example.com"), "https anywhere")
check(BackendResolution.cleartextPermitted("http://11.33.6.234:8787"),
      "a LAN IP literal, even in public-range space (ResNet hands out 11/8)")
check(BackendResolution.cleartextPermitted("http://192.168.1.50:8787"), "a private IP")
check(BackendResolution.cleartextPermitted("http://[fe80::1%25en0]:8787"), "a link-local IPv6 literal")
check(BackendResolution.cleartextPermitted("http://sensordyme-backend.local:8787"), "an mDNS name")
check(BackendResolution.cleartextPermitted("http://localhost:8787"), "loopback")
check(!BackendResolution.cleartextPermitted("http://raspberrypi:8899"),
      "NOT a single-label name: a DNS search domain can resolve it to the internet")
check(BackendResolution.cleartextPermitted("http://raspberrypi.local:8899"), "its .local name instead")
check(!BackendResolution.cleartextPermitted("http://user@evil.com"), "NOT userinfo smuggling an internet host")
check(!BackendResolution.cleartextPermitted("http://example.com."), "NOT a trailing-dot internet name")
check(BackendResolution.cleartextPermitted("HTTP://11.33.6.234:8787"), "scheme case does not matter")
check(!BackendResolution.cleartextPermitted("http://api.example.com"), "NOT an internet DNS name over http")
check(!BackendResolution.cleartextPermitted("http://sensordyme.fly.dev:8787"), "NOT a deployed backend over http")
check(!BackendResolution.cleartextPermitted("ftp://192.168.1.50"), "NOT another scheme")
check(!BackendResolution.cleartextPermitted("not a url"), "NOT garbage")

// MARK: - The health check confirms WHICH backend, not just that one answered

print("\na cached name or address must lead back to the backend that was pinned:")
check(!BackendResolution.healthIdentityMatches(reported: "bk_theirs", expected: "bk_mine"),
      "a different backend at the shared sensordyme-backend.local name is refused")
check(BackendResolution.healthIdentityMatches(reported: "bk_mine", expected: "bk_mine"),
      "our own backend is accepted")
check(BackendResolution.healthIdentityMatches(reported: "bk_theirs", expected: ""),
      "with nothing pinned yet there is nothing to contradict")
check(BackendResolution.healthIdentityMatches(reported: "bk_theirs", expected: nil),
      "a typed address carries no expectation: it is the user's explicit choice")
check(BackendResolution.healthIdentityMatches(reported: nil, expected: "bk_mine"),
      "a backend that predates the id field is not stranded")
check(BackendResolution.healthIdentityMatches(reported: "", expected: "bk_mine"),
      "an empty id is treated as no id")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)

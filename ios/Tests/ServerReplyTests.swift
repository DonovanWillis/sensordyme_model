import Foundation
// run.sh deps: AccessibilityPolicy

// Run with ios/Tests/run.sh. Whatever a server sends back, the person reads a
// sentence: never raw JSON or HTML, never a bare "request failed (502)".

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}
func body(_ text: String) -> Data { Data(text.utf8) }
func clean(_ message: String) -> Bool {
    !message.contains("<") && !message.contains("{") && !message.contains("request failed")
}

print("what the body is, sniffed:")
check(ServerReply.kind(of: Data()) == .empty, "no bytes")
check(ServerReply.kind(of: body("  \n")) == .empty, "whitespace only")
check(ServerReply.kind(of: body("{\"a\":1}")) == .json, "an object")
check(ServerReply.kind(of: body("\n<!DOCTYPE html><html>")) == .html, "a web page after a newline")
check(ServerReply.kind(of: body("Internal Server Error")) == .text, "plain text")
check(ServerReply.kind(of: Data([0xff, 0xfe, 0x00, 0x80])) == .binary, "not UTF-8")

print("\nerror replies keep the server's sentence:")
do {
    let coded = ServerReply.httpFailure(status: 409, data: body(
        "{\"detail\": \"this device's role is driver_safety\", \"code\": \"role_forbids_preset\"}"))
    check(coded.message == "this device's role is driver_safety", "a coded refusal's detail is shown as said")
    check(coded.code == "role_forbids_preset", "and its code is kept for the caller to branch on")
    check(coded.title == ServerReply.Title.refused, "headed as a refusal")

    let validation = ServerReply.httpFailure(status: 422, data: body(
        "{\"detail\": [{\"loc\": [\"body\", \"name\"], \"msg\": \"field required\", \"type\": \"missing\"}]}"))
    check(validation.message == "The server couldn't accept that — name: field required.",
          "FastAPI's validation list becomes a sentence (was 'request failed (422)')")

    let nested = ServerReply.httpFailure(status: 400, data: body("{\"detail\": {\"message\": \"bad zone\"}}"))
    check(nested.message == "bad zone", "a detail object's message")
    let other = ServerReply.httpFailure(status: 400, data: body("{\"error\": \"quota exceeded\"}"))
    check(other.message == "quota exceeded", "another server's {error: ...}")

    let long = String(repeating: "x", count: 1000)
    let bounded = ServerReply.httpFailure(status: 400, data: body("{\"detail\": \"\(long)\"}"))
    check(bounded.message.count == 280 && bounded.message.hasSuffix("…"), "a runaway detail is cut to a sentence")
    let markup = ServerReply.httpFailure(status: 500, data: body("{\"detail\": \"<html>boom</html>\"}"))
    check(clean(markup.message), "a detail that is itself markup is not shown")
}

print("\nerror replies with nothing readable say what the status means:")
do {
    let html = "<html><head><title>502 Bad Gateway</title></head><body><center>nginx</center></body></html>"
    let gateway = ServerReply.httpFailure(status: 502, data: body(html))
    check(gateway.title == ServerReply.Title.serverError, "a 502 is the server's problem")
    check(gateway.message.contains("try again in a moment") && clean(gateway.message),
          "a proxy's HTML page becomes advice, not markup: \(gateway.message)")
    let crash = ServerReply.httpFailure(status: 500, data: body("Internal Server Error"))
    check(crash.message.contains("internal error (500)") && crash.title == ServerReply.Title.serverError,
          "Starlette's plain-text 500")
    let empty = ServerReply.httpFailure(status: 503, data: Data())
    check(empty.message.contains("503") && clean(empty.message), "an empty 503")
    let missing = ServerReply.httpFailure(status: 404, data: body("{\"detail\": \"Not Found\"}"))
    check(missing.title == ServerReply.Title.notSupported, "an uncoded 404 is a missing feature")
    let notFound = ServerReply.httpFailure(status: 404, data: body("{\"detail\": \"no such device\", \"code\": \"device_not_found\"}"))
    check(notFound.title == ServerReply.Title.refused && notFound.message == "no such device",
          "a coded 404 is a refusal with its detail")
    check(ServerReply.httpFailure(status: 429, data: Data()).message.contains("busy"), "429 says busy")
    check(clean(ServerReply.httpFailure(status: 418, data: body("<p>teapot</p>")).message), "any other status")
}

print("\na success the app can't decode says why:")
do {
    let empty = ServerReply.undecodable(Data())
    check(empty.message.contains("empty reply") && empty.title == ServerReply.Title.unreadable, "an empty 200")
    let portal = ServerReply.undecodable(body("<!doctype html><title>Sign in to Wi-Fi</title>"))
    check(portal.title == ServerReply.Title.wrongServer, "a 200 web page is something else answering")
    check(portal.message.contains("Wi-Fi sign-in") && clean(portal.message), "and it says what that usually is")
    let garbage = ServerReply.undecodable(body("{\"devices\": [ this is not JSON"))
    check(garbage.message == ReplyError.unreadable, "broken JSON keeps the unreadable-reply message")
    check(ReplyError.bannerTitle(for: garbage.message) == ServerReply.Title.unreadable,
          "and the banner's headline agrees with it")
}

print("\na 401 ends the session only when there was one:")
check(ServerReply.sessionEnded(status: 401, sentToken: true, signedIn: true)?.contains("expired") == true,
      "a rejected token: sign in again")
check(ServerReply.sessionEnded(status: 401, sentToken: false, signedIn: true)?.contains("sign in again") == true,
      "signed in but no token sent: sign in again (was 'missing bearer token' on every refresh)")
check(ServerReply.sessionEnded(status: 401, sentToken: false, signedIn: false) == nil,
      "sign-in's own 401 (wrong password) is not an expired session")
check(ServerReply.sessionEnded(status: 403, sentToken: true, signedIn: true) == nil, "only 401")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)

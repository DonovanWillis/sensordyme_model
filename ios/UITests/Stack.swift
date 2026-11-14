import XCTest

/// The scenario controller (scripts/sim/uitest_control.py) on 127.0.0.1:8790.
/// The UI-test runner is a Simulator process, so 127.0.0.1 is the Mac it runs on.
///
/// Test isolation: every test (journeys and audits alike) starts with `fresh`,
/// from its setUp. The controller then replaces the whole world: a new backend
/// process with an empty database (new JWT secret; no events, rules, zones or
/// AI drafts; empty login and claim limiters), a new account, a newly
/// provisioned virtual device with a new identity, proxy faults off, and the
/// device claimed and READY unless the test asks for it unclaimed. Nothing one
/// test does can reach another, so the suite passes in any order (alphabetical,
/// or XCTest's random ordering: scripts/sim/ios_uitest.sh with ORDER=random).
/// The tests run one at a time (one controller, fixed ports, one Simulator, one
/// app install): parallel testing stays off — XcodeGen's scheme default, and
/// forced off on the xcodebuild command line by scripts/sim/ios_uitest.sh.
enum Stack {
    static let control = URL(string: "http://127.0.0.1:8790")!

    static var reachable: Bool { (try? request("GET", "/state")) != nil }

    /// Skips when the controller is not running (the tests started from Xcode
    /// on their own), but FAILS in a run scripts/sim/ios_uitest.sh started
    /// (ORYN_UITEST_REQUIRE_STACK=1): a run that skipped every test is no pass.
    static func requireController() throws {
        if reachable { return }
        let message = "the scenario controller is not running (scripts/sim/ios_uitest.sh)"
        if ProcessInfo.processInfo.environment["ORYN_UITEST_REQUIRE_STACK"] == "1" {
            throw ControllerError(message: message)
        }
        throw XCTSkip(message)
    }

    /// A new world for this test (see above). `claimed: false` leaves the new
    /// device unclaimed, for the journey that pairs it through Add device.
    /// Throws, failing the test in setUp, when the controller could not build it.
    static func fresh(for test: XCTestCase, claimed: Bool = true) throws {
        let tag = testName(test)
        let answer = try request("POST", "/fresh?claimed=\(claimed ? 1 : 0)&tag=\(tag)", timeout: 150)
        guard answer["device_running"] as? Bool == true, (answer["device_id"] is Int) == claimed else {
            throw ControllerError(message: "the fresh state for \(tag) is not what was asked for: \(answer)")
        }
    }

    /// "test03_Discovery" from "-[OrynJourneysUITests test03_Discovery]".
    static func testName(_ test: XCTestCase) -> String {
        String(test.name.split(separator: " ").last?.dropLast() ?? "")
    }

    struct ControllerError: Error, CustomStringConvertible {
        let message: String
        var description: String { message }
    }

    static func state() -> [String: Any] {
        (try? request("GET", "/state")) ?? [:]
    }

    static func session() -> (token: String, email: String) {
        let answer = (try? request("GET", "/session")) ?? [:]
        return (answer["token"] as? String ?? "", answer["email"] as? String ?? "")
    }

    @discardableResult
    static func post(_ path: String) -> [String: Any] {
        (try? request("POST", path)) ?? [:]
    }

    /// A bounded wait on something only the controller can see (no UI to watch).
    static func waitUntil(timeout: TimeInterval, _ condition: @escaping () -> Bool) -> Bool {
        let predicate = NSPredicate { _, _ in condition() }
        return XCTWaiter().wait(for: [XCTNSPredicateExpectation(predicate: predicate, object: nil)],
                                timeout: timeout) == .completed
    }

    /// Extra launch arguments for every launch (e.g. a text size), from the runner's
    /// environment: ORYN_UITEST_ARGS, space-separated (TEST_RUNNER_ORYN_UITEST_ARGS).
    static var extraLaunchArguments: [String] {
        (ProcessInfo.processInfo.environment["ORYN_UITEST_ARGS"] ?? "")
            .split(separator: " ").map(String.init)
    }

    /// Kept in the result bundle, and written as a PNG when ORYN_UITEST_SHOTS names a folder.
    static func saveScreenshot(_ screenshot: XCUIScreenshot, name: String, test: XCTestCase) {
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        test.add(attachment)
        if let dir = ProcessInfo.processInfo.environment["ORYN_UITEST_SHOTS"], !dir.isEmpty {
            try? screenshot.pngRepresentation.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).png"))
        }
    }

    /// Kept in the result bundle, and written as a .txt when ORYN_UITEST_SHOTS names a folder.
    static func saveText(_ text: String, name: String, test: XCTestCase) {
        let attachment = XCTAttachment(string: text)
        attachment.name = name
        attachment.lifetime = .keepAlways
        test.add(attachment)
        if let dir = ProcessInfo.processInfo.environment["ORYN_UITEST_SHOTS"], !dir.isEmpty {
            try? text.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).txt"),
                            atomically: true, encoding: .utf8)
        }
    }

    /// Up to three attempts on a fresh connection each: one dropped connection
    /// (-1005) once launched the app with no session and failed two journeys.
    /// Only a transport failure is tried again; an answer from the controller
    /// saying something failed is final.
    private static func request(_ method: String, _ path: String, timeout: TimeInterval = 20) throws -> [String: Any] {
        var lastError: Error = URLError(.unknown)
        for _ in 0..<3 {
            do {
                return try attempt(method, path, timeout: timeout)
            } catch let error as ControllerError {
                throw error
            } catch {
                lastError = error
            }
        }
        throw lastError
    }

    private static let urlSession = URLSession(configuration: .ephemeral)

    private static func attempt(_ method: String, _ path: String, timeout: TimeInterval) throws -> [String: Any] {
        var request = URLRequest(url: URL(string: path, relativeTo: control)!)
        request.httpMethod = method
        request.timeoutInterval = timeout
        request.setValue("close", forHTTPHeaderField: "Connection")
        var result: Result<Data, Error> = .failure(URLError(.timedOut))
        let done = DispatchSemaphore(value: 0)
        urlSession.dataTask(with: request) { data, response, error in
            if let error {
                result = .failure(error)
            } else if let http = response as? HTTPURLResponse, http.statusCode == 200, let data {
                result = .success(data)
            } else {
                // The controller says why ("claim failed (429): ..."): keep it for the failure.
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                let body = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
                result = .failure(ControllerError(message: "\(method) \(path): HTTP \(status) \(body)"))
            }
            done.signal()
        }.resume()
        _ = done.wait(timeout: .now() + timeout + 5)
        let data = try result.get()
        return (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }
}

import XCTest

/// The scenario controller (scripts/sim/uitest_control.py) on 127.0.0.1:8790.
/// The UI-test runner is a Simulator process, so 127.0.0.1 is the Mac it runs on.
enum Stack {
    static let control = URL(string: "http://127.0.0.1:8790")!

    static var reachable: Bool { (try? request("GET", "/state")) != nil }

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

    /// Device claimed, running "attentive", protection resumed, server answering.
    static func prepare() {
        if !(state()["device_id"] is Int) { post("/claim") }
        post("/reset")
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

    private static func request(_ method: String, _ path: String) throws -> [String: Any] {
        var request = URLRequest(url: URL(string: path, relativeTo: control)!)
        request.httpMethod = method
        request.timeoutInterval = 20
        var result: Result<Data, Error> = .failure(URLError(.timedOut))
        let done = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error {
                result = .failure(error)
            } else if let http = response as? HTTPURLResponse, http.statusCode == 200, let data {
                result = .success(data)
            } else {
                result = .failure(URLError(.badServerResponse))
            }
            done.signal()
        }.resume()
        _ = done.wait(timeout: .now() + 25)
        let data = try result.get()
        return (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }
}

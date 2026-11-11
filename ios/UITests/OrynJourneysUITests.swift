import XCTest

/// The app's user journeys, end to end: the real app (Debug build) in the iOS
/// Simulator, the real backend (cloud/app.py via sim.backend) and a virtual
/// SensorDyme (sim.virtual_device), started and faulted by
/// scripts/sim/uitest_control.py. Run them with scripts/sim/ios_uitest.sh.
///
/// Nothing is bypassed: a signed-in launch uses a token from a real /v1/login
/// (`-orynSession`, DebugLaunch.swift); sign-in itself is typed. Waits are for
/// conditions, each with a bound — no sleeps.
///
/// The journeys run in name order and share one backend: 03 claims the device
/// that every later journey uses. Each one sets the device state it needs.
final class OrynJourneysUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        try XCTSkipUnless(Stack.reachable, "the scenario controller is not running (scripts/sim/ios_uitest.sh)")
    }

    /// Set by `record(_:)`. `testRun.hasSucceeded` is not final during tearDown, so it
    /// read false for passing journeys too and every one got a FAIL_ screenshot.
    private var failed = false

    override func record(_ issue: XCTIssue) {
        failed = true
        super.record(issue)
    }

    override func tearDownWithError() throws {
        // What was on screen when it failed, named after the journey.
        if failed, let app, app.state == .runningForeground {
            shot("FAIL_\(name.split(separator: " ").last?.dropLast() ?? "")")
        }
        app?.terminate()
    }

    // MARK: Journeys

    func test01_Launch() throws {
        launch(signedIn: false)
        let email = app.textFields["auth.email"]
        XCTAssertTrue(email.waitForExistence(timeout: 20), "the sign-in screen shows")
        XCTAssertEqual(email.label, "Email", "the email field has a spoken name")
        XCTAssertEqual(app.secureTextFields["auth.password"].label, "Password")
        XCTAssertFalse(app.buttons["auth.submit"].isEnabled, "nothing to submit yet")
        shot("01_launch")
    }

    func test02_SignIn() throws {
        launch(signedIn: false)
        signIn(password: Stack.state()["password"] as? String ?? "")
        XCTAssertTrue(app.staticTexts["devices.title"].waitForExistence(timeout: 30), "signed in to the device list")
        shot("02_signed_in")
    }

    func test03_Discovery() throws {
        try XCTSkipIf(Stack.state()["device_id"] is Int, "the device is already claimed on this backend")
        launch()
        XCTAssertTrue(app.descendants(matching: .any)["state.empty"].waitForExistence(timeout: 30),
                      "an empty account says so, not 'loading' forever")
        shot("03a_no_devices")
        app.buttons["devices.add"].tap()
        let next = app.buttons["wizard.next"]
        for step in ["It's glowing", "I'm connected", "Device joined my Wi-Fi"] {
            XCTAssertTrue(next.waitForExistence(timeout: 10))
            XCTAssertEqual(next.label, step)
            next.tap()
        }
        let code = app.textFields["wizard.claimCode"]
        XCTAssertTrue(code.waitForExistence(timeout: 10))
        XCTAssertEqual(code.label, "Claim code")
        code.tap()
        code.typeText(Stack.state()["claim_code"] as? String ?? "")
        next.tap()
        XCTAssertTrue(app.buttons["wizard.done"].waitForExistence(timeout: 30), "the claim succeeded")
        shot("03b_paired")
        app.buttons["wizard.done"].tap()
        let card = app.buttons["device.card"]
        XCTAssertTrue(card.waitForExistence(timeout: 30))
        XCTAssertTrue(card.label.contains("SIMULATED"), "the virtual device is listed: \(card.label)")
        shot("03c_listed")
    }

    func test04_Ready() throws {
        Stack.prepare()
        launch()
        let card = app.buttons["device.card"]
        XCTAssertTrue(card.waitForExistence(timeout: 30))
        waitFor(card, labelContains: "Online", timeout: 60)
        card.tap()
        waitFor(monitoring, label: "All clear", timeout: 90)
        XCTAssertEqual(app.descendants(matching: .any)["detail.online"].label, "Online")
        shot("04_ready")
    }

    func test05_Stop() throws {
        Stack.prepare()
        launch(["-orynOpenDevice"])
        waitFor(monitoring, label: "All clear", timeout: 90)
        let protection = reveal(app.buttons["detail.protection"])
        XCTAssertTrue(protection.label.hasPrefix("Stop"), protection.label)
        protection.tap()
        let confirm = app.buttons["Stop monitoring"]
        XCTAssertTrue(confirm.waitForExistence(timeout: 10), "a stop asks first")
        confirm.tap()
        waitFor(monitoring, label: "Stopped", timeout: 60)
        waitFor(protection, labelContains: "Resume", timeout: 30)
        shot("05_stopped")
    }

    func test06_Resume() throws {
        Stack.prepare()
        Stack.post("/protection?action=stop")
        launch(["-orynOpenDevice"])
        waitFor(monitoring, label: "Stopped", timeout: 60)
        XCTAssertNotEqual(monitoring.label, "All clear", "a stopped device never reads as ready")
        let protection = reveal(app.buttons["detail.protection"])
        waitFor(protection, labelContains: "Resume", timeout: 30)
        protection.tap()
        waitFor(monitoring, label: "All clear", timeout: 60)
        shot("06_resumed")
    }

    func test07_PresetBrowse() throws {
        Stack.prepare()
        launch(["-orynOpenDevice"])
        waitFor(monitoring, label: "All clear", timeout: 90)
        let driver = reveal(app.buttons["preset.row.driver_safety"])
        for id in ["security_monitoring", "baby_monitoring", "elderly_monitoring"] {
            XCTAssertTrue(app.buttons["preset.row.\(id)"].exists, "\(id) is listed")
        }
        XCTAssertEqual(driver.value as? String, "Active", "the running mode says Active")
        shot("07a_presets")
        app.buttons["preset.info.driver_safety"].tap()
        XCTAssertTrue(app.navigationBars["Driver Safety"].waitForExistence(timeout: 10), "the details sheet opens")
        shot("07b_preset_detail")
        app.navigationBars["Driver Safety"].buttons["Done"].tap()
    }

    func test08_PresetActivate() throws {
        Stack.prepare()
        launch(["-orynOpenDevice"])
        waitFor(monitoring, label: "All clear", timeout: 90)
        let strict = reveal(app.buttons["preset.variant.strict"])
        strict.tap()
        waitFor(strict, selected: true, timeout: 60)  // from what the device reports, not the tap
        XCTAssertFalse(app.buttons["preset.variant.standard"].isSelected)
        shot("08_preset_strict")
        let standard = reveal(app.buttons["preset.variant.standard"])
        standard.tap()
        waitFor(standard, selected: true, timeout: 60)
    }

    func test09_PresetReject() throws {
        Stack.prepare()
        launch(["-orynOpenDevice"])
        waitFor(monitoring, label: "All clear", timeout: 90)
        let security = reveal(app.buttons["preset.row.security_monitoring"])
        waitFor(security, value: "Not available on this device", timeout: 30)
        XCTAssertFalse(security.isEnabled, "a preset the role forbids can't be picked")
        XCTAssertTrue(app.buttons["preset.info.security_monitoring"].isEnabled, "its reason can still be read")
        shot("09_preset_rejected")
        XCTAssertEqual(app.buttons["preset.row.driver_safety"].value as? String, "Active", "the mode is unchanged")
    }

    func test10_JupiterPrompt() throws {
        Stack.prepare()
        launch(["-orynOpenDevice"])
        waitFor(monitoring, label: "All clear", timeout: 90)
        let input = app.textFields["jupiter.input"].firstMatch
        XCTAssertTrue(input.waitForExistence(timeout: 10))
        input.tap()
        input.typeText(Self.dwellPrompt)
        XCTAssertTrue(app.buttons["jupiter.send"].firstMatch.isEnabled, "a typed request can be sent")
        input.typeText("\n")  // the keyboard's Send key
        let explanation = app.staticTexts["jupiter.explanation"]
        XCTAssertTrue(explanation.waitForExistence(timeout: 45), "Jupiter answers")
        XCTAssertTrue(explanation.label.contains("marked area"), explanation.label)
        shot("10_jupiter_understood")
    }

    func test11_PolicyPreview() throws {
        Stack.prepare()
        openJupiter(prompt: Self.dwellPrompt)
        let preview = reveal(element("jupiter.preview"))
        XCTAssertTrue(preview.exists)
        XCTAssertTrue(app.staticTexts["Will watch for"].exists || label(containing: "Will watch for").exists)
        XCTAssertTrue(label(containing: "area is drawn").exists, "the preview says what the rule waits for")
        shot("11_policy_preview")
        XCTAssertTrue(app.buttons["jupiter.activate"].exists, "nothing is applied before Activate")
    }

    func test12_ZoneRequirement() throws {
        Stack.prepare()
        openJupiter(prompt: Self.dwellPrompt)
        let zone = reveal(element("jupiter.zoneRequirement"))
        XCTAssertTrue(zone.label.contains("Draw the"), zone.label)
        XCTAssertTrue(zone.label.contains("marked area"), zone.label)
        shot("12_zone_required")
    }

    func test13_RuleEnableDisable() throws {
        Stack.prepare()
        openJupiter(prompt: Self.dwellPrompt)
        let activate = reveal(app.buttons["jupiter.activate"])
        activate.tap()
        let confirm = app.buttons["Activate"]
        XCTAssertTrue(confirm.waitForExistence(timeout: 10), "activating asks first")
        confirm.tap()
        XCTAssertTrue(element("jupiter.applied").waitForExistence(timeout: 60))
        shot("13a_rule_sent")
        app.buttons["jupiter.done"].tap()
        let toggle = reveal(app.switches["rule.toggle"].firstMatch, timeout: 30)
        XCTAssertFalse(toggle.label.hasPrefix("Turn"), "the switch is named by its rule: \(toggle.label)")
        waitFor(toggle, value: "1", timeout: 30)
        toggle.tap()
        waitFor(toggle, value: "0", timeout: 30)  // what the server answered, not an optimistic flip
        shot("13b_rule_disabled")
        toggle.tap()
        waitFor(toggle, value: "1", timeout: 30)
        shot("13c_rule_enabled")
    }

    func test14_MonitoringStatus() throws {
        Stack.prepare()
        Stack.post("/device/start?scenario=detector_failure")
        launch(["-orynOpenDevice"])
        waitFor(monitoring, label: "Degraded", timeout: 120)
        shot("14a_degraded")
        // The fault lasts 40 s, but the device retries the detector with backoff
        // capped at 60 s (modules/driver_attention.py), so recovery can take ~100 s.
        waitFor(monitoring, label: "All clear", timeout: 180)
        shot("14b_ready_again")
    }

    func test15_Offline() throws {
        Stack.prepare()
        launch()
        let card = app.buttons["device.card"]
        XCTAssertTrue(card.waitForExistence(timeout: 30))
        waitFor(card, labelContains: "Online", timeout: 60)
        Stack.post("/device/stop")
        waitFor(card, labelContains: "Offline", timeout: 90)
        XCTAssertTrue(card.label.contains("Last seen"), "offline says since when: \(card.label)")
        shot("15a_offline_card")
        card.tap()
        waitFor(element("banner.message"), labelContains: "isn't reporting", timeout: 30)
        waitFor(monitoring, label: "Status unknown", timeout: 30)
        shot("15b_offline_detail")
    }

    func test16_Stale() throws {
        Stack.prepare()
        Stack.post("/device/start?scenario=backend_outage")  // the device loses the server for 60 s
        launch(["-orynOpenDevice"])
        waitFor(monitoring, label: "All clear", timeout: 90)
        let banner = element("banner.message")
        waitFor(banner, labelContains: "isn't reporting", timeout: 90)
        XCTAssertNotEqual(monitoring.label, "All clear", "a stale device is never healthy")
        XCTAssertEqual(monitoring.label, "Status unknown")
        shot("16a_stale")
        waitFor(monitoring, label: "All clear", timeout: 120)  // fresh again on its own
        waitForAbsence(banner, timeout: 30)
        shot("16b_fresh_again")
    }

    func test17_Recovery() throws {
        Stack.prepare()
        Stack.post("/device/stop")
        launch(["-orynOpenDevice"])
        waitFor(monitoring, label: "Status unknown", timeout: 90)
        Stack.post("/device/start?scenario=attentive")  // the device restarts
        waitFor(monitoring, label: "All clear", timeout: 120)
        waitForAbsence(element("banner.message"), timeout: 30)
        shot("17_recovered")
    }

    func test18_AlertHistory() throws {
        Stack.prepare()
        let before = Stack.state()["events"] as? Int ?? 0
        Stack.post("/device/start?scenario=drowsy")
        XCTAssertTrue(Stack.waitUntil(timeout: 120) { (Stack.state()["events"] as? Int ?? 0) > before },
                      "the drowsy scenario produced an alert on the backend")
        launch(["-orynOpenDevice", "-orynOpenHistory"])
        let row = app.descendants(matching: .any)["event.row"].firstMatch
        XCTAssertTrue(row.waitForExistence(timeout: 30), "the alert is in the history")
        XCTAssertTrue(row.label.contains("Alert") || row.label.contains("Event"), "severity is spoken: \(row.label)")
        shot("18a_history")
        row.tap()
        shot("18b_event_detail")
    }

    func test19_InvalidResponses() throws {
        Stack.prepare()
        launch()
        let card = app.buttons["device.card"]
        XCTAssertTrue(card.waitForExistence(timeout: 30))
        Stack.post("/mode?m=garbage")
        defer { Stack.post("/mode?m=pass") }
        let banner = element("banner.message")
        waitFor(banner, labelContains: "couldn't read its reply", timeout: 45)
        XCTAssertFalse(banner.label.contains("correct format"), "no raw decoder text: \(banner.label)")
        XCTAssertTrue(card.exists, "the last good list stays on screen")
        shot("19a_invalid_reply")
        Stack.post("/mode?m=pass")
        app.buttons["banner.retry"].tap()
        waitForAbsence(banner, timeout: 30)
        shot("19b_recovered")
    }

    func test20_ErrorMessaging() throws {
        Stack.prepare()
        launch(signedIn: false)
        signIn(password: "not-the-password")
        let error = element("auth.error")
        XCTAssertTrue(error.waitForExistence(timeout: 30), "a refused sign-in says so")
        XCTAssertTrue(error.label.hasPrefix("Error:"), error.label)
        shot("20a_wrong_password")

        Stack.post("/mode?m=down")
        defer { Stack.post("/mode?m=pass") }
        launch()
        let banner = element("banner.message")
        waitFor(banner, labelContains: "Can't reach the server", timeout: 60)
        XCTAssertTrue(banner.label.contains("Settings") || banner.label.contains("backend"),
                      "the message says what to check: \(banner.label)")
        shot("20b_server_down")
        Stack.post("/mode?m=pass")
        app.buttons["banner.retry"].tap()
        waitForAbsence(banner, timeout: 45)
        XCTAssertTrue(app.buttons["device.card"].waitForExistence(timeout: 30))
        shot("20c_server_back")
    }

    // MARK: Helpers

    static let dwellPrompt = "Alert me if someone stays inside the marked area for longer than 10 seconds"

    private var monitoring: XCUIElement { element("detail.monitoring") }

    private func element(_ id: String) -> XCUIElement {
        app.descendants(matching: .any)[id].firstMatch
    }

    private func label(containing text: String) -> XCUIElement {
        app.descendants(matching: .any).matching(NSPredicate(format: "label CONTAINS %@", text)).firstMatch
    }

    private func launch(_ extra: [String] = [], signedIn: Bool = true) {
        var args = ["-orynNoPrompt", "-orynSignOut"]
        if signedIn {
            let session = Stack.session()
            args += ["-orynSession", session.token, session.email]
        }
        app.launchArguments = args + extra + Stack.extraLaunchArguments
        app.launch()
    }

    private func openJupiter(prompt: String) {
        launch(["-orynOpenDevice", "-orynOpenJupiter", "-orynPrompt", prompt])
        XCTAssertTrue(app.staticTexts["jupiter.explanation"].waitForExistence(timeout: 60), "Jupiter answers")
    }

    private func signIn(password: String) {
        let email = app.textFields["auth.email"]
        XCTAssertTrue(email.waitForExistence(timeout: 20))
        email.tap()
        email.typeText(Stack.state()["email"] as? String ?? "")
        let field = app.secureTextFields["auth.password"]
        field.tap()
        field.typeText(password)
        let submit = app.buttons["auth.submit"]
        XCTAssertTrue(submit.isEnabled)
        submit.tap()
    }

    /// Scroll until the element is on screen and clear of the floating tab bar
    /// (bounded), and return it. `isHittable` alone passed for a chip lying under
    /// the tab bar, and the tap landed on the tab bar.
    @discardableResult
    private func reveal(_ element: XCUIElement, timeout: TimeInterval = 15) -> XCUIElement {
        XCTAssertTrue(element.waitForExistence(timeout: timeout), "\(element) exists")
        let limit = app.frame.maxY * 0.78
        var swipes = 0
        while (!element.isHittable || element.frame.maxY > limit) && swipes < 10 {
            app.swipeUp(velocity: .slow)
            swipes += 1
        }
        XCTAssertTrue(element.isHittable, "\(element) is reachable on screen")
        return element
    }

    private func waitFor(_ element: XCUIElement, label: String, timeout: TimeInterval,
                         file: StaticString = #filePath, line: UInt = #line) {
        expectPredicate("label == %@", label, on: element, timeout: timeout, file: file, line: line)
    }

    private func waitFor(_ element: XCUIElement, labelContains text: String, timeout: TimeInterval,
                         file: StaticString = #filePath, line: UInt = #line) {
        expectPredicate("label CONTAINS %@", text, on: element, timeout: timeout, file: file, line: line)
    }

    private func waitFor(_ element: XCUIElement, value: String, timeout: TimeInterval,
                         file: StaticString = #filePath, line: UInt = #line) {
        expectPredicate("value == %@", value, on: element, timeout: timeout, file: file, line: line)
    }

    private func waitFor(_ element: XCUIElement, selected: Bool, timeout: TimeInterval,
                         file: StaticString = #filePath, line: UInt = #line) {
        let predicate = NSPredicate(format: "isSelected == %@", NSNumber(value: selected))
        let result = XCTWaiter().wait(for: [XCTNSPredicateExpectation(predicate: predicate, object: element)],
                                      timeout: timeout)
        XCTAssertEqual(result, .completed, "\(element) selected == \(selected)", file: file, line: line)
    }

    private func waitForAbsence(_ element: XCUIElement, timeout: TimeInterval,
                                file: StaticString = #filePath, line: UInt = #line) {
        let predicate = NSPredicate(format: "exists == false")
        let result = XCTWaiter().wait(for: [XCTNSPredicateExpectation(predicate: predicate, object: element)],
                                      timeout: timeout)
        XCTAssertEqual(result, .completed, "\(element) went away", file: file, line: line)
    }

    private func expectPredicate(_ format: String, _ argument: String, on element: XCUIElement,
                                 timeout: TimeInterval, file: StaticString, line: UInt) {
        let predicate = NSPredicate(format: format, argument)
        let result = XCTWaiter().wait(for: [XCTNSPredicateExpectation(predicate: predicate, object: element)],
                                      timeout: timeout)
        let seen = element.exists ? "label=\(element.label) value=\(String(describing: element.value))" : "missing"
        XCTAssertEqual(result, .completed, "waited \(Int(timeout)) s for \(format) \(argument); \(seen)",
                       file: file, line: line)
    }

    private func shot(_ name: String) {
        Stack.saveScreenshot(XCUIScreen.main.screenshot(), name: name, test: self)
    }
}

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
        // The row opens the details (one activation path); Activate is what is off.
        security.tap()
        let activate = app.buttons["preset.activate"]
        XCTAssertTrue(activate.waitForExistence(timeout: 10), "the details open")
        XCTAssertFalse(activate.isEnabled, "a preset the role forbids can't be activated")
        shot("09_preset_rejected")
        app.navigationBars.buttons["Done"].firstMatch.tap()
        XCTAssertEqual(app.buttons["preset.row.driver_safety"].value as? String, "Active", "the mode is unchanged")
    }

    func test10_JupiterPrompt() throws {
        Stack.prepare()
        launch(["-orynOpenDevice"])
        waitFor(monitoring, label: "All clear", timeout: 90)
        // Clear of the floating tab bar first: a tap there lands on the tab bar
        // and the field never gets focus.
        let input = reveal(app.textFields["jupiter.input"].firstMatch, timeout: 10)
        input.tap()
        XCTAssertTrue(app.keyboards.firstMatch.waitForExistence(timeout: 10), "tapping the field brings up the keyboard")
        waitFor(input, keyboardFocus: true, timeout: 10)
        // Focus must survive the screen's own refreshes (every 4 s, plus the
        // device list's every 10 s): a refresh that rebuilt the field would drop
        // the keyboard mid-sentence. Only a field that is found AND says it has
        // no focus counts: during a refresh XCUITest can briefly fail to resolve
        // the field (seen once in M8, ~20 s after the tap, with the field still
        // focused on screen), and the old journey typed through `input` at such
        // a moment — "neither element nor any descendant has keyboard focus".
        let lost = Stack.waitUntil(timeout: 11) { self.focusState(input) == false }
        if lost {  // what the app looked like at that moment, for diagnosis
            Stack.saveText(app.debugDescription, name: "10_focus_lost_tree", test: self)
            shot("10_focus_lost")
        }
        XCTAssertFalse(lost, "the field keeps keyboard focus across refreshes")
        // Typed into whatever holds focus (just shown to be the field), so the
        // keystrokes don't depend on re-resolving the field mid-refresh.
        app.typeText(Self.dwellPrompt)
        XCTAssertEqual(input.value as? String, Self.dwellPrompt, "the request went into the Ask Jupiter field")
        XCTAssertTrue(app.buttons["jupiter.send"].firstMatch.isEnabled, "a typed request can be sent")
        app.typeText("\n")  // the keyboard's Send key
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
        XCTAssertFalse(banner.label.contains("Can't reach the server"),
                       "the server answered: the headline must not say it is unreachable: \(banner.label)")
        XCTAssertTrue(card.exists, "the last good list stays on screen")
        waitFor(card, labelContains: "Unknown", timeout: 15)  // not a stale "Online"
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

    // MARK: Mission 7: the Driver Safety ("Distracted Driving") card, end to end

    /// Tap the card, read what it detects on THIS device and how far it is
    /// validated, activate, see the device's own confirmation, trigger a
    /// distraction, find it in history, break the detector, see degraded, recover.
    /// (The failed component named under the status is journey 25.)
    func test23_DriverSafetyCard() throws {
        Stack.prepare()
        launch(["-orynOpenDevice"])
        waitFor(monitoring, label: "All clear", timeout: 90)

        reveal(app.buttons["preset.row.driver_safety"]).tap()
        XCTAssertTrue(app.navigationBars["Driver Safety"].waitForExistence(timeout: 10), "the card opens its details")
        let face = element("preset.capability.face")
        XCTAssertTrue(face.waitForExistence(timeout: 10), "what it needs, checked on this device")
        XCTAssertTrue(face.label.contains("Working"), "face, as the device reports it: \(face.label)")
        waitFor(element("preset.monitoring"), label: "All clear", timeout: 30)
        XCTAssertFalse(app.staticTexts.containing(NSPredicate(format: "label CONTAINS[c] %@",
                                                              "validated for production")).firstMatch.exists,
                       "nothing on the card claims production validation")
        shot("23a_card_details")

        let activate = reveal(app.buttons["preset.activate"])
        activate.tap()  // "Save settings": the running preset, re-applied with its settings
        let confirmed = element("preset.outcome.confirmed")
        XCTAssertTrue(confirmed.waitForExistence(timeout: 60), "the DEVICE confirms, not the tap")
        XCTAssertTrue(confirmed.label.contains("monitoring"), confirmed.label)
        shot("23b_activated")
        app.navigationBars["Driver Safety"].buttons["Done"].tap()

        let before = Stack.state()["events"] as? Int ?? 0
        Stack.post("/device/start?scenario=phone")
        XCTAssertTrue(Stack.waitUntil(timeout: 120) { (Stack.state()["events"] as? Int ?? 0) > before },
                      "the scripted distraction produced an event on the backend")
        waitFor(monitoring, label: "All clear", timeout: 90)
        reveal(app.buttons["detail.history"]).tap()
        let row = app.descendants(matching: .any)["event.row"].firstMatch
        XCTAssertTrue(row.waitForExistence(timeout: 30), "the distraction is in the history")
        shot("23c_history")
        app.navigationBars.buttons.element(boundBy: 0).tap()

        Stack.post("/device/start?scenario=detector_failure")
        waitFor(monitoring, label: "Degraded", timeout: 120)
        shot("23d_degraded")
        waitFor(monitoring, label: "All clear", timeout: 180)
        shot("23e_recovered")
    }

    /// The occupancy companion runs beside Driver Safety on the same detections:
    /// the switch follows the device's report, both ways.
    func test24_CompanionNode() throws {
        Stack.prepare()
        launch(["-orynOpenDevice"])
        waitFor(monitoring, label: "All clear", timeout: 90)
        try requirePerception()
        let toggle = reveal(app.switches["node.toggle.people_counter"])
        if (toggle.value as? String) == "1" {  // left on by an earlier run
            flip(toggle)
            waitFor(toggle, value: "0", timeout: 60)
        }
        flip(toggle)
        waitFor(toggle, value: "1", timeout: 60)  // the device reports the node running
        let node = element("node.people_counter")
        XCTAssertTrue(node.waitForExistence(timeout: 30), "the node is listed as running")
        XCTAssertTrue(node.label.contains("Experimental"), "an experimental node says so: \(node.label)")
        XCTAssertTrue(element("node.driver_safety").exists, "Driver Safety keeps running beside it")
        waitFor(monitoring, label: "All clear", timeout: 30)
        shot("24a_companion_on")
        flip(toggle)
        waitFor(toggle, value: "0", timeout: 60)
        waitForAbsence(node, timeout: 30)
        shot("24b_companion_off")
    }

    /// A failing perception component is named under the status, in the
    /// device's words, and disappears when it recovers.
    func test25_PerceptionDegraded() throws {
        Stack.prepare()
        launch(["-orynOpenDevice"])
        waitFor(monitoring, label: "All clear", timeout: 90)
        try requirePerception()
        Stack.post("/device/start?scenario=detector_failure")
        waitFor(monitoring, label: "Degraded", timeout: 120)
        let impaired = element("perception.degraded")
        XCTAssertTrue(impaired.waitForExistence(timeout: 30), "the failed component is named under the status")
        XCTAssertTrue(impaired.label.contains("Object detection"), impaired.label)
        shot("25a_component_failed")
        waitFor(monitoring, label: "All clear", timeout: 180)
        waitForAbsence(impaired, timeout: 30)
        shot("25b_component_recovered")
    }

    // MARK: Mission 8: telemetry truth and readable failures

    /// The app receives stale telemetry: every reply from the server is fine,
    /// but the device data in it stops changing (a cache or replica replaying
    /// the last device list: online=true, last_seen frozen). The app must not
    /// keep saying ONLINE: the card, the count and the device screen go Stale
    /// together, and recover together.
    func test26_StaleTelemetry() throws {
        Stack.prepare()
        launch()
        let card = app.buttons["device.card"]
        XCTAssertTrue(card.waitForExistence(timeout: 30))
        waitFor(card, labelContains: "Online", timeout: 60)
        Stack.post("/mode?m=freeze")
        defer { Stack.post("/mode?m=pass") }
        // last_seen + live window (30 s) + clock tolerance (30 s), polled every 10 s.
        waitFor(card, labelContains: "Stale data", timeout: 100)
        XCTAssertFalse(card.label.contains("Online"), "stale is never online: \(card.label)")
        XCTAssertTrue(card.label.contains("showing its last report"), card.label)
        XCTAssertTrue(label(containing: "0 ONLINE").exists || label(containing: "0 online").exists,
                      "the summary counts it out")
        shot("26a_stale_card")
        card.tap()
        waitFor(element("detail.online"), labelContains: "Stale", timeout: 30)
        waitFor(monitoring, label: "Status unknown", timeout: 30)  // never "All clear" on old telemetry
        waitFor(element("banner.message"), labelContains: "may no longer be true", timeout: 30)
        shot("26b_stale_detail")
        Stack.post("/mode?m=pass")
        waitFor(element("detail.online"), label: "Online", timeout: 60)
        waitFor(monitoring, label: "All clear", timeout: 60)
        waitForAbsence(element("banner.message"), timeout: 30)
        shot("26c_fresh_again")
    }

    /// Replies the app can't use are said in words, with a headline that
    /// matches: a web page where SensorDyme should be, an empty body, a 500
    /// with the server's own detail, a proxy's 502 page — never raw JSON or
    /// HTML, never "request failed (502)", never silence. Ask Jupiter (its own
    /// request path) says the same.
    func test27_UnreadableReplies() throws {
        Stack.prepare()
        launch()
        let card = app.buttons["device.card"]
        XCTAssertTrue(card.waitForExistence(timeout: 30))
        defer { Stack.post("/mode?m=pass") }
        let banner = element("banner.message")
        let cases: [(mode: String, says: String, shot: String)] = [
            ("html", "Something else answered at this address", "27a_web_page"),
            ("empty", "empty reply", "27b_empty"),
            ("error500", "database is locked", "27c_server_error"),
            ("gateway", "gateway", "27d_gateway"),
        ]
        for fault in cases {
            Stack.post("/mode?m=\(fault.mode)")
            waitFor(banner, labelContains: fault.says, timeout: 45)
            for raw in ["<", "{", "request failed", "correct format"] {
                XCTAssertFalse(banner.label.contains(raw), "\(fault.mode): no raw reply text: \(banner.label)")
            }
            XCTAssertFalse(banner.label.contains("Can't reach the server"),
                           "\(fault.mode): the server answered; the headline must not say otherwise: \(banner.label)")
            waitFor(card, labelContains: "Unknown", timeout: 15)
            shot(fault.shot)
        }
        XCTAssertTrue(banner.label.contains("reported a problem") || banner.label.contains("gateway"), banner.label)
        Stack.post("/mode?m=pass")
        app.buttons["banner.retry"].tap()
        waitForAbsence(banner, timeout: 30)

        // Ask Jupiter's request fails with a server error: the screen says so.
        Stack.post("/mode?m=error500&path=/v1/ai/configure")
        app.terminate()
        launch(["-orynOpenDevice", "-orynOpenJupiter", "-orynPrompt", Self.dwellPrompt])
        let error = element("jupiter.error")
        XCTAssertTrue(error.waitForExistence(timeout: 60), "a failed request is shown, not a silent spinner")
        XCTAssertTrue(error.label.contains("database is locked"), error.label)
        XCTAssertFalse(error.label.contains("{"), error.label)
        shot("27e_jupiter_server_error")
    }

    // MARK: Helpers

    private func hasFocus(_ element: XCUIElement) -> Bool {
        focusState(element) ?? false
    }

    /// The element's keyboard focus, or nil when it can't be resolved right now.
    private func focusState(_ element: XCUIElement) -> Bool? {
        guard element.exists else { return nil }
        return element.value(forKey: "hasKeyboardFocus") as? Bool
    }

    /// Flip a labelled switch by its knob: XCUITest taps an element's centre,
    /// which for a SwiftUI Toggle with a long label is the label, and iOS does
    /// not toggle a switch from its label (journey 24 tapped and nothing moved).
    private func flip(_ toggle: XCUIElement) {
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.94, dy: 0.5)).tap()
    }

    private func waitFor(_ element: XCUIElement, keyboardFocus: Bool, timeout: TimeInterval,
                         file: StaticString = #filePath, line: UInt = #line) {
        let ok = Stack.waitUntil(timeout: timeout) { self.hasFocus(element) == keyboardFocus }
        XCTAssertTrue(ok, "\(element) keyboard focus == \(keyboardFocus)", file: file, line: line)
    }

    /// Skips, saying why, when the stack does not report `status.perception`
    /// (sd-perception/1) yet: that needs the M7 device and backend streams.
    /// A skip is reported as a skip, never as a pass.
    private func requirePerception() throws {
        let section = element("nodes.section")
        if !section.waitForExistence(timeout: 20) {
            throw XCTSkip("the device/backend don't report status.perception yet (M7 device + backend streams)")
        }
        _ = reveal(section)
    }

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
            // An empty token signs the app in with nothing to send (see uitest_control.py).
            XCTAssertFalse(session.token.isEmpty, "the scenario controller could not sign the demo account in")
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

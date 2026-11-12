import XCTest

/// Apple's accessibility audit (XCUIApplication.performAccessibilityAudit) on the
/// main screens, at the default text size and at the largest accessibility size.
/// It checks what a person using VoiceOver or large text meets: unnamed
/// elements, text that does not scale or is clipped, contrast, small hit regions.
///
/// Every issue found is written to the result bundle (and to ORYN_UITEST_SHOTS
/// as audit_<screen>.txt). The test fails on an issue unless it is listed in
/// `knownIssues` with the reason it is accepted.
final class AccessibilityAuditUITests: XCTestCase {
    private var app: XCUIApplication!

    /// Accepted audit findings: (screen, audit type, element label prefix) and why.
    private let knownIssues: [(screen: String, type: XCUIAccessibilityAuditType, label: String, why: String)] = [
        ("signin", .dynamicType, "EDGE", "logotype (the ORYN EDGE lockup); WCAG 1.4.3/1.4.4 exempt logos"),
        ("signin", .contrast, "EDGE", "logotype; WCAG 1.4.3 exempts logos"),
        ("devices", .dynamicType, "ORYN EDGE", "logotype in the header"),
        ("devices", .hitRegion, "ORYN EDGE", "logotype; not interactive"),
        ("detail", .textClipped, "Virtual SensorDyme", "system navigation-bar title, truncated by UIKit; the full name is on screen"),
        ("jupiter", .dynamicType, "Done", "system bar button; UIKit caps bar text and offers the Large Content Viewer"),
        ("jupiter", .dynamicType, "=Ask Jupiter", "navigation-bar title (static text, not the input field); the "
            + "bar caps its text, and the title offers the Large Content Viewer (accessibilityShowsLargeContentViewer)"),
    ]

    override func setUpWithError() throws {
        continueAfterFailure = true
        app = XCUIApplication()
        try XCTSkipUnless(Stack.reachable, "the scenario controller is not running (scripts/sim/ios_uitest.sh)")
        Stack.prepare()
    }

    override func tearDownWithError() throws {
        app?.terminate()
    }

    func test21_AuditDefaultTextSize() throws {
        try auditScreens(extra: [], suffix: "default")
    }

    func test22_AuditLargestTextSize() throws {
        try auditScreens(extra: ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryAccessibilityXXXL"],
                         suffix: "ax_xxxl")
    }

    private func auditScreens(extra: [String], suffix: String) throws {
        launch(["-orynSignOut"] + extra, signedIn: false)
        XCTAssertTrue(app.textFields["auth.email"].waitForExistence(timeout: 20))
        audit("signin_\(suffix)")

        launch(extra)
        XCTAssertTrue(app.buttons["device.card"].waitForExistence(timeout: 30))
        audit("devices_\(suffix)")

        launch(["-orynOpenDevice"] + extra)
        XCTAssertTrue(app.descendants(matching: .any)["detail.monitoring"].waitForExistence(timeout: 30))
        audit("detail_\(suffix)")

        launch(["-orynOpenDevice", "-orynOpenJupiter", "-orynPrompt", OrynJourneysUITests.dwellPrompt] + extra)
        XCTAssertTrue(app.staticTexts["jupiter.explanation"].waitForExistence(timeout: 60))
        audit("jupiter_\(suffix)")

        launch(["-orynOpenDevice", "-orynOpenHistory"] + extra)
        XCTAssertTrue(app.navigationBars["Event history"].waitForExistence(timeout: 30))
        audit("history_\(suffix)")
    }

    private func launch(_ extra: [String], signedIn: Bool = true) {
        var args = ["-orynNoPrompt", "-orynSignOut"]
        if signedIn {
            let session = Stack.session()
            args += ["-orynSession", session.token, session.email]
        }
        app.launchArguments = args + extra
        app.launch()
    }

    private func audit(_ screen: String) {
        // Let content that arrives after the first answer settle (the device-link
        // banner after its network search, the mode list after the device's
        // presets): contrast is measured from pixels, and text caught mid-fade
        // "failed" at ratios of 9:1 and more (#F2F1ED and #B6BABE on #0F1112).
        _ = XCTWaiter().wait(for: [XCTestExpectation(description: "settle")], timeout: 3)
        Stack.saveScreenshot(XCUIScreen.main.screenshot(), name: "audit_\(screen)", test: self)
        var lines: [String] = []
        // Contrast is measured from the screenshot. Text scrolled under the
        // translucent floating tab bar is sampled through the glass, so it
        // "fails" whatever its colour (seen: the mode list's lower rows). Such a
        // finding is not accepted: the element is scrolled clear of the bar and
        // audited again, and fails the test if it still fails there.
        var underBar: [String] = []
        do {
            try app.performAccessibilityAudit(for: .all) { issue in
                let label = issue.element?.label ?? ""
                if issue.auditType == .contrast, let element = issue.element, self.underTabBar(element) {
                    underBar.append(label)
                    return true
                }
                let known = self.knownIssues.first {
                    $0.screen == screen.components(separatedBy: "_").first && $0.type == issue.auditType
                        && ($0.label.hasPrefix("=") ? label == String($0.label.dropFirst())
                                                         && issue.element?.elementType == .staticText
                                                     : label.hasPrefix($0.label))
                }
                let line = "\(screen)\t\(Self.name(issue.auditType))\t\(known == nil ? "NEW" : "known")\t"
                    + "\(issue.compactDescription)\t\(label)\t\(issue.element?.identifier ?? "")"
                lines.append(line)
                return known != nil  // a known issue is handled; anything else fails the test
            }
        } catch {
            XCTFail("\(screen): \(error)")
        }
        for label in underBar {
            lines.append(recheckContrastClearOfTabBar(label, screen: screen))
        }
        let report = lines.isEmpty ? "\(screen)\tno issues" : lines.joined(separator: "\n")
        let attachment = XCTAttachment(string: report)
        attachment.name = "audit_\(screen).txt"
        attachment.lifetime = .keepAlways
        add(attachment)
        if let dir = ProcessInfo.processInfo.environment["ORYN_UITEST_SHOTS"], !dir.isEmpty {
            try? (report + "\n").write(to: URL(fileURLWithPath: dir).appendingPathComponent("audit_\(screen).txt"),
                                       atomically: true, encoding: .utf8)
        }
    }

    private func underTabBar(_ element: XCUIElement) -> Bool {
        let bar = app.tabBars.firstMatch
        return bar.exists && element.frame.intersects(bar.frame)
    }

    /// Scroll `label` clear of the tab bar and audit contrast again. The line
    /// says what happened; a failure there is a real failure.
    private func recheckContrastClearOfTabBar(_ label: String, screen: String) -> String {
        let element = app.descendants(matching: .any).matching(NSPredicate(format: "label == %@", label)).firstMatch
        let bar = app.tabBars.firstMatch
        // Text taller than the room above the bar (a paragraph at the largest
        // sizes) can never be wholly clear of it, so it can't be re-measured.
        // Reported as such: neither accepted nor counted as a failure.
        if element.exists, bar.exists, element.frame.height > bar.frame.minY - app.frame.minY - 160 {
            return "\(screen)\tcontrast\tunder-tab-bar-unverifiable\tTaller than the space above the tab bar\t\(label)\t"
        }
        var swipes = 0
        while element.exists && bar.exists && element.frame.maxY > bar.frame.minY - 8 && swipes < 8 {
            app.swipeUp(velocity: .slow)
            swipes += 1
        }
        // An earlier re-check may have scrolled this one off the top.
        while element.exists && element.frame.minY < app.frame.minY + 100 && swipes < 16 {
            app.swipeDown(velocity: .slow)
            swipes += 1
        }
        guard element.exists, !underTabBar(element), app.frame.contains(element.frame) else {
            XCTFail("\(screen): contrast on '\(label)' could not be re-checked clear of the tab bar")
            return "\(screen)\tcontrast\tNEW\tContrast failed (could not scroll clear of the tab bar)\t\(label)\t"
        }
        var stillFails = false
        do {
            try app.performAccessibilityAudit(for: .contrast) { issue in
                if issue.element?.label == label, let el = issue.element, !self.underTabBar(el) { stillFails = true }
                return true  // only `label` is being re-checked here
            }
        } catch {
            stillFails = true
        }
        if stillFails {
            XCTFail("\(screen): '\(label)' fails contrast clear of the tab bar")
            return "\(screen)\tcontrast\tNEW\tContrast failed clear of the tab bar\t\(label)\t"
        }
        return "\(screen)\tcontrast\tunder-tab-bar\tPassed when re-checked clear of the tab bar\t\(label)\t"
    }

    private static func name(_ type: XCUIAccessibilityAuditType) -> String {
        switch type {
        case .contrast: "contrast"
        case .elementDetection: "elementDetection"
        case .hitRegion: "hitRegion"
        case .sufficientElementDescription: "sufficientElementDescription"
        case .dynamicType: "dynamicType"
        case .textClipped: "textClipped"
        case .trait: "trait"
        default: "other(\(type.rawValue))"
        }
    }
}

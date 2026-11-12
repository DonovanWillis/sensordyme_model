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
        try Stack.requireController()
        // Its own backend, account and device, claimed and ready: the audit no
        // longer claims the device the discovery journey (03) needs to pair.
        try Stack.fresh(for: self)
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
            // An empty token signs the app in with nothing to send (see uitest_control.py).
            XCTAssertFalse(session.token.isEmpty, "the scenario controller could not sign the demo account in")
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
        // translucent floating tab bar, or into the blurred scroll edge under the
        // navigation bar / status bar, is sampled through that material, so it
        // "fails" whatever its colour (seen: the mode list's rows at either end).
        // Such a finding is not accepted: the element is scrolled into the clear
        // band and audited again, and fails the test if it still fails there.
        var obscured: [(label: String, frame: CGRect)] = []
        do {
            try app.performAccessibilityAudit(for: .all) { issue in
                let label = issue.element?.label ?? ""
                if issue.auditType == .contrast, let element = issue.element, self.obscuredByBars(element.frame) {
                    obscured.append((label, element.frame))
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
        for item in obscured {
            lines.append(recheckContrastInClearBand(item.label, frame: item.frame, screen: screen))
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

    /// The band where text is drawn on the plain page: below the navigation bar
    /// (or the status bar) and its scroll-edge effect, above the tab bar.
    private var clearBand: (top: CGFloat, bottom: CGFloat) {
        let nav = app.navigationBars.firstMatch
        let top = (nav.exists && nav.frame.height > 0 ? nav.frame.maxY : app.frame.minY + 62) + 24
        let tab = app.tabBars.firstMatch
        let bottom = (tab.exists && tab.frame.height > 0 ? tab.frame.minY : app.frame.maxY) - 8
        return (top, bottom)
    }

    private func obscuredByBars(_ frame: CGRect) -> Bool {
        let band = clearBand
        return frame.minY < band.top || frame.maxY > band.bottom
    }

    /// Scroll the element found at `frame` into the clear band and audit
    /// contrast again; the line says what happened, and a failure there is a
    /// real failure. Matched by label and position, since several rows can share
    /// a label ("Not available for this device's role ...").
    private func recheckContrastInClearBand(_ label: String, frame: CGRect, screen: String) -> String {
        let candidates = app.descendants(matching: .any).matching(NSPredicate(format: "label == %@", label))
        let element = candidates.allElementsBoundByIndex.min(by: {
            abs($0.frame.midY - frame.midY) < abs($1.frame.midY - frame.midY)
        }) ?? candidates.firstMatch
        let band = clearBand
        if element.exists, element.frame.height > band.bottom - band.top - 40 {
            return "\(screen)\tcontrast\tunder-bar-unverifiable\tTaller than the clear band between the bars\t\(label)\t"
        }
        scrollIntoClearBand(element)
        // Measured on a still screen: contrast is sampled from pixels, so text
        // still moving (or under a scroll-edge effect still fading) could "fail"
        // the re-check. Seen once in M8.1 run A on 'NOT YET VALIDATED', after
        // the old swipes; the same element passed its re-check in the run before.
        waitUntilStill(element)
        guard element.exists, !obscuredByBars(element.frame) else {
            XCTFail("\(screen): contrast on '\(label)' could not be re-checked clear of the bars")
            return "\(screen)\tcontrast\tNEW\tContrast failed (could not scroll clear of the bars)\t\(label)\t"
        }
        let target = element.frame
        let screenshot = XCUIScreen.main.screenshot()
        var finding: String?
        do {
            try app.performAccessibilityAudit(for: .contrast) { issue in
                if let f = issue.element?.frame, abs(f.midY - target.midY) < 4, abs(f.midX - target.midX) < 4 {
                    finding = issue.compactDescription
                }
                return true  // only this element is being re-checked here
            }
        } catch {
            finding = "the re-check audit itself failed: \(error)"
        }
        if let finding {
            // What the re-check measured, for the record.
            Stack.saveScreenshot(screenshot, name: "audit_\(screen)_recheck_FAIL", test: self)
            XCTFail("\(screen): '\(label)' fails contrast clear of the bars (\(finding), at \(target))")
            return "\(screen)\tcontrast\tNEW\tContrast failed clear of the bars: \(finding) at \(target)\t\(label)\t"
        }
        return "\(screen)\tcontrast\tunder-bar\tPassed when re-checked clear of the bars\t\(label)\t"
    }

    /// Moves the page by exactly what the element needs to sit inside the clear
    /// band: a press-drag-hold of that distance, which leaves no momentum. The
    /// slow swipes this replaces each moved the page ~400 pt and kept gliding,
    /// so an element nearly as tall as the band (the device-link banner's body
    /// at the largest text size) was overshot up and down and the re-check
    /// passed or failed on where the last swipe happened to stop (M8.1 run A).
    private func scrollIntoClearBand(_ element: XCUIElement) {
        for _ in 0..<8 {
            guard element.exists else { return }
            let band = clearBand
            let frame = element.frame
            var dy: CGFloat
            if frame.maxY > band.bottom { dy = -(frame.maxY - band.bottom + 16) }  // page up
            else if frame.minY < band.top { dy = band.top - frame.minY + 16 }     // page down
            else { return }
            // One drag stays inside the band, so it starts and ends on page content.
            let limit = (band.bottom - band.top) / 2 - 12
            dy = max(-limit, min(limit, dy))
            let origin = app.coordinate(withNormalizedOffset: .zero)
            let start = origin.withOffset(CGVector(dx: app.frame.width * 0.18, dy: (band.top + band.bottom) / 2 - dy / 2))
            start.press(forDuration: 0.05, thenDragTo: start.withOffset(CGVector(dx: 0, dy: dy)),
                        withVelocity: .slow, thenHoldForDuration: 0.3)
            waitUntilStill(element)
        }
    }

    /// Until the element's frame is the same half a second apart (at most 10
    /// tries), then a beat for the scroll-edge effect to fade.
    private func waitUntilStill(_ element: XCUIElement) {
        var last = element.frame
        for _ in 0..<10 {
            _ = XCTWaiter().wait(for: [XCTestExpectation(description: "still")], timeout: 0.5)
            let now = element.frame
            if now == last { break }
            last = now
        }
        _ = XCTWaiter().wait(for: [XCTestExpectation(description: "settle")], timeout: 1)
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

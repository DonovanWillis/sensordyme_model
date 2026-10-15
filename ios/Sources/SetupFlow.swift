import Combine
import Foundation
import Network
import SwiftUI

/// Drives setup from "where is my device" to "it is online", across both radios.
///
/// The two halves already existed and never spoke: `BLEProvisioner` handed the
/// device its credentials and stopped, and `DeviceLink` could find a device on
/// Wi-Fi but nothing told it to start looking. So the app went quiet at exactly
/// the moment the device was doing the thing it had been asked to do, and the
/// person watching had to work out for themselves that setup had worked.
///
/// This owns both and carries the handover. It exists because the interesting
/// part of provisioning is not either radio — it is the seam between them.
@MainActor
final class SetupFlow: ObservableObject {
    enum Phase: Equatable {
        case bluetoothUnavailable
        case findingDevice
        case connecting(name: String)
        case choosingNetwork
        case sending(ssid: String)
        case deviceJoining(ssid: String)
        case rediscovering(ssid: String, elapsed: Int)
        case ready(ssid: String, url: String?)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .findingDevice
    /// The network the device was told to join, kept so every later screen can
    /// name it rather than saying "your network".
    @Published private(set) var targetSSID: String = ""

    let ble = BLEProvisioner()
    let link = DeviceLink()

    private var observers: [AnyCancellable] = []
    private var rediscoverTask: Task<Void, Never>?
    private var searchClock: Task<Void, Never>?

    /// How long to keep looking for the device on its new network.
    ///
    /// Generous on purpose. The device has to associate, complete a handshake,
    /// get an address by DHCP, and re-announce itself over mDNS — and the phone
    /// often has to move networks too. Giving up early would report failure for
    /// a setup that was about to succeed, which is the more damaging mistake:
    /// the device really is on the new network, and the app has just told
    /// someone it is not.
    private let rediscoverBudget = 90

    // MARK: Lifecycle

    func start() {
        phase = .findingDevice
        link.start()
        ble.startScan()
        observeBLE()
        startSearchClock()
    }

    func stop() {
        rediscoverTask?.cancel()
        rediscoverTask = nil
        searchClock?.cancel()
        searchClock = nil
        ble.stop()
        // Its path monitor would otherwise outlive the setup sheet.
        link.stop()
    }

    func connect(to device: DiscoveredDevice) {
        phase = .connecting(name: device.name)
        ble.connect(id: device.id)
    }

    func send(ssid: String, password: String) {
        targetSSID = ssid
        phase = .sending(ssid: ssid)
        ble.send(ssid: ssid, password: password)
    }

    /// Start over from discovery, keeping nothing.
    func retry() {
        rediscoverTask?.cancel()
        rediscoverTask = nil
        ble.startScan()
        phase = .findingDevice
        startSearchClock()
    }

    // MARK: Bridging the two radios

    private func observeBLE() {
        // Once: start() can run again, and every extra sink re-ran the
        // absorption below for each change.
        guard observers.isEmpty else { return }
        // Mirror the Bluetooth state machine into setup phases, rather than
        // duplicating it. The one thing added here is what happens *after*
        // success, which Bluetooth has no opinion about.
        observers.append(ble.objectWillChange.sink { [weak self] in
            guard let self else { return }
            // Forward the change on. SwiftUI does NOT propagate through nested
            // ObservableObjects: the view observes this flow, so a change to
            // `ble.discovered` reaches nobody unless it is republished here.
            //
            // Without this line the app found the device every time and simply
            // never redrew — it sat on "Searching for SensorDyme" forever with
            // the device sitting in `ble.discovered` the whole time. The status
            // quo was invisible rather than broken, which is why it survived a
            // clean build and a passing test suite.
            self.objectWillChange.send()
            Task { @MainActor in self.absorbBLEState() }
        })
        observers.append(link.objectWillChange.sink { [weak self] in
            self?.objectWillChange.send()
        })
    }

    private func absorbBLEState() {
        switch ble.state {
        case .bluetoothOff:
            // Its own phase, not "searching".
            //
            // Denied permission and a switched-off radio both produce exactly
            // the symptom of a device that is not there, and the app used to
            // show the same endless "Looking for your SensorDyme" for all
            // three. Two of those are fixed in Settings in five seconds, but
            // only if the app says so.
            if isBeforeSending { phase = .bluetoothUnavailable }
        case .idle, .scanning:
            if case .findingDevice = phase {} else if isBeforeSending { phase = .findingDevice }
        case .connecting(let name):
            phase = .connecting(name: name)
        case .ready:
            if isBeforeSending { phase = .choosingNetwork }
        case .sending:
            phase = .sending(ssid: targetSSID)
        case .deviceConnecting:
            phase = .deviceJoining(ssid: targetSSID)
        case .success:
            // The device has joined. From here Bluetooth has nothing more to
            // say — the device is on a different network from the one this
            // conversation started on, and may be on a different one from the
            // phone. Everything after this is Wi-Fi's problem.
            beginRediscovery()
        case .failed(let reason):
            phase = .failed(reason)
        }
    }

    private var isBeforeSending: Bool {
        switch phase {
        case .bluetoothUnavailable, .findingDevice, .connecting, .choosingNetwork: true
        default: false
        }
    }

    /// How long discovery has been running, so the UI can offer help rather
    /// than spin indefinitely. A spinner that never changes tells someone
    /// nothing about whether to keep waiting or go and check something.
    @Published private(set) var searchingSeconds = 0

    private func startSearchClock() {
        searchClock?.cancel()
        searchingSeconds = 0
        searchClock = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                guard let self else { return }
                guard case .findingDevice = phase else { continue }
                searchingSeconds += 1
            }
        }
    }

    private func beginRediscovery() {
        guard rediscoverTask == nil else { return }
        // Which device was just provisioned, read off Bluetooth before the link
        // is dropped. Without it this went looking for whatever camera was
        // linked BEFORE (setting up a second camera "verified" the first one,
        // at the first one's address) or adopted the first camera to answer.
        let provisionedID = ble.deviceID
        // Bluetooth has done its job; holding the link open keeps the device's
        // radio busy and stops it settling on the network it just joined.
        ble.stop()

        rediscoverTask = Task { [weak self] in
            guard let self else { return }
            let started = Date()
            while !Task.isCancelled {
                let elapsed = Int(Date().timeIntervalSince(started))
                if elapsed >= rediscoverBudget { break }
                phase = .rediscovering(ssid: targetSSID, elapsed: elapsed)

                // Firmware that reports no id falls back to the general rule
                // (the linked camera, else one this account owns, else a lone one).
                await link.resolve(force: true, expecting: provisionedID)
                if case let .connected(url) = link.state {
                    phase = .ready(ssid: targetSSID, url: url)
                    rediscoverTask = nil
                    return
                }
                try? await Task.sleep(for: .seconds(3))
            }
            guard !Task.isCancelled else { return }
            // Not found is not the same as not working. The commonest cause by
            // far is that the phone is still on the old network — mDNS does not
            // cross networks, so the device can be perfectly online and
            // invisible from here.
            // A knowable reason beats a guess. Telling someone to check their
            // Wi-Fi when the real cause is a denied Local Network permission
            // sends them round a loop that can never succeed.
            phase = .failed(
                link.discoveryProblem
                ?? "Your SensorDyme was given the details for \(targetSSID) and said it joined, "
                    + "but this phone can't see it yet. Check your phone is on \(targetSSID) too, "
                    + "then search again."
            )
            rediscoverTask = nil
        }
    }
}

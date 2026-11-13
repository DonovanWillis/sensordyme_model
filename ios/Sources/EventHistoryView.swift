import SwiftUI

/// A device's event history — what used to be the "Recent activity" card on the
/// device screen, given its own page so the device screen can stay about the
/// camera and what it is watching for.
///
/// Same data, same endpoint (`GET /v1/devices/{id}/events`), same detail sheet
/// (EventDetailView, with "why did this alert fire"). Nothing about how events
/// are produced, stored or notified changed; only where the list lives.
struct EventHistoryView: View {
    @EnvironmentObject private var state: AppState
    let device: Device
    /// Whatever the device screen already had, so this opens populated rather
    /// than blank while its own fetch is in flight.
    let initialEvents: [EventItem]

    @State private var events: [EventItem] = []
    @State private var hasLoaded = false
    @State private var errorMessage: String?
    @State private var selectedEvent: EventItem?
    @State private var filter: Filter = .all
    @State private var reloadGuard = LatestRequestGuard()

    enum Filter: String, CaseIterable, Identifiable {
        case all, alerts
        var id: String { rawValue }
        var title: String { self == .all ? "All" : "Alerts" }
    }

    /// "Alerts" uses the same test the device screen and the notification layer
    /// use, so the three can never disagree about what counted as an alert.
    private var visible: [EventItem] {
        switch filter {
        case .all: events
        case .alerts: events.filter { $0.isCritical || $0.isConfigured }
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: SD.Space.lg) {
                if let errorMessage {
                    ConnectionBanner(message: errorMessage) { await reload() }
                }

                // The shape of today, on open space: when things happened and
                // how serious they were. Each mark opens its event.
                if !events.isEmpty {
                    EventTimeline(events: events) { selectedEvent = $0 }
                }

                Picker("Show", selection: $filter) {
                    ForEach(Filter.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)

                list
            }
            .padding(.horizontal, SD.Space.md)
            .padding(.top, SD.Space.xs)
            .padding(.bottom, SD.Space.xl)
        }
        .orynScreen()
        .navigationTitle("Event history")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $selectedEvent) { event in
            EventDetailView(device: device, event: event, allEvents: events)
        }
        .refreshable { await reload() }
        .task {
            if events.isEmpty { events = EventFeed.ordered(initialEvents) }
            await reload()
        }
    }

    @ViewBuilder private var list: some View {
        if visible.isEmpty {
            if !hasLoaded && events.isEmpty {
                SDLoadingState(message: "Loading events…")
            } else {
                SDEmptyState(
                    icon: filter == .alerts ? "bell.slash" : "clock",
                    title: filter == .alerts ? "No alerts" : "Nothing to report",
                    message: "Events appear here as soon as \(device.name) sees something worth telling you about."
                )
            }
        } else {
            // A chronological flow grouped by day; a row opens in place and
            // leads to "why this happened".
            ActivityTimeline(events: visible) { selectedEvent = $0 }
        }
    }

    private func reload() async {
        let token = reloadGuard.begin()
        do {
            let fetched = try await state.backend.events(deviceID: device.id, limit: 200)
            guard reloadGuard.isLatest(token) else { return }
            // Merged, not replaced (M8.3): one copy per event, a newer alert
            // report kept over a stale one, ordered by when it happened.
            events = EventFeed.merge(events, fetched)
            errorMessage = nil
        } catch let error as APIError where error.isAuthFailure {
            guard reloadGuard.isLatest(token) else { return }
            state.expireSession(error.message)
        } catch {
            guard reloadGuard.isLatest(token) else { return }
            errorMessage = error.localizedDescription
        }
        if reloadGuard.isLatest(token) { hasLoaded = true }
    }
}

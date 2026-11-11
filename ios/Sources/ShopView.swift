import SwiftUI

// MARK: - Shop: request more devices, track requests.
//
// There is no payment in the app. Placing an order records an ORDER REQUEST on
// the server (POST /v1/orders); nothing is charged and nothing ships until
// someone follows up. The screen says exactly that.

struct ShopView: View {
    @EnvironmentObject private var state: AppState
    @State private var quantity = 1
    @State private var address = ""
    @State private var placing = false
    @State private var placedOrderID: Int?
    @State private var orderHistory: [Order] = []
    @State private var errorMessage: String?
    /// The same race as DeviceDetailView/DevicesView: `.task` (on appear),
    /// `.refreshable` (pull-to-refresh), and a successful placeOrder() all
    /// call reloadOrders() independently, with no guarantee their network
    /// responses land in the order they were sent. See LatestRequestGuard's
    /// own docs — found by an explicit sweep of the codebase for this exact
    /// pattern after fixing it on the two screens where it was first reported.
    @State private var reloadGuard = LatestRequestGuard()

    private let unitPrice = 199

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    productCard
                    orderForm
                    if !orderHistory.isEmpty {
                        ordersCard
                    }
                }
                .padding()
            }
            .orynScreen()
            .navigationTitle("Shop")
            .task { await reloadOrders() }
            .refreshable { await reloadOrders() }
        }
    }

    private var productCard: some View {
        VStack(spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous).fill(SD.Colors.canvas)
                RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous).strokeBorder(SD.Colors.borderStrong, lineWidth: 1)
                CameraGrid().stroke(SD.Colors.border, lineWidth: 1)
                    .clipShape(RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous))
                VStack(spacing: 10) {
                    OrynMark(height: 54)
                    Text("SensorDyme Unit")
                        .font(SD.Typography.title)
                        .foregroundStyle(SD.Colors.textPrimary)
                    Text("IR night vision · edge AI · every mode")
                        .sdEyebrowFont()
                        .kerning(1)
                        .multilineTextAlignment(.center)
                        .foregroundStyle(SD.Colors.textTertiary)
                        .padding(.horizontal, SD.Space.md)
                }
                .padding(.vertical, 30)
            }
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("$\(unitPrice)").font(.title2.weight(.heavy))
                    Text("No payment in the app").font(.caption).foregroundStyle(SD.Colors.textSecondary)
                }
                Spacer()
                Stepper(value: $quantity, in: 1...25) {
                    Text("Qty \(quantity)").font(.headline).monospacedDigit()
                }
                .fixedSize()
            }
        }
        .brandCard()
    }

    private var orderForm: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Ship to").font(.headline)
            TextField("Full shipping address", text: $address, axis: .vertical)
                .lineLimit(2...4)
                .padding(12)
                .background(RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous).fill(SD.Colors.field))
                .overlay(RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous).strokeBorder(SD.Colors.borderStrong))

            Button {
                placeOrder()
            } label: {
                if placing {
                    ProgressView().tint(SD.Colors.textOnAccent)
                } else {
                    Text("Send order request · est. $\(unitPrice * quantity)")
                }
            }
            .buttonStyle(BrandButtonStyle())
            .disabled(placing || address.trimmingCharacters(in: .whitespaces).count < 10)

            Text("This sends an order request. You won't be charged here; we'll contact you to confirm the order and arrange payment before anything ships.")
                .font(.caption)
                .foregroundStyle(SD.Colors.textTertiary)
                .fixedSize(horizontal: false, vertical: true)

            if let placedOrderID {
                Label("Request #\(placedOrderID) received — not yet confirmed or charged",
                      systemImage: "checkmark.circle")
                    .font(.subheadline.bold())
                    .foregroundStyle(SD.Colors.success)
            }
            if let errorMessage {
                Text(errorMessage).font(.callout).foregroundStyle(SD.Colors.danger)
            }
        }
        .brandCard()
    }

    private var ordersCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Your order requests").font(.headline)
            ForEach(orderHistory) { order in
                HStack {
                    Image(systemName: "shippingbox.fill").foregroundStyle(SD.Colors.accent)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Order #\(order.id) — \(order.quantity) unit\(order.quantity > 1 ? "s" : "")")
                            .font(.subheadline.bold())
                        Text(order.address).font(.caption).foregroundStyle(SD.Colors.textSecondary).lineLimit(1)
                    }
                    Spacer()
                    Text(Self.statusLabel(order.status))
                        .font(.caption.bold())
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .background(Capsule().fill(.thinMaterial))
                }
                .padding(.vertical, 4)
            }
        }
        .brandCard()
    }

    /// "placed" is the server's word for a request nobody has acted on yet.
    static func statusLabel(_ status: String) -> String {
        status == "placed" ? "Requested" : status.replacingOccurrences(of: "_", with: " ").capitalized
    }

    private func placeOrder() {
        placing = true
        errorMessage = nil
        Task {
            defer { placing = false }
            do {
                placedOrderID = try await state.backend.placeOrder(
                    quantity: quantity, address: address.trimmingCharacters(in: .whitespaces)
                )
                await reloadOrders()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func reloadOrders() async {
        // A transient failure here (`try?` … `?? []`) used to overwrite
        // orderHistory with an empty list unconditionally — a real order
        // history replaced by "no orders" on a one-off network blip, instead
        // of leaving the stale-but-correct data on screen the way
        // DevicesView/DeviceDetailView's own reload() already does for
        // exactly this reason.
        let token = reloadGuard.begin()
        do {
            let fetched = try await state.backend.orders()
            guard reloadGuard.isLatest(token) else { return }
            orderHistory = fetched
            errorMessage = nil
        } catch {
            guard reloadGuard.isLatest(token) else { return }
            errorMessage = error.localizedDescription
        }
    }
}

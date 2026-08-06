import SwiftUI

// MARK: - Shop: order more devices, track orders.

struct ShopView: View {
    @EnvironmentObject private var state: AppState
    @State private var quantity = 1
    @State private var address = ""
    @State private var placing = false
    @State private var placedOrderID: Int?
    @State private var orderHistory: [Order] = []
    @State private var errorMessage: String?

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
            .navigationTitle("Shop")
            .task { await reloadOrders() }
            .refreshable { await reloadOrders() }
        }
    }

    private var productCard: some View {
        VStack(spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 24).fill(SD.Colors.accent)
                VStack(spacing: 10) {
                    LogoMark(size: 66)
                        .colorScheme(.dark)
                    Text("SensorDyme Unit")
                        .font(.title3.bold())
                        .foregroundStyle(.white)
                    Text("IR night vision · edge AI · every mode included")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.85))
                }
                .padding(.vertical, 30)
            }
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("$\(unitPrice)").font(.title2.weight(.heavy))
                    Text("free shipping · 30-day returns").font(.caption).foregroundStyle(SD.Colors.textSecondary)
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
                .background(RoundedRectangle(cornerRadius: 12).fill(.background.secondary))
                .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary))

            Button {
                placeOrder()
            } label: {
                if placing {
                    ProgressView().tint(.white)
                } else {
                    Text("Place order — $\(unitPrice * quantity)")
                }
            }
            .buttonStyle(BrandButtonStyle())
            .disabled(placing || address.trimmingCharacters(in: .whitespaces).count < 10)

            if let placedOrderID {
                Label("Order #\(placedOrderID) placed — \(quantity) unit\(quantity > 1 ? "s" : "") on the way",
                      systemImage: "shippingbox.fill")
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
            Text("Your orders").font(.headline)
            ForEach(orderHistory) { order in
                HStack {
                    Image(systemName: "shippingbox.fill").foregroundStyle(SD.Colors.accent)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Order #\(order.id) — \(order.quantity) unit\(order.quantity > 1 ? "s" : "")")
                            .font(.subheadline.bold())
                        Text(order.address).font(.caption).foregroundStyle(SD.Colors.textSecondary).lineLimit(1)
                    }
                    Spacer()
                    Text(order.status.capitalized)
                        .font(.caption.bold())
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .background(Capsule().fill(.thinMaterial))
                }
                .padding(.vertical, 4)
            }
        }
        .brandCard()
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
        orderHistory = (try? await state.backend.orders()) ?? []
    }
}

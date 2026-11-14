import Foundation

// Standalone, run by ios/Tests/run.sh (concatenated with ios/Sources/AccountDeletion.swift).
//
// M8.4: deleting the account says what is and is not deleted (docs/PRIVACY.md
// §5), can only be sent with the password, and every refusal says nothing was
// deleted.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

print("what is and is not deleted is said, in the privacy document's terms:")
check(AccountDeletion.deleted.contains { $0.contains("event") }, "events")
check(AccountDeletion.deleted.contains { $0.contains("order") }, "orders")
check(AccountDeletion.released.joined().contains("released"), "devices are released, not deleted")
check(AccountDeletion.notDeleted.contains { $0.contains("on each device itself") && $0.contains("aren't wiped") },
      "the device's own copies are not wiped remotely, said plainly")
check(AccountDeletion.confirmMessage.contains("can't be undone"), "the confirmation says it is final")

print("\nthe form needs the password:")
check(!AccountDeletion.canSubmit(password: "", deleting: false), "no password, no request")
check(AccountDeletion.canSubmit(password: "password1", deleting: false), "with a password")
check(!AccountDeletion.canSubmit(password: "password1", deleting: true), "not twice while one is in flight")

print("\nevery refusal says nothing was deleted:")
let refusals = [
    AccountDeletion.message(status: 403, code: "wrong_password", detail: "the password is not correct"),
    AccountDeletion.message(status: 400, code: "confirmation_required", detail: ""),
    AccountDeletion.message(status: 429, code: nil, detail: ""),
    AccountDeletion.message(status: 0, code: nil, detail: "The Internet connection appears to be offline."),
    AccountDeletion.message(status: 404, code: nil, detail: "Not Found"),
    AccountDeletion.message(status: 500, code: nil, detail: "boom"),
]
check(refusals.allSatisfy { $0.contains("Nothing was deleted") || $0.contains("nothing was deleted") },
      "all of them")
check(refusals[0].hasPrefix("That password isn't right"), "wrong password, in words")
check(refusals[1].contains("confirmation"), "confirmation required, in words")

print("\nthe server's counts, after:")
check(AccountDeletion.summary(devicesReleased: 1, eventsDeleted: 12)
      == "Your account was deleted. 12 events deleted. 1 device released.", "counted, as the server reported")
check(AccountDeletion.summary(devicesReleased: nil, eventsDeleted: nil) == "Your account was deleted.",
      "nothing invented when the server sends no counts")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)

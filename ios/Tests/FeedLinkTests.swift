import Foundation

// Standalone, run by ios/Tests/run.sh (concatenated with ios/Sources/FeedLink.swift).

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

print("a device shows the camera that is actually it:")
check(FeedLink.showsLiveFeed(deviceHardwareID: "SD-3D248632", linkedID: "SD-3D248632", accountDeviceCount: 2),
      "matching hardware id streams")
check(!FeedLink.showsLiveFeed(deviceHardwareID: "SD-AAAAAAAA", linkedID: "SD-3D248632", accountDeviceCount: 2),
      "the OTHER device on the account does not show this camera (the bug)")
check(!FeedLink.showsLiveFeed(deviceHardwareID: "SD-AAAAAAAA", linkedID: "SD-3D248632", accountDeviceCount: 1),
      "even a lone device does not show a camera that is provably someone else")
check(FeedLink.showsLiveFeed(deviceHardwareID: "sd-3d248632", linkedID: "SD-3D248632", accountDeviceCount: 2),
      "ids compare case-insensitively")

print("\nwhen one side is unknown, only an unambiguous account streams:")
check(FeedLink.showsLiveFeed(deviceHardwareID: nil, linkedID: "SD-3D248632", accountDeviceCount: 1),
      "single device on old firmware: unchanged behaviour")
check(FeedLink.showsLiveFeed(deviceHardwareID: "SD-3D248632", linkedID: "SensorDyme on sensordyme", accountDeviceCount: 1),
      "single device with a legacy stored service name: unchanged behaviour")
check(!FeedLink.showsLiveFeed(deviceHardwareID: nil, linkedID: "SD-3D248632", accountDeviceCount: 2),
      "several devices and no id: refuse to guess")
check(!FeedLink.showsLiveFeed(deviceHardwareID: nil, linkedID: "", accountDeviceCount: 0),
      "nothing known: nothing shown")

// ------------------------------------------------- which camera to link (F4)
print("\nchoosing a camera out of what Bonjour found:")
let mine = "SD-AAAA1111", neighbour = "SD-BBBB2222", newOne = "SD-CCCC3333"
check(FeedLink.chooseCamera(found: [neighbour, mine], linked: mine) == mine,
      "the linked camera is kept, wherever it is in the list")
check(FeedLink.chooseCamera(found: [neighbour], linked: mine) == nil,
      "a linked camera that is absent is NOT replaced by whatever answered")
check(FeedLink.chooseCamera(found: [neighbour, mine], linked: "") == nil,
      "nothing linked, several found, no account ids: refuse rather than take the first")
check(FeedLink.chooseCamera(found: [mine], linked: "") == mine,
      "nothing linked and one camera: that one")
check(FeedLink.chooseCamera(found: [neighbour, mine], linked: "", accountHardwareIDs: [mine]) == mine,
      "nothing linked: the camera on THIS account wins over a neighbour that answered first")
check(FeedLink.chooseCamera(found: [neighbour], linked: "", accountHardwareIDs: [mine]) == nil,
      "a lone camera that is provably someone else's is not adopted")
check(FeedLink.chooseCamera(found: ["SensorDyme on pi"], linked: "SensorDyme on pi") == "SensorDyme on pi",
      "a legacy service name still matches itself")
check(FeedLink.chooseCamera(found: [mine], linked: "SensorDyme on pi") == mine,
      "a legacy name migrates to the only camera present")
check(FeedLink.chooseCamera(found: [mine, neighbour], linked: "SensorDyme on pi") == nil,
      "but never by guessing between several")

print("\nsetup looks for the device it just provisioned, and only that one:")
check(FeedLink.chooseCamera(found: [mine, newOne], linked: mine, expecting: newOne) == newOne,
      "a second camera is found as itself, not as the one linked before (the bug)")
check(FeedLink.chooseCamera(found: [mine], linked: mine, expecting: newOne) == nil,
      "and the old camera answering is not mistaken for the new one being ready")
check(FeedLink.chooseCamera(found: ["sd-cccc3333"], linked: "", expecting: newOne) == "sd-cccc3333",
      "ids compare case-insensitively")

// ------------------------------------------- an address is not an identity (F10)
print("\na cached address must still be the linked camera:")
check(FeedLink.cachedAddressIsLinkedCamera(reportedID: mine, linkedID: mine), "same camera: fine")
check(!FeedLink.cachedAddressIsLinkedCamera(reportedID: neighbour, linkedID: mine),
      "another camera answering at the cached address is rejected (DHCP swap / name swap)")
check(FeedLink.cachedAddressIsLinkedCamera(reportedID: nil, linkedID: mine),
      "no identity reported (no token): cannot contradict, so not rejected")
check(FeedLink.cachedAddressIsLinkedCamera(reportedID: neighbour, linkedID: ""),
      "nothing linked: nothing to contradict")

// --------------------------------------------- which backend device is "yours" (F13)
print("\nthe banner describes the linked camera's own backend record:")
check(FeedLink.linkedDeviceIndex(hardwareIDs: [neighbour, mine], linkedID: mine) == 1,
      "the device whose hardware id is the linked camera — not devices.first")
check(FeedLink.linkedDeviceIndex(hardwareIDs: [nil], linkedID: mine) == 0,
      "a single device is unambiguous")
check(FeedLink.linkedDeviceIndex(hardwareIDs: [nil, nil], linkedID: mine) == nil,
      "several devices and no matching id: ambiguous, not the first one")
check(FeedLink.linkedDeviceIndex(hardwareIDs: [], linkedID: mine) == nil, "no devices: none")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)

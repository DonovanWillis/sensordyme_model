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

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)

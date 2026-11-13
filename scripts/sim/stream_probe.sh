#!/bin/bash
# Run the iOS app's Foundation-only live-feed logic (parser, liveness tracker,
# request builder, shared-connection coordinator) against a running device
# dashboard -- normally the virtual device: SIMULATOR evidence, not a phone.
#
#   scripts/sim/stream_probe.sh http://127.0.0.1:8899 DEVICE_TOKEN out.json
set -euo pipefail
cd "$(dirname "$0")/../.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cat ios/Sources/StreamLiveness.swift ios/Sources/MJPEGStream.swift scripts/sim/stream_probe.swift > "$tmp/main.swift"
swift "$tmp/main.swift" "$@"

#!/bin/bash
# Run the Swift logic tests.
#
# The app has no XCTest target, and adding one is a bigger change than the
# handful of pure functions worth testing justifies. `swift a.swift b.swift`
# looks like it works and does not: with more than one file, Swift builds a
# module and top-level code never executes, so the whole suite "passes" in
# silence. Concatenating into one main.swift is what actually runs it.
set -euo pipefail
cd "$(dirname "$0")/../.."

status=0
for test in ios/Tests/*Tests.swift; do
  name=$(basename "$test" .swift)
  subject="ios/Sources/${name%Tests}.swift"
  [ -f "$subject" ] || { echo "no subject for $name"; status=1; continue; }
  tmp=$(mktemp -d)
  cat "$subject" "$test" > "$tmp/main.swift"
  echo "== $name =="
  swift "$tmp/main.swift" || status=1
  rm -rf "$tmp"
done
exit $status

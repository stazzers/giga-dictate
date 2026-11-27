#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DESTINATION="$ROOT/build/python"
if [ -x "$DESTINATION/bin/python3.11" ] && [ -f "$DESTINATION/lib/libpython3.11.dylib" ]; then
    echo "Portable CPython is ready at $DESTINATION"
    exit 0
fi
if [ -e "$DESTINATION" ]; then
    echo "Choose a clean build directory or move the incomplete runtime at $DESTINATION." >&2
    exit 1
fi

mkdir -p "$ROOT/build"
WORK=$(mktemp -d "$ROOT/build/.python-runtime.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
URL='https://github.com/astral-sh/python-build-standalone/releases/download/20261124/cpython-3.11.18%2B20261124-aarch64-apple-darwin-install_only.tar.gz'
SHA256='89ce773f5eb6f072ceea3a30d86258ecd5d02ea0f8e87fda0b940b9067aebf81'
curl --fail --location --retry 2 "$URL" --output "$WORK/python.tar.gz"
printf '%s  %s\n' "$SHA256" "$WORK/python.tar.gz" | shasum -a 256 -c -
tar -xf "$WORK/python.tar.gz" -C "$WORK"
"$WORK/python/bin/python3.11" -c 'import platform, ssl; assert platform.machine() == "arm64"'
mv "$WORK/python" "$DESTINATION"
echo "Portable CPython is ready at $DESTINATION"

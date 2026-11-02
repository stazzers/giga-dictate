#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
APP_NAME="GigaDictate.app"
APP_DIR="$DIR/$APP_NAME"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

echo "Building $APP_NAME in $DIR..."

rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Copy Icon
if [ -f "$DIR/AppIcon.icns" ]; then
    cp "$DIR/AppIcon.icns" "$RESOURCES_DIR/AppIcon.icns"
fi

# Create Info.plist (identical structure to SuperDictate)
cat << 'EOF' > "$CONTENTS_DIR/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>ru</string>
    <key>CFBundleDisplayName</key>
    <string>GigaDictate</string>
    <key>CFBundleExecutable</key>
    <string>GigaDictate</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.salute.gigadictate</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>GigaDictate</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppleEventsUsageDescription</key>
    <string>GigaDictate использует события ввода для вставки распознанного текста под курсор.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>GigaDictate записывает аудио при активации горячей клавиши и распознает речь локально с помощью GigaAM-v3.</string>
</dict>
</plist>
EOF

# Compile native Mach-O launcher binary with embedded Python runtime
PY_INCLUDES=$(/opt/homebrew/opt/python@3.11/bin/python3.11-config --includes)
PY_LDFLAGS=$(/opt/homebrew/opt/python@3.11/bin/python3.11-config --ldflags --embed)
clang -arch arm64 -O2 $PY_INCLUDES "$DIR/launcher.c" $PY_LDFLAGS -o "$MACOS_DIR/GigaDictate"
chmod +x "$MACOS_DIR/GigaDictate"

# Ad-hoc code sign the application bundle with permanent designated requirement and microphone entitlements
codesign -s - --force --deep --entitlements "$DIR/entitlements.plist" -r='designated => identifier "com.salute.gigadictate"' "$APP_DIR"

echo "GigaDictate.app built and signed successfully at: $APP_DIR"

# Install into /Applications
echo "Installing into /Applications/GigaDictate.app..."
rm -rf "/Applications/GigaDictate.app"
cp -R "$APP_DIR" "/Applications/GigaDictate.app"
codesign -s - --force --deep --entitlements "$DIR/entitlements.plist" -r='designated => identifier "com.salute.gigadictate"' "/Applications/GigaDictate.app"

# Stamp white-toned icon directly into Finder metadata and flush LaunchServices cache
"$DIR/.venv/bin/python" -c "
import os, subprocess
from AppKit import NSWorkspace, NSImage

dir_path = '$DIR'
icns_path = os.path.join(dir_path, 'AppIcon.icns')
if os.path.exists(icns_path):
    img = NSImage.alloc().initWithContentsOfFile_(icns_path)
    for app in ['$APP_DIR', '/Applications/GigaDictate.app']:
        if os.path.exists(app):
            NSWorkspace.sharedWorkspace().setIcon_forFile_options_(img, app, 0)
            subprocess.run(['touch', app])

lsreg = '/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister'
if os.path.exists(lsreg):
    subprocess.run([lsreg, '-f', '/Applications/GigaDictate.app'])
"

# Refresh Finder and Dock icon cache
killall Finder 2>/dev/null || true

echo "Installed successfully in /Applications/GigaDictate.app with updated white-toned icon!"

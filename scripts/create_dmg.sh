#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
APP_NAME="GigaDictate.app"
APP_DIR="$DIR/$APP_NAME"
FINAL_DMG="$DIR/GigaDictate-arm64.dmg"
BUILD_DIR="$DIR/build"
DMG_WORK_DIR=""
DEV_NODE=""
VOLUME_NAME="GigaDictate"
VOLUME_PATH="/Volumes/$VOLUME_NAME"

cleanup() {
    if [ -n "$DEV_NODE" ]; then
        hdiutil detach "$DEV_NODE" -quiet 2>/dev/null || true
    fi
    if [ -n "$DMG_WORK_DIR" ]; then
        rm -rf "$DMG_WORK_DIR"
    fi
}
trap cleanup EXIT

echo "================================================="
echo " Creating Compact DMG (Big Icons, Pure White BG)"
echo "================================================="

# 1. Ensure GigaDictate.app is built and valid
if [ ! -d "$APP_DIR" ]; then
    echo "GigaDictate.app not found. Building it first..."
    bash "$DIR/scripts/create_app.sh"
fi

# Ensure previous volumes are detached
if [ -d "$VOLUME_PATH" ]; then
    diskutil eject "$VOLUME_PATH" 2>/dev/null || true
    sleep 1
fi

mkdir -p "$BUILD_DIR"
DMG_WORK_DIR="$(mktemp -d "$BUILD_DIR/dmg.XXXXXX")"
TEMP_DMG="$DMG_WORK_DIR/temp_rw.dmg"
OUTPUT_DMG="$DMG_WORK_DIR/GigaDictate-arm64.dmg"

# 2. Generate pure clean white background (no text, no graphics)
"$DIR/.venv/bin/python" -c "
from PIL import Image
img = Image.new('RGB', (1040, 600), (255, 255, 255))
img.save('$DMG_WORK_DIR/background.png', 'PNG')
"

# 3. Create temporary writable DMG
echo "Creating temporary writable DMG image..."
hdiutil create -size 2500m -fs HFS+ -volname "$VOLUME_NAME" -ov "$TEMP_DMG"

# 4. Mount temporary DMG
echo "Mounting temporary DMG..."
ATTACH_INFO=$(hdiutil attach -readwrite -noverify -noautoopen "$TEMP_DMG")
DEV_NODE=$(echo "$ATTACH_INFO" | awk '/\/dev\/disk[0-9]+s[0-9]+/ {print $1; exit}')
if [ -z "$DEV_NODE" ]; then
    DEV_NODE=$(echo "$ATTACH_INFO" | awk '/\/dev\/disk[0-9]+/ {print $1; exit}')
fi

echo "Volume mounted at: $VOLUME_PATH on $DEV_NODE"
sleep 2

# 5. Populate volume: ONLY GigaDictate.app, Applications shortcut, and hidden pure white background
echo "Copying GigaDictate.app into volume..."
cp -R "$APP_DIR" "$VOLUME_PATH/"

echo "Creating Applications shortcut..."
ln -s /Applications "$VOLUME_PATH/Applications"

echo "Adding pure white background..."
mkdir -p "$VOLUME_PATH/.background"
cp "$DMG_WORK_DIR/background.png" "$VOLUME_PATH/.background/background.png"
chflags hidden "$VOLUME_PATH/.background" 2>/dev/null || true

# 6. Configure compact Finder window (large 120px icons, pure white background, App on left, Apps on right)
echo "Configuring Finder window geometry and icon layout..."
# Headless release builds can skip Finder automation and use its default layout.
if [ "${GIGA_DMG_SKIP_FINDER:-0}" != "1" ]; then
osascript << 'EOF'
tell application "Finder"
    tell disk "GigaDictate"
        open
        delay 1
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {350, 180, 870, 480}
        
        set theViewOptions to the icon view options of container window
        set arrangement of theViewOptions to not arranged
        set icon size of theViewOptions to 120
        set text size of theViewOptions to 13
        set background picture of theViewOptions to file ".background:background.png"
        
        set position of item "GigaDictate.app" of container window to {130, 135}
        set position of item "Applications" of container window to {390, 135}
        
        close
        open
        update without registering applications
        delay 2
    end tell
end tell
EOF
fi

# Ensure changes are flushed to disk
echo "Flushing volume changes..."
sync
sleep 2

# 7. Unmount volume
echo "Detaching temporary volume..."
diskutil eject "$VOLUME_PATH" || hdiutil detach "$DEV_NODE"
DEV_NODE=""

# 8. Convert to final read-only, compressed DMG
echo "Compressing into final read-only DMG (UDZO zlib-level 9)..."
hdiutil convert "$TEMP_DMG" -format UDZO -imagekey zlib-level=9 -o "$OUTPUT_DMG"
hdiutil verify "$OUTPUT_DMG"
mv -f "$OUTPUT_DMG" "$FINAL_DMG"

# 9. Clean up temporary files
# The EXIT trap removes this build's workspace, preserving models and backups.

echo "================================================="
echo " Final DMG created successfully at: $FINAL_DMG"
du -sh "$FINAL_DMG"
echo "================================================="

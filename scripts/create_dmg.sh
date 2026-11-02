#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
APP_NAME="GigaDictate.app"
APP_DIR="$DIR/$APP_NAME"
FINAL_DMG="$DIR/GigaDictate-arm64.dmg"
TEMP_DMG="$DIR/build/temp_rw.dmg"
BUILD_DIR="$DIR/build"
MOUNT_DIR="$BUILD_DIR/mnt_gigadictate"

echo "================================================="
echo " Creating Pure Minimalist DMG (App + Applications)"
echo "================================================="

# 1. Ensure GigaDictate.app is built and valid
if [ ! -d "$APP_DIR" ]; then
    echo "GigaDictate.app not found. Building it first..."
    bash "$DIR/scripts/create_app.sh"
fi

rm -rf "$BUILD_DIR" "$FINAL_DMG"
mkdir -p "$BUILD_DIR" "$MOUNT_DIR"

# 2. Create temporary writable DMG
echo "Creating temporary writable DMG image..."
hdiutil create -size 2500m -fs HFS+ -volname "GigaDictate" -ov "$TEMP_DMG"

# 3. Mount temporary DMG at specific mountpoint
echo "Mounting temporary DMG at $MOUNT_DIR..."
hdiutil attach -readwrite -noverify -noautoopen -mountpoint "$MOUNT_DIR" "$TEMP_DMG"
sleep 1

# 4. Populate volume: ONLY GigaDictate.app and Applications shortcut
echo "Copying GigaDictate.app into volume..."
cp -R "$APP_DIR" "$MOUNT_DIR/"

echo "Creating Applications shortcut..."
ln -s /Applications "$MOUNT_DIR/Applications"

# 5. Configure clean minimal Finder window via AppleScript
echo "Configuring clean Finder window layout (App on left, Applications on right)..."
osascript << 'EOF'
tell application "Finder"
    tell disk "GigaDictate"
        open
        delay 1
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {400, 200, 920, 500}
        
        set theViewOptions to the icon view options of container window
        set arrangement of theViewOptions to not arranged
        set icon size of theViewOptions to 110
        set text size of theViewOptions to 13
        
        set position of item "GigaDictate.app" of container window to {130, 140}
        set position of item "Applications" of container window to {390, 140}
        
        close
        open
        update without registering applications
        delay 2
    end tell
end tell
EOF

# Ensure changes are flushed to disk
echo "Flushing volume changes..."
sync
sleep 2

# 6. Unmount volume
echo "Detaching temporary volume..."
hdiutil detach "$MOUNT_DIR"

# 7. Convert to final read-only, compressed DMG
echo "Compressing into final read-only DMG (UDZO zlib-level 9)..."
hdiutil convert "$TEMP_DMG" -format UDZO -imagekey zlib-level=9 -o "$FINAL_DMG"

# 8. Clean up temporary files
rm -rf "$BUILD_DIR"

echo "================================================="
echo " Pure Minimal DMG created successfully: $FINAL_DMG"
du -sh "$FINAL_DMG"
echo "================================================="

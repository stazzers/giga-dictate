#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
APP_NAME="GigaDictate.app"
APP_DIR="$DIR/$APP_NAME"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
FRAMEWORKS_DIR="$CONTENTS_DIR/Frameworks"

echo "================================================="
echo " Building Standalone $APP_NAME (Apple Silicon)  "
echo "================================================="

rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"
mkdir -p "$FRAMEWORKS_DIR"

# 1. Copy Application Icon
if [ -f "$DIR/AppIcon.icns" ]; then
    cp "$DIR/AppIcon.icns" "$RESOURCES_DIR/AppIcon.icns"
    cp "$DIR/AppIcon.icns" "$RESOURCES_DIR/AppIcon"
fi

# 2. Bundle Python Code & Repositories into Resources
echo "Bundling Python modules into Resources..."
cp -R "$DIR/giga_dictate" "$RESOURCES_DIR/giga_dictate"
cp -R "$DIR/GigaAM_repo" "$RESOURCES_DIR/GigaAM_repo"
# Clean .git, tests, and __pycache__ from bundle
rm -rf "$RESOURCES_DIR/GigaAM_repo/.git"
rm -rf "$RESOURCES_DIR/GigaAM_repo/tests"
rm -rf "$RESOURCES_DIR/GigaAM_repo/tutorial"
rm -rf "$RESOURCES_DIR/GigaAM_repo/docs"
find "$RESOURCES_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
chmod -R u+w "$APP_DIR"

# 3. Bundle Python.framework into Frameworks (standalone relocatable runtime)
PYTHON_FW="/opt/homebrew/opt/python@3.11/Frameworks/Python.framework"
if [ -d "$PYTHON_FW" ]; then
    echo "Bundling Python.framework into Frameworks..."
    cp -R "$PYTHON_FW" "$FRAMEWORKS_DIR/Python.framework"
    # Prune unneeded development headers and test suites to optimize size
    rm -rf "$FRAMEWORKS_DIR/Python.framework/Headers"
    rm -rf "$FRAMEWORKS_DIR/Python.framework/Versions/3.11/Headers"
    rm -rf "$FRAMEWORKS_DIR/Python.framework/Versions/3.11/include"
    rm -rf "$FRAMEWORKS_DIR/Python.framework/Versions/3.11/share"
    rm -rf "$FRAMEWORKS_DIR/Python.framework/Versions/3.11/lib/python3.11/test"
    rm -rf "$FRAMEWORKS_DIR/Python.framework/Versions/3.11/lib/python3.11/ensurepip"
    rm -rf "$FRAMEWORKS_DIR/Python.framework/Versions/3.11/lib/python3.11/idlelib"
    rm -rf "$FRAMEWORKS_DIR/Python.framework/Versions/3.11/lib/python3.11/tkinter"
    rm -rf "$FRAMEWORKS_DIR/Python.framework/Versions/3.11/Resources/Python.app"
    rm -rf "$FRAMEWORKS_DIR/Python.framework/Versions/3.11/_CodeSignature"
    rm -f "$FRAMEWORKS_DIR/Python.framework/Versions/3.11/lib/python3.11/site-packages"

    # Bundle runtime dylibs that Python depends on so the app runs without Homebrew
    echo "Bundling standalone runtime dependencies (OpenSSL, SQLite, LZMA, MPDecimal)..."
    cp -f /opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib "$FRAMEWORKS_DIR/"
    cp -f /opt/homebrew/opt/openssl@3/lib/libssl.3.dylib "$FRAMEWORKS_DIR/"
    cp -fL /opt/homebrew/opt/sqlite/lib/libsqlite3.dylib "$FRAMEWORKS_DIR/libsqlite3.dylib"
    cp -fL /opt/homebrew/opt/xz/lib/liblzma.5.dylib "$FRAMEWORKS_DIR/liblzma.5.dylib"
    cp -fL /opt/homebrew/opt/mpdecimal/lib/libmpdec.4.dylib "$FRAMEWORKS_DIR/libmpdec.4.dylib"
    chmod 755 "$FRAMEWORKS_DIR"/*.dylib

    install_name_tool -id @rpath/libcrypto.3.dylib "$FRAMEWORKS_DIR/libcrypto.3.dylib"
    install_name_tool -id @rpath/libssl.3.dylib "$FRAMEWORKS_DIR/libssl.3.dylib"
    install_name_tool -change /opt/homebrew/Cellar/openssl@3/3.6.4/lib/libcrypto.3.dylib @rpath/libcrypto.3.dylib "$FRAMEWORKS_DIR/libssl.3.dylib" 2>/dev/null || true
    install_name_tool -change /opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib @rpath/libcrypto.3.dylib "$FRAMEWORKS_DIR/libssl.3.dylib" 2>/dev/null || true
    install_name_tool -id @rpath/libsqlite3.dylib "$FRAMEWORKS_DIR/libsqlite3.dylib"
    install_name_tool -id @rpath/liblzma.5.dylib "$FRAMEWORKS_DIR/liblzma.5.dylib"
    install_name_tool -id @rpath/libmpdec.4.dylib "$FRAMEWORKS_DIR/libmpdec.4.dylib"

    DYNLOAD="$FRAMEWORKS_DIR/Python.framework/Versions/3.11/lib/python3.11/lib-dynload"
    install_name_tool -change /opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib @rpath/libcrypto.3.dylib "$DYNLOAD/_hashlib.cpython-311-darwin.so"
    install_name_tool -change /opt/homebrew/opt/openssl@3/lib/libssl.3.dylib @rpath/libssl.3.dylib "$DYNLOAD/_ssl.cpython-311-darwin.so"
    install_name_tool -change /opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib @rpath/libcrypto.3.dylib "$DYNLOAD/_ssl.cpython-311-darwin.so"
    install_name_tool -change /opt/homebrew/opt/sqlite/lib/libsqlite3.dylib @rpath/libsqlite3.dylib "$DYNLOAD/_sqlite3.cpython-311-darwin.so"
    install_name_tool -change /opt/homebrew/opt/xz/lib/liblzma.5.dylib @rpath/liblzma.5.dylib "$DYNLOAD/_lzma.cpython-311-darwin.so"
    install_name_tool -change /opt/homebrew/opt/mpdecimal/lib/libmpdec.4.dylib @rpath/libmpdec.4.dylib "$DYNLOAD/_decimal.cpython-311-darwin.so"

    # Establish standard macOS framework symlinks for codesign compliance
    (
        cd "$FRAMEWORKS_DIR/Python.framework/Versions" && ln -sf 3.11 Current
        cd "$FRAMEWORKS_DIR/Python.framework" && ln -sf Versions/Current/Python Python
        cd "$FRAMEWORKS_DIR/Python.framework" && ln -sf Versions/Current/Resources Resources
    )
    codesign -s - --force --deep "$FRAMEWORKS_DIR/Python.framework"
fi

# 4. Bundle Virtualenv site-packages into Resources/venv
if [ -d "$DIR/.venv" ]; then
    echo "Bundling dependencies from .venv into Resources/venv..."
    mkdir -p "$RESOURCES_DIR/venv/lib/python3.11"
    cp -R "$DIR/.venv/lib/python3.11/site-packages" "$RESOURCES_DIR/venv/lib/python3.11/"

    # Install standalone gigaam package directly into site-packages
    echo "Installing clean standalone gigaam package into site-packages..."
    rm -rf "$RESOURCES_DIR/venv/lib/python3.11/site-packages/gigaam"
    cp -R "$DIR/GigaAM_repo/gigaam" "$RESOURCES_DIR/venv/lib/python3.11/site-packages/gigaam"

    # Purge editable links, .pth files, and metadata containing local developer paths
    echo "Purging editable links and local developer paths..."
    rm -f "$RESOURCES_DIR/venv/lib/python3.11/site-packages/__editable__"*
    rm -rf "$RESOURCES_DIR/venv/lib/python3.11/site-packages/gigaam-"*.dist-info
    find "$RESOURCES_DIR/venv/lib/python3.11/site-packages" -name "direct_url.json" -exec rm -f {} +

    # Prune developer headers and unused protoc binary to optimize footprint
    rm -rf "$RESOURCES_DIR/venv/lib/python3.11/site-packages/torch/include"
    rm -f "$RESOURCES_DIR/venv/lib/python3.11/site-packages/torch/bin/protoc"*
    rm -rf "$RESOURCES_DIR/venv/lib/python3.11/site-packages/pip"*
    rm -rf "$RESOURCES_DIR/venv/lib/python3.11/site-packages/PyObjCTest"*
    rm -rf "$RESOURCES_DIR/venv/lib/python3.11/site-packages/setuptools"*
    find "$RESOURCES_DIR/venv" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
fi

# 5. Create Info.plist with exact permissions
echo "Generating Info.plist..."
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
    <string>AppIcon</string>
    <key>CFBundleIconName</key>
    <string>AppIcon</string>
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

# Standard macOS package signature
echo -n 'APPL????' > "$CONTENTS_DIR/PkgInfo"

# 6. Compile native Mach-O launcher binary linked to @rpath/Python.framework
echo "Compiling native arm64 Mach-O launcher binary..."
PY_INC="-I/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/include/python3.11"
PY_LIB="-L/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/config-3.11-darwin -lpython3.11 -ldl -framework CoreFoundation"

clang -arch arm64 -O2 $PY_INC \
  $PY_LIB \
  -Wl,-rpath,@executable_path/../Frameworks \
  "$DIR/launcher.c" -o "$MACOS_DIR/GigaDictate"

# Update dylib reference to @rpath so it loads the bundled framework on any Mac
install_name_tool -change \
  /opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/Python \
  @rpath/Python.framework/Versions/3.11/Python \
  "$MACOS_DIR/GigaDictate"

chmod +x "$MACOS_DIR/GigaDictate"

# 7. Inside-out codesign: sign all Mach-O binaries (.so and .dylib), frameworks, launcher binary, then application bundle
echo "Codesigning all bundled dynamic libraries and extensions..."
find "$APP_DIR" -type f \( -name "*.so" -o -name "*.dylib" -o -name "torch_shm_manager" \) -exec codesign -s - --force {} +
codesign -s - --force --deep "$FRAMEWORKS_DIR/Python.framework"
codesign -s - --force "$MACOS_DIR/GigaDictate"
codesign -s - --force --entitlements "$DIR/entitlements.plist" "$APP_DIR"
codesign -vvv --deep --strict "$APP_DIR"

# 8. Strict verification: ensure zero references to local developer home directory
echo "Verifying zero developer paths in release bundle..."
DEV_PATH_MATCHES=$(grep -rnI "/Users/ctazzerc" "$APP_DIR" 2>/dev/null || true)
if [ -n "$DEV_PATH_MATCHES" ]; then
    echo "❌ ERROR: Found leftover references to developer directory in $APP_DIR:"
    echo "$DEV_PATH_MATCHES"
    exit 1
fi
echo "✅ Bundle verified: ZERO references to developer directory."

echo "================================================="
echo " Standalone bundle built successfully: $APP_DIR"
echo "================================================="

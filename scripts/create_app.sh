#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
APP_NAME="GigaDictate.app"
PUBLISH_PATH="$DIR/$APP_NAME"
PYTHON_RUNTIME="${GIGA_PYTHON_RUNTIME:-$DIR/build/python}"
SITE_PACKAGES="${GIGA_SITE_PACKAGES:-$DIR/.venv/lib/python3.11/site-packages}"
GIGAAM_REPO="${GIGA_GIGAAM_REPO:-$DIR/GigaAM_repo}"
MEETING_MODELS="${GIGA_MEETING_MODELS:-$DIR/build/meeting_models}"
MIN_MACOS=14.0
PYTHON="$PYTHON_RUNTIME/bin/python3.11"

cd "$DIR"
if [ ! -x "$PYTHON" ] || [ ! -f "$PYTHON_RUNTIME/lib/libpython3.11.dylib" ]; then
    echo "Prepare the portable arm64 CPython runtime with scripts/prepare_macos_python.sh." >&2
    exit 1
fi
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SITE_PACKAGES" "$PYTHON" -c \
    'import platform, sys, ScreenCaptureKit, AVFoundation, CoreMedia, CoreAudio, onnxruntime, kaldi_native_fbank, soxr; from giga_dictate.meeting_models import verify_models; assert platform.machine() == "arm64"; verify_models(sys.argv[1])' \
    "$MEETING_MODELS"

mkdir -p "$DIR/build"
STAGE=$(mktemp -d "$DIR/build/.giga-release.XXXXXX")
APP_DIR="$STAGE/$APP_NAME"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
cleanup() {
    if [ -d "$STAGE/previous.app" ] && [ ! -e "$PUBLISH_PATH" ]; then
        mv "$STAGE/previous.app" "$PUBLISH_PATH" || return
    fi
    rm -rf "$STAGE"
}
trap cleanup EXIT

echo "Building standalone $APP_NAME (arm64, macOS $MIN_MACOS or later)"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

# 1. Copy Application Icon
if [ -f "$DIR/AppIcon.icns" ]; then
    cp "$DIR/AppIcon.icns" "$RESOURCES_DIR/AppIcon.icns"
    cp "$DIR/AppIcon.icns" "$RESOURCES_DIR/AppIcon"
fi

# 2. Bundle Python Code & Repositories into Resources
echo "Bundling Python modules into Resources..."
cp -R "$DIR/giga_dictate" "$RESOURCES_DIR/giga_dictate"
cp -R "$GIGAAM_REPO" "$RESOURCES_DIR/GigaAM_repo"
cp -R "$MEETING_MODELS" "$RESOURCES_DIR/meeting_models"
# Clean .git, tests, and __pycache__ from bundle
rm -rf "$RESOURCES_DIR/GigaAM_repo/.git"
rm -rf "$RESOURCES_DIR/GigaAM_repo/tests"
rm -rf "$RESOURCES_DIR/GigaAM_repo/tutorial"
rm -rf "$RESOURCES_DIR/GigaAM_repo/docs"
find "$RESOURCES_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
chmod -R u+w "$APP_DIR"

# Embed the relocatable runtime, with its own native libraries.
echo "Bundling portable CPython"
mkdir -p "$RESOURCES_DIR/PythonRuntime"
cp -R "$PYTHON_RUNTIME/lib" "$RESOURCES_DIR/PythonRuntime/"
RUNTIME_LIB="$RESOURCES_DIR/PythonRuntime/lib"
rm -rf "$RUNTIME_LIB"/python3.11/config-* "$RUNTIME_LIB/pkgconfig" \
    "$RUNTIME_LIB/python3.11/test" "$RUNTIME_LIB/python3.11/site-packages" \
    "$RUNTIME_LIB/python3.11/ensurepip" "$RUNTIME_LIB/python3.11/idlelib"
if [ -f "$PYTHON_RUNTIME/LICENSE.txt" ]; then
    cp "$PYTHON_RUNTIME/LICENSE.txt" "$RESOURCES_DIR/Python-LICENSE.txt"
fi

echo "Bundling Python dependencies"
mkdir -p "$RESOURCES_DIR/venv/lib/python3.11"
cp -R "$SITE_PACKAGES" "$RESOURCES_DIR/venv/lib/python3.11/site-packages"
SITE_DEST="$RESOURCES_DIR/venv/lib/python3.11/site-packages"
rm -rf "$SITE_DEST/gigaam" "$SITE_DEST"/gigaam-*.dist-info "$SITE_DEST"/__editable__* \
    "$SITE_DEST/torch/include" "$SITE_DEST"/torch/bin/protoc* "$SITE_DEST"/PyObjCTest* \
    "$SITE_DEST"/pip* "$SITE_DEST"/setuptools*
cp -R "$GIGAAM_REPO/gigaam" "$SITE_DEST/gigaam"
find "$SITE_DEST" -name direct_url.json -type f -delete
find "$CONTENTS_DIR" -name __pycache__ -type d -prune -exec rm -rf {} +

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
    <string>3.1.0</string>
    <key>CFBundleVersion</key>
    <string>3.1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppleEventsUsageDescription</key>
    <string>GigaDictate использует события ввода для вставки распознанного текста под курсор.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>GigaDictate записывает микрофон при диктовке или записи встречи. Обработка выполняется локально.</string>
    <key>NSScreenCaptureUsageDescription</key>
    <string>В режиме созвона GigaDictate захватывает системный звук для локальной стенограммы. Изображение экрана не сохраняется.</string>
</dict>
</plist>
EOF

# Standard macOS package signature
echo -n 'APPL????' > "$CONTENTS_DIR/PkgInfo"

# Compile the launcher and validate every native object before signing.
echo "Compiling native arm64 launcher"
xcrun clang -arch arm64 -mmacosx-version-min="$MIN_MACOS" -O2 \
    -I"$PYTHON_RUNTIME/include/python3.11" \
    "$PYTHON_RUNTIME/lib/libpython3.11.dylib" -framework CoreFoundation \
    -Wl,-rpath,@executable_path/../Resources/PythonRuntime/lib \
    -Wl,-rpath,@executable_path/../Resources/venv/lib/python3.11/site-packages/torch/lib \
    "$DIR/launcher.c" -o "$MACOS_DIR/GigaDictate"

"$PYTHON" "$DIR/scripts/verify_macos_bundle.py" "$APP_DIR" "$MIN_MACOS" \
    --strip-external-rpaths --signing-list "$STAGE/signing-list"
while IFS= read -r -d '' binary; do
    if [ "$binary" != "$MACOS_DIR/GigaDictate" ]; then
        codesign -s - --force "$binary" 2>"$STAGE/signing-error" || { cat "$STAGE/signing-error" >&2; exit 1; }
    fi
done < "$STAGE/signing-list"
codesign -s - --force --entitlements "$DIR/entitlements.plist" "$APP_DIR"
codesign --verify --verbose --deep --strict "$APP_DIR"

# Publish only after the staged bundle passes validation and imports its runtime.
"$MACOS_DIR/GigaDictate" -c 'import ssl, sqlite3, torch, sounddevice, ScreenCaptureKit'
if [ -n "$HOME" ] && { \
    if command -v rg >/dev/null; then rg --hidden --no-ignore -l -F -- "$HOME/" "$APP_DIR"
    else grep -rIlF -- "$HOME/" "$APP_DIR"; fi
}; then
    echo "Release contains local home-directory paths." >&2
    exit 1
fi
if [ -e "$PUBLISH_PATH" ]; then
    mv "$PUBLISH_PATH" "$STAGE/previous.app"
fi
mv "$APP_DIR" "$PUBLISH_PATH"
echo "Standalone bundle built successfully: $PUBLISH_PATH"

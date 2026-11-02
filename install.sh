#!/usr/bin/env bash
# ==============================================================================
# GigaDictate - One-command Installer for macOS (Apple Silicon)
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/install.sh | bash
# ==============================================================================

set -eo pipefail

APP_NAME="GigaDictate.app"
DEST_DIR="/Applications"
TARGET_APP="$DEST_DIR/$APP_NAME"
VOLUME_NAME="GigaDictate"

# Default repo (can be overridden via environment variable GITHUB_REPO)
REPO="${GITHUB_REPO:-stazzers/giga-dictate}"
RELEASE_URL="https://github.com/${REPO}/releases/latest/download/GigaDictate-arm64.dmg"

echo "======================================================"
echo "         🎙️ GigaDictate macOS Installer              "
echo "======================================================"

# 1. macOS & Architecture Check
OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" != "Darwin" ]; then
    echo "❌ Ошибка: GigaDictate разработан исключительно для macOS."
    exit 1
fi

if [ "$ARCH" != "arm64" ]; then
    echo "⚠️ Внимание: Обнаружена архитектура $ARCH. GigaDictate и модели GigaAM оптимизированы для Apple Silicon (M1/M2/M3/M4)."
fi

# 2. Check if running inside local cloned repository with pre-built artifacts
LOCAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" 2>/dev/null && pwd )" || LOCAL_DIR=""
INSTALLED_FROM_LOCAL=0

if [ -n "$LOCAL_DIR" ] && [ -f "$LOCAL_DIR/GigaDictate-arm64.dmg" ]; then
    echo "📦 Найден локальный образ GigaDictate-arm64.dmg..."
    DMG_PATH="$LOCAL_DIR/GigaDictate-arm64.dmg"
    INSTALLED_FROM_LOCAL=1
elif [ -n "$LOCAL_DIR" ] && [ -d "$LOCAL_DIR/GigaDictate.app" ]; then
    echo "📦 Найден локальный бандл GigaDictate.app..."
    pkill -x "GigaDictate" 2>/dev/null || true
    rm -rf "$TARGET_APP"
    cp -R "$LOCAL_DIR/GigaDictate.app" "$DEST_DIR/"
    chmod -R u+w "$TARGET_APP" 2>/dev/null || true
    xattr -cr "$TARGET_APP" 2>/dev/null || true
    echo "✅ Приложение скопировано в $TARGET_APP"
    open "$TARGET_APP"
    exit 0
fi

# 3. Download release DMG if not local
TMP_DIR="$(mktemp -d -t gigadictate_install_XXXXXX)"
cleanup() {
    if [ -d "/Volumes/$VOLUME_NAME" ]; then
        hdiutil detach "/Volumes/$VOLUME_NAME" -quiet 2>/dev/null || true
    fi
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if [ $INSTALLED_FROM_LOCAL -eq 0 ]; then
    DMG_PATH="$TMP_DIR/GigaDictate-arm64.dmg"
    echo "⬇️ Скачивание GigaDictate DMG из GitHub Releases..."
    echo "   URL: $RELEASE_URL"

    if ! curl -fL --progress-bar "$RELEASE_URL" -o "$DMG_PATH"; then
        echo "❌ Не удалось скачать предсобранный DMG с $RELEASE_URL."
        echo "   Возможно, релиз еще не создан или репозиторий является приватным."
        echo "   Для ручной сборки из исходников склонируйте репозиторий:"
        echo "     git clone https://github.com/${REPO}.git"
        echo "     cd $(basename "${REPO}") && ./run.sh"
        exit 1
    fi
fi

# 4. Mount DMG
echo "💿 Монтирование образа диска..."
MOUNT_INFO=$(hdiutil attach "$DMG_PATH" -nobrowse -noverify -noautoopen)
MOUNT_DIR=$(echo "$MOUNT_INFO" | awk '/\/Volumes\// {print $NF; exit}')

if [ -z "$MOUNT_DIR" ] || [ ! -d "$MOUNT_DIR/$APP_NAME" ]; then
    echo "❌ Ошибка при монтировании образа диска."
    exit 1
fi

# 5. Stop running instance and copy to /Applications
echo "🚀 Установка $APP_NAME в $DEST_DIR..."
pkill -x "GigaDictate" 2>/dev/null || true
rm -rf "$TARGET_APP"
cp -R "$MOUNT_DIR/$APP_NAME" "$DEST_DIR/"

# 6. Unmount
hdiutil detach "$MOUNT_DIR" -quiet 2>/dev/null || true

# 7. Remove macOS quarantine flag for seamless launch
echo "🛡️ Снятие атрибута карантина macOS (Gatekeeper)..."
chmod -R u+w "$TARGET_APP" 2>/dev/null || true
xattr -cr "$TARGET_APP" 2>/dev/null || true

# 8. Launch App
echo "✨ Запуск GigaDictate..."
open "$TARGET_APP"

echo ""
echo "======================================================"
echo "🎉 GigaDictate успешно установлен в /Applications!"
echo ""
echo "Важно для первого запуска:"
echo "1. Предоставьте доступ к Микрофону (запрос появится в окне)."
echo "2. Включите доступ в 'Системные настройки → Конфиденциальность → Универсальный доступ',"
echo "   чтобы приложение могло автоматически вставлять распознанный текст."
echo "======================================================"

#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
APP_NAME="GigaDictate.app"
APP_DIR="$DIR/$APP_NAME"
FINAL_DMG="$DIR/GigaDictate-arm64.dmg"
TEMP_DMG="$DIR/build/temp_rw.dmg"
BUILD_DIR="$DIR/build"

echo "================================================="
echo " Creating Professional macOS DMG for GigaDictate "
echo "================================================="

# 1. Ensure GigaDictate.app is built and valid
if [ ! -d "$APP_DIR" ]; then
    echo "GigaDictate.app not found. Building it first..."
    bash "$DIR/scripts/create_app.sh"
fi

rm -rf "$BUILD_DIR" "$FINAL_DMG"
mkdir -p "$BUILD_DIR"

# 2. Generate Retina background image
echo "Generating DMG window background image..."
"$DIR/.venv/bin/python" "$DIR/scripts/generate_dmg_background.py"

# 3. Create temporary writable DMG (2500MB allows ample staging room before compression)
echo "Creating temporary writable DMG image..."
hdiutil create -size 2500m -fs HFS+ -volname "GigaDictate" -ov "$TEMP_DMG"

# 4. Mount temporary DMG
echo "Mounting temporary DMG..."
MOUNT_OUTPUT=$(hdiutil attach -readwrite -noverify -noautoopen "$TEMP_DMG")
DEV_NAME=$(echo "$MOUNT_OUTPUT" | awk '/\/dev\/disk/ {print $1; exit}')
VOLUME_PATH=$(echo "$MOUNT_OUTPUT" | grep "/Volumes/" | awk '{print $NF}')

echo "Volume mounted at: $VOLUME_PATH (Device: $DEV_NAME)"
sleep 1

# 5. Populate volume contents
echo "Copying GigaDictate.app into volume..."
cp -R "$APP_DIR" "$VOLUME_PATH/"

echo "Creating Applications shortcut..."
ln -s /Applications "$VOLUME_PATH/Applications"

echo "Adding background image..."
mkdir -p "$VOLUME_PATH/.background"
cp "$BUILD_DIR/background.png" "$VOLUME_PATH/.background/background.png"
chflags hidden "$VOLUME_PATH/.background" 2>/dev/null || true

echo "Adding uninstaller and instructions..."
cat << 'EOF' > "$VOLUME_PATH/Uninstall GigaDictate.command"
#!/bin/bash
clear
echo "======================================================"
echo "    Деинсталляция GigaDictate с компьютера Mac       "
echo "======================================================"
echo ""

# 1. Close application if running
if pgrep -f "GigaDictate" >/dev/null; then
    echo "Завершение запущенного процесса GigaDictate..."
    pkill -f "GigaDictate" 2>/dev/null || true
    sleep 1
fi

# 2. Delete application bundle from /Applications
if [ -d "/Applications/GigaDictate.app" ]; then
    rm -rf "/Applications/GigaDictate.app"
    echo "✓ Приложение /Applications/GigaDictate.app успешно удалено."
else
    echo "ℹ В /Applications/GigaDictate.app приложение не найдено."
fi

# 3. Ask whether to purge cached GigaAM models and configs
echo ""
read -p "Удалить кэшированные модели GigaAM и настройки (~/.config/giga_dictate, ~/.cache/gigaam)? [y/N]: " choice
case "$choice" in 
  y|Y ) 
    rm -rf "$HOME/.config/giga_dictate"
    rm -rf "$HOME/.cache/gigaam"
    echo "✓ Настройки и скачанные модели GigaAM полностью удалены с диска."
    ;;
  * ) 
    echo "ℹ Настройки и модели сохранены."
    ;;
esac

echo ""
echo "======================================================"
echo "          Удаление завершено успешно!                 "
echo "======================================================"
echo "Вы можете закрыть это окно терминала."
EOF
chmod +x "$VOLUME_PATH/Uninstall GigaDictate.command"

cat << 'EOF' > "$VOLUME_PATH/Инструкция по установке.txt"
======================================================
           УСТАНОВКА GIGADICTATE ДЛЯ MACOS            
======================================================

1. УСТАНОВКА:
   Перетащите иконку "GigaDictate" в папку "Applications" (Программы).

2. ЗАПУСК:
   Откройте "GigaDictate" из папки "Программы" (Launchpad / Finder -> Программы).
   При запуске иконка 🎙️ Giga появится в верхней строке меню (Menu Bar).

3. РАЗРЕШЕНИЯ MACOS (Shield / Privacy):
   Приложению требуются строго 2 разрешения macOS:
   - Микрофон: системный запрос появится автоматически при первом включении записи.
   - Универсальный доступ (Accessibility): требуется macOS для автоматической
     вставки текста под курсор (имитация нажатия Cmd+V).
     Включите переключатель для GigaDictate в:
     Системные настройки -> Конфиденциальность и безопасность -> Универсальный доступ.
   (Никаких лишних доступов к файлам, камере или геолокации не требуется).

4. УДАЛЕНИЕ:
   - Стандартное удаление: перетащите GigaDictate из папки "Программы" в Корзину.
   - Полная очистка с кэшем моделей: запустите "Uninstall GigaDictate.command".

Приятного использования высокоскоростной локальной диктовки GigaAM-v3!
EOF

# 6. Configure Finder window appearance & icon coordinates via AppleScript
echo "Configuring Finder window geometry and icon layout..."
osascript << 'EOF'
tell application "Finder"
    tell disk "GigaDictate"
        open
        delay 1
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {300, 150, 860, 510}
        
        set theViewOptions to the icon view options of container window
        set arrangement of theViewOptions to not arranged
        set icon size of theViewOptions to 96
        set text size of theViewOptions to 12
        set background picture of theViewOptions to file ".background:background.png"
        
        set position of item "GigaDictate.app" of container window to {140, 175}
        set position of item "Applications" of container window to {420, 175}
        set position of item "Uninstall GigaDictate.command" of container window to {170, 305}
        set position of item "Инструкция по установке.txt" of container window to {390, 305}
        
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

# 7. Unmount volume
echo "Detaching temporary volume..."
hdiutil detach "$DEV_NAME"

# 8. Convert to final read-only, compressed DMG
echo "Compressing into final read-only DMG (UDZO zlib-level 9)..."
hdiutil convert "$TEMP_DMG" -format UDZO -imagekey zlib-level=9 -o "$FINAL_DMG"

# 9. Clean up temporary files
rm -rf "$BUILD_DIR"

echo "================================================="
echo " Final DMG created successfully at: $FINAL_DMG"
du -sh "$FINAL_DMG"
echo "================================================="

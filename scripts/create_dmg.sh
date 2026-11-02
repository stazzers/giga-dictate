#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
APP_NAME="GigaDictate.app"
APP_DIR="$DIR/$APP_NAME"
DMG_NAME="GigaDictate-arm64.dmg"
DMG_PATH="$DIR/$DMG_NAME"
STAGING_DIR="$DIR/build/dmg_staging"

echo "================================================="
echo " Creating GigaDictate DMG for Apple Silicon      "
echo "================================================="

# 1. Ensure GigaDictate.app is built
if [ ! -d "$APP_DIR" ]; then
    echo "GigaDictate.app not found. Building it first..."
    bash "$DIR/scripts/create_app.sh"
fi

# 2. Prepare DMG staging directory
rm -rf "$DIR/build" "$DMG_PATH"
mkdir -p "$STAGING_DIR"

echo "Copying GigaDictate.app to staging..."
cp -R "$APP_DIR" "$STAGING_DIR/"

echo "Creating Applications shortcut..."
ln -s /Applications "$STAGING_DIR/Applications"

# 3. Create uninstaller script
cat << 'EOF' > "$STAGING_DIR/Uninstall GigaDictate.command"
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
chmod +x "$STAGING_DIR/Uninstall GigaDictate.command"

# 4. Create README instruction file
cat << 'EOF' > "$STAGING_DIR/Инструкция по установке.txt"
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

# 5. Build compressed DMG
echo "Generating DMG image via hdiutil..."
hdiutil create \
  -volname "GigaDictate" \
  -srcfolder "$STAGING_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

# Clean staging build folder
rm -rf "$DIR/build"

echo "================================================="
echo " DMG created successfully at: $DMG_PATH"
du -sh "$DMG_PATH"
echo "================================================="

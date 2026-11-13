#!/bin/bash
set -e

echo "================================================="
echo "   GigaDictate — Полное удаление из системы      "
echo "================================================="

# 1. Завершение запущенных процессов
echo "1. Остановка процессов GigaDictate..."
pkill -f "GigaDictate" 2>/dev/null || true

# 2. Удаление приложения из /Applications
echo "2. Удаление бандлов приложения..."
if [ -d "/Applications/GigaDictate.app" ]; then
    rm -rf "/Applications/GigaDictate.app"
    echo "   Удалено: /Applications/GigaDictate.app"
fi
if [ -d "$HOME/Applications/GigaDictate.app" ]; then
    rm -rf "$HOME/Applications/GigaDictate.app"
    echo "   Удалено: $HOME/Applications/GigaDictate.app"
fi

# 3. Дерегистрация из LaunchServices
echo "3. Очистка кэша LaunchServices..."
LSREG="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
if [ -x "$LSREG" ]; then
    "$LSREG" -u "/Applications/GigaDictate.app" 2>/dev/null || true
    "$LSREG" -gc 2>/dev/null || true
fi

# 4. Сброс системных разрешений macOS (TCC)
echo "4. Сброс доступов (Микрофон, Входной мониторинг, Универсальный доступ)..."
BUNDLE_ID="com.salute.gigadictate"
tccutil reset Microphone "$BUNDLE_ID" 2>/dev/null || true
tccutil reset Accessibility "$BUNDLE_ID" 2>/dev/null || true
tccutil reset ListenEvent "$BUNDLE_ID" 2>/dev/null || true
tccutil reset SpeechRecognition "$BUNDLE_ID" 2>/dev/null || true
tccutil reset ScreenCapture "$BUNDLE_ID" 2>/dev/null || true
tccutil reset All "$BUNDLE_ID" 2>/dev/null || true

# 5. Очистка автозагрузки, конфигурации и кэша моделей нейросети (~900 МБ)
echo "5. Очистка автозагрузки, конфигурации и кэша моделей..."
rm -f "$HOME/Library/LaunchAgents/com.salute.gigadictate.plist" 2>/dev/null || true
rm -rf "$HOME/.config/giga_dictate"
rm -rf "$HOME/.cache/gigaam"

# 6. Удаление отчетов о сбоях/диагностике
echo "6. Очистка логов диагностики..."
rm -f "$HOME/Library/Logs/DiagnosticReports/GigaDictate"* 2>/dev/null || true
rm -f "$HOME/Library/Application Support/CrashReporter/GigaDictate"* 2>/dev/null || true

echo "================================================="
echo " ✅ GigaDictate полностью удален без следов.     "
echo "================================================="

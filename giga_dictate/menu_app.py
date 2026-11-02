import logging
import os
import sys
import threading
import time
from typing import List, Optional
import objc
from AppKit import (
    NSAlert,
    NSAlertFirstButtonReturn,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSImage,
    NSImageLeft,
    NSImageOnly,
    NSMenu,
    NSMenuItem,
    NSPoint,
    NSRect,
    NSSize,
    NSStatusBar,
    NSTextField,
    NSVariableStatusItemLength,
)
from Foundation import NSObject
from PyObjCTools import AppHelper

from .audio import AudioRecorder
from .config import AppConfig
from .engine import GigaAMEngine
from .hotkey import HotkeyListener, parse_hotkey
from .hud import VoiceInputHUD
from .inserter import TextInserter
from .sounds import SoundManager

logger = logging.getLogger("giga_dictate.menu_app")

MODELS = [
    ("v3_e2e_ctc", "v3 E2E CTC (Рекомендуется, быстрый, пунктуация)"),
    ("v3_e2e_rnnt", "v3 E2E RNN-T (Высокая точность, пунктуация)"),
    ("v3_ctc", "v3 CTC (Быстрый, строчные буквы)"),
    ("v3_rnnt", "v3 RNN-T (Стандартный RNN-T)"),
]

DEVICES = [
    ("mps", "Apple Silicon GPU (Metal / MPS)"),
    ("cpu", "CPU (Процессор)"),
]

HOTKEY_PRESETS = [
    ("right_cmd", "Правый Command (Right Cmd) — как в SuperDictate"),
    ("right_shift", "Правый Shift (Right Shift)"),
    ("alt_r", "Правый Option / Alt (Right Option)"),
    ("ctrl_r", "Правый Control (Right Control)"),
    ("f8", "Клавиша F8"),
    ("/", "Слэш ( / )"),
]


_GLOBAL_STATUS_ITEMS = []


class MenuAppController(NSObject):
    def init(self):
        self = objc.super(MenuAppController, self).init()
        if self is None:
            return None

        self.config = AppConfig.load()
        self.sounds = SoundManager(enabled=self.config.sound_feedback)
        self.recorder = AudioRecorder(device=self.config.audio_device)
        self.engine = GigaAMEngine(
            model_name=self.config.model_name,
            device=self.config.device,
        )
        self.inserter = TextInserter()

        self.status_item = None
        self.menu = None
        self.menu_icon = None
        self.menu_icon_rec = None

        # Voice Input HUD
        self.hud = VoiceInputHUD.alloc().init()
        self.hud.setAudioRecorder_(self.recorder)
        self._perm_check_timer = None

        # Hotkey listener
        self.hotkey_listener = HotkeyListener(
            hotkey_str=self.config.hotkey,
            mode=self.config.hotkey_mode,
            on_start=self.on_hotkey_start,
            on_stop=self.on_hotkey_stop,
        )

        return self

    def applicationDidFinishLaunching_(self, notification):
        # Configure Menu Bar item
        status_bar = NSStatusBar.systemStatusBar()
        self.status_item = status_bar.statusItemWithLength_(NSVariableStatusItemLength)
        self.status_item.setVisible_(True)
        _GLOBAL_STATUS_ITEMS.append(self.status_item)

        # Load menu icons from resources
        res_dir = os.path.join(os.path.dirname(__file__), "resources")
        icon_path = os.path.join(res_dir, "menu_icon.png")
        if os.path.exists(icon_path):
            self.menu_icon = NSImage.alloc().initWithContentsOfFile_(icon_path)
            if self.menu_icon:
                self.menu_icon.setTemplate_(True)

        rec_icon_path = os.path.join(res_dir, "menu_icon_rec.png")
        if os.path.exists(rec_icon_path):
            self.menu_icon_rec = NSImage.alloc().initWithContentsOfFile_(rec_icon_path)

        btn = self.status_item.button()
        if btn:
            btn.setImagePosition_(NSImageOnly)
            btn.setToolTip_("GigaDictate — Голосовой ввод GigaAM-v3")
            if self.menu_icon:
                btn.setImage_(self.menu_icon)

        self.update_status_title("Загрузка...")
        self.rebuild_menu()

        # Check Accessibility permission on launch
        if not TextInserter.is_accessibility_granted():
            logger.info("Accessibility not granted on launch, prompting user...")
            TextInserter.request_accessibility()

        # Start hotkey listener
        self.hotkey_listener.start()
        if self.hotkey_listener._tap is None:
            self._perm_check_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.5, self, "onCheckPermissionsTick:", None, True
            )

        # Ensure pure menu bar accessory policy (no Dock icon, managed exclusively from menu bar)
        NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        # Load GigaAM model on background thread
        threading.Thread(target=self._initial_model_load, daemon=True).start()

    def onCheckPermissionsTick_(self, timer):
        """Periodically checks if user granted Accessibility and connects event tap automatically."""
        if TextInserter.is_accessibility_granted():
            if self.hotkey_listener._tap is None:
                logger.info("Accessibility granted by user! Activating hotkey listener...")
                self.hotkey_listener.start()
            if self.hotkey_listener._tap is not None:
                if self._perm_check_timer is not None:
                    try:
                        self._perm_check_timer.invalidate()
                    except (AttributeError, RuntimeError) as e:
                        logger.debug(f"Timer invalidate error: {e}")
                    self._perm_check_timer = None
                self.rebuild_menu()

    def _initial_model_load(self):
        success = self.engine.load()
        if success:
            AppHelper.callAfter(self.onModelLoaded_, None)
        else:
            AppHelper.callAfter(self.onModelError_, self.engine.load_error or "Unknown error")

    def onModelLoaded_(self, _):
        self.update_status_title("")
        self.rebuild_menu()
        logger.info("Ready for dictation.")

    def onModelError_(self, error_str):
        self.update_status_title("Ошибка GigaAM")
        self.rebuild_menu()
        logger.error(f"Model load error: {error_str}")

    def update_status_title(self, title: str, recording: bool = False):
        if self.status_item and self.status_item.button():
            btn = self.status_item.button()
            if recording:
                if self.menu_icon_rec:
                    btn.setImage_(self.menu_icon_rec)
                btn.setImagePosition_(NSImageOnly)
                btn.setTitle_("")
            elif not self.engine.is_loaded:
                if self.menu_icon:
                    btn.setImage_(self.menu_icon)
                btn.setImagePosition_(NSImageLeft)
                btn.setTitle_(" ⏳")
            else:
                # Ready state: clean icon only, takes ~24px, never hidden by MacBook notch!
                if self.menu_icon:
                    btn.setImage_(self.menu_icon)
                btn.setImagePosition_(NSImageOnly)
                btn.setTitle_("")
            self.status_item.setVisible_(True)

    def rebuild_menu(self):
        menu = NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)

        # 1. Header / Status
        if self.recorder.is_recording:
            status_text = "🔴 Идет запись голоса..."
        elif not self.engine.is_loaded:
            status_text = "⏳ Загрузка модели GigaAM-v3..."
        else:
            mode_desc = "Удерживайте" if self.config.hotkey_mode == "push_to_talk" else "Нажмите"
            status_text = f"🟢 Готов к вводу ({mode_desc} {self.config.hotkey})"

        status_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            status_text, None, ""
        )
        status_item.setEnabled_(False)
        menu.addItem_(status_item)

        # 2. Manual toggle recording
        action_title = "⏹️ Остановить запись" if self.recorder.is_recording else "🎙️ Начать запись"
        toggle_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            action_title, "onMenuToggleRecord:", ""
        )
        toggle_item.setTarget_(self)
        toggle_item.setEnabled_(self.engine.is_loaded)
        menu.addItem_(toggle_item)

        menu.addItem_(NSMenuItem.separatorItem())

        # 3. Model Submenu
        model_menu_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Модель GigaAM", None, ""
        )
        model_sub = NSMenu.alloc().init()
        for m_id, m_label in MODELS:
            m_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                m_label, "onSelectModel:", ""
            )
            m_item.setTarget_(self)
            m_item.setRepresentedObject_(m_id)
            if self.config.model_name == m_id:
                m_item.setState_(1)  # Checked
            model_sub.addItem_(m_item)
        model_menu_item.setSubmenu_(model_sub)
        menu.addItem_(model_menu_item)

        # 4. Device Submenu
        device_menu_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Устройство ускорения", None, ""
        )
        device_sub = NSMenu.alloc().init()
        for d_id, d_label in DEVICES:
            d_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                d_label, "onSelectDevice:", ""
            )
            d_item.setTarget_(self)
            d_item.setRepresentedObject_(d_id)
            if self.config.device == d_id:
                d_item.setState_(1)
            device_sub.addItem_(d_item)
        device_menu_item.setSubmenu_(device_sub)
        menu.addItem_(device_menu_item)

        # 5. Hotkey Mode Submenu
        mode_menu_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Режим вызова клавиши", None, ""
        )
        mode_sub = NSMenu.alloc().init()
        ptt_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Push-to-Talk (удерживать пока говорите)", "onSelectMode:", ""
        )
        ptt_item.setTarget_(self)
        ptt_item.setRepresentedObject_("push_to_talk")
        if self.config.hotkey_mode == "push_to_talk":
            ptt_item.setState_(1)
        mode_sub.addItem_(ptt_item)

        tog_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Переключение (нажать для старта, нажать для стопа)", "onSelectMode:", ""
        )
        tog_item.setTarget_(self)
        tog_item.setRepresentedObject_("toggle")
        if self.config.hotkey_mode == "toggle":
            tog_item.setState_(1)
        mode_sub.addItem_(tog_item)
        mode_menu_item.setSubmenu_(mode_sub)
        menu.addItem_(mode_menu_item)

        # 6. Hotkey Selection Submenu
        hotkey_menu_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            f"Горячая клавиша: {self.config.hotkey}", None, ""
        )
        hotkey_sub = NSMenu.alloc().init()
        for h_id, h_label in HOTKEY_PRESETS:
            h_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                h_label, "onSelectHotkeyPreset:", ""
            )
            h_item.setTarget_(self)
            h_item.setRepresentedObject_(h_id)
            if self.config.hotkey == h_id:
                h_item.setState_(1)
            hotkey_sub.addItem_(h_item)

        hotkey_sub.addItem_(NSMenuItem.separatorItem())
        custom_h_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "✏️ Назначить другую клавишу...", "onCustomHotkeyDialog:", ""
        )
        custom_h_item.setTarget_(self)
        hotkey_sub.addItem_(custom_h_item)
        hotkey_menu_item.setSubmenu_(hotkey_sub)
        menu.addItem_(hotkey_menu_item)

        menu.addItem_(NSMenuItem.separatorItem())

        # 7. Preferences Checkboxes
        sound_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "🔊 Звуковые сигналы", "onToggleSound:", ""
        )
        sound_item.setTarget_(self)
        sound_item.setState_(1 if self.config.sound_feedback else 0)
        menu.addItem_(sound_item)

        space_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "␣ Добавлять пробел после текста", "onToggleSpace:", ""
        )
        space_item.setTarget_(self)
        space_item.setState_(1 if self.config.add_trailing_space else 0)
        menu.addItem_(space_item)

        # 8. Check Permissions
        perm_has = TextInserter.is_accessibility_granted()
        perm_label = "🛡️ Универсальный доступ: Разрешено" if perm_has else "⚠️ Разрешить Универсальный доступ (Вставка)"
        perm_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            perm_label, "onCheckPermissions:", ""
        )
        perm_item.setTarget_(self)
        menu.addItem_(perm_item)

        menu.addItem_(NSMenuItem.separatorItem())

        # 9. About & Quit
        about_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "О программе GigaDictate...", "onAboutDialog:", ""
        )
        about_item.setTarget_(self)
        menu.addItem_(about_item)

        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Завершить GigaDictate", "onQuitApp:", "q"
        )
        quit_item.setTarget_(self)
        menu.addItem_(quit_item)

        self.menu = menu
        if self.status_item:
            self.status_item.setMenu_(menu)

    # --- Actions ---

    def onMenuToggleRecord_(self, sender):
        if self.recorder.is_recording:
            self.stop_and_transcribe()
        else:
            self.start_recording()

    def on_hotkey_start(self):
        """Called by hotkey listener on background thread."""
        logger.info("on_hotkey_start -> dispatching start_recording to main thread")
        AppHelper.callAfter(self.start_recording)

    def on_hotkey_stop(self):
        """Called by hotkey listener on background thread."""
        logger.info("on_hotkey_stop -> dispatching stop_and_transcribe to main thread")
        AppHelper.callAfter(self.stop_and_transcribe)

    def start_recording(self):
        logger.info("start_recording executing on main thread")
        if not self.engine.is_loaded:
            logger.info("Model still warming up in background. Pre-roll recording started.")

        if self.recorder.start():
            logger.info("Audio recording started! Showing HUD.")
            self.update_status_title("Запись...", recording=True)
            self.sounds.play_start()
            self.hud.showRecording()
            self.rebuild_menu()
        else:
            logger.warning("recorder.start() returned False")

    def stop_and_transcribe(self):
        logger.info("stop_and_transcribe executing on main thread")
        if not self.recorder.is_recording:
            logger.warning("stop_and_transcribe called, but recorder is not recording.")
            return

        self.update_status_title("Обработка...")
        self.sounds.play_stop()
        self.hud.showTranscribing()
        self.rebuild_menu()

        # Stop recording and run transcription in worker thread
        audio, duration = self.recorder.stop()

        def _worker():
            if duration < self.config.min_duration or len(audio) == 0:
                logger.info(f"Audio too short ({duration:.2f}s), skipping.")
                self.sounds.play_cancel()
                AppHelper.callAfter(self._finishTranscription_, "")
                return

            if not self.engine.is_loaded:
                logger.info("Waiting for model warmup to finish before transcription...")
                self.engine.wait_loaded(timeout=15.0)

            t0 = time.time()
            try:
                text = self.engine.transcribe(audio)
                proc_time = time.time() - t0
                word_count = len(text.split()) if text else 0
                logger.info(
                    f"Transcribed speech in {proc_time:.2f}s ({len(text)} chars, {word_count} words). "
                    "Zero-Persistence: text content is not written to disk logs."
                )

                try:
                    import torch
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except (RuntimeError, AttributeError, ImportError) as err:
                    logger.debug(f"MPS cache cleanup skipped: {err}")

                if text:
                    # Insert text under cursor
                    self.inserter.insert_text(
                        text,
                        add_trailing_space=self.config.add_trailing_space,
                        restore_clipboard=self.config.restore_clipboard,
                        conceal_clipboard=self.config.conceal_clipboard,
                        delay=self.config.paste_delay,
                    )
                    self.sounds.play_success()
                    AppHelper.callAfter(self._finishTranscription_, text)
                else:
                    self.sounds.play_cancel()
                    AppHelper.callAfter(self._finishTranscription_, "")
            except Exception as e:
                logger.error(f"Transcription error: {e}", exc_info=True)
                self.sounds.play_cancel()
                AppHelper.callAfter(self._finishTranscription_, "")

        threading.Thread(target=_worker, daemon=True).start()

    def _finishTranscription_(self, text):
        self.hud.hideHUD()
        self.update_status_title("")
        self.rebuild_menu()

    def onSelectModel_(self, sender):
        new_model = sender.representedObject()
        if new_model == self.config.model_name:
            return
        self.config.model_name = new_model
        self.config.save()
        self.update_status_title(f"⏳ Загрузка {new_model}...")
        self.rebuild_menu()

        def _reload():
            self.engine.load(model_name=new_model)
            AppHelper.callAfter(self.onModelLoaded_, None)

        threading.Thread(target=_reload, daemon=True).start()

    def onSelectDevice_(self, sender):
        new_device = sender.representedObject()
        if new_device == self.config.device:
            return
        self.config.device = new_device
        self.config.save()
        self.update_status_title(f"⏳ Переключение на {new_device}...")
        self.rebuild_menu()

        def _reload():
            self.engine.load(device=new_device)
            AppHelper.callAfter(self.onModelLoaded_, None)

        threading.Thread(target=_reload, daemon=True).start()

    def onSelectMode_(self, sender):
        new_mode = sender.representedObject()
        self.config.hotkey_mode = new_mode
        self.config.save()
        self.hotkey_listener.update_hotkey(self.config.hotkey, mode=new_mode)
        self.rebuild_menu()

    def onSelectHotkeyPreset_(self, sender):
        new_hotkey = sender.representedObject()
        self.config.hotkey = new_hotkey
        self.config.save()
        self.hotkey_listener.update_hotkey(new_hotkey)
        self.rebuild_menu()

    def onCustomHotkeyDialog_(self, sender):
        alert = NSAlert.alloc().init()
        alert.setMessageText_("Назначить горячую клавишу")
        alert.setInformativeText_(
            "Введите клавишу или комбинацию (например: right_shift, alt_r, f8, f5, ctrl+space, cmd+shift+d):"
        )
        alert.addButtonWithTitle_("Сохранить")
        alert.addButtonWithTitle_("Отмена")

        field = NSTextField.alloc().initWithFrame_(NSRect(NSPoint(0, 0), NSSize(260, 24)))
        field.setStringValue_(self.config.hotkey)
        alert.setAccessoryView_(field)

        resp = alert.runModal()
        if resp == NSAlertFirstButtonReturn:
            val = str(field.stringValue()).strip()
            try:
                parse_hotkey(val)
                self.config.hotkey = val
                self.config.save()
                self.hotkey_listener.update_hotkey(val)
                self.rebuild_menu()
            except Exception as e:
                err_alert = NSAlert.alloc().init()
                err_alert.setMessageText_("Некорректная клавиша")
                err_alert.setInformativeText_(str(e))
                err_alert.runModal()

    def onToggleSound_(self, sender):
        self.config.sound_feedback = not self.config.sound_feedback
        self.sounds.enabled = self.config.sound_feedback
        self.config.save()
        self.rebuild_menu()

    def onToggleSpace_(self, sender):
        self.config.add_trailing_space = not self.config.add_trailing_space
        self.config.save()
        self.rebuild_menu()

    def onCheckPermissions_(self, sender):
        if not TextInserter.is_accessibility_granted():
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Требуется Универсальный доступ")
            alert.setInformativeText_(
                "macOS требует разрешение 'Универсальный доступ' (Accessibility) "
                "для симуляции нажатия Cmd+V и автовставки текста под курсор.\n\n"
                "Если разрешение не выдано, текст всё равно копируется в буфер обмена, "
                "и вы можете вставить его вручную нажав Cmd+V."
            )
            alert.addButtonWithTitle_("Открыть Настройки macOS")
            alert.addButtonWithTitle_("Позже")
            if alert.runModal() == NSAlertFirstButtonReturn:
                TextInserter.request_accessibility()
        else:
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Разрешения в порядке")
            alert.setInformativeText_("Разрешение 'Универсальный доступ' уже предоставлено. Автовставка работает!")
            alert.addButtonWithTitle_("OK")
            alert.runModal()

    def onAboutDialog_(self, sender):
        alert = NSAlert.alloc().init()
        alert.setMessageText_("О программе GigaDictate")
        alert.setInformativeText_(
            "GigaDictate — локальное высокоскоростное голосовое распознавание для macOS.\n\n"
            "• Основано на моделях Salute Developers GigaAM-v3\n"
            "• Ускорение на Apple Silicon (MPS / Metal)\n"
            "• Автоматическая пунктуация и капитализация\n"
            "• Работа в фоне и вставка под курсор"
        )
        alert.addButtonWithTitle_("OK")
        alert.runModal()

    def onRestartApp_(self, sender):
        self.restart_application()

    def restart_application(self):
        logger.info("Restarting application...")
        self.hotkey_listener.stop()
        self.hud.hideHUD()
        self.recorder.close()
        from AppKit import NSURL, NSWorkspace
        app_url = NSURL.fileURLWithPath_("/Applications/GigaDictate.app")
        if app_url:
            NSWorkspace.sharedWorkspace().openURL_(app_url)
        NSApplication.sharedApplication().terminate_(self)

    def onQuitApp_(self, sender):
        self.hotkey_listener.stop()
        self.hud.hideHUD()
        self.recorder.close()
        NSApplication.sharedApplication().terminate_(self)

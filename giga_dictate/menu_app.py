import logging
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import objc
from AppKit import (
    NSAlert,
    NSAlertFirstButtonReturn,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSEvent,
    NSEventMaskFlagsChanged,
    NSEventMaskKeyDown,
    NSFloatingWindowLevel,
    NSFont,
    NSImage,
    NSImageLeft,
    NSImageOnly,
    NSMenu,
    NSMenuItem,
    NSPoint,
    NSRect,
    NSRoundedBezelStyle,
    NSSize,
    NSStatusBar,
    NSTextAlignmentCenter,
    NSTextField,
    NSVariableStatusItemLength,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSBundle, NSObject, NSTimer
from PyObjCTools import AppHelper

from .audio import AudioRecorder
from .config import AppConfig
from .engine import GigaAMEngine
from .hotkey import (
    MODIFIER_MASKS,
    HotkeyListener,
    get_hotkey_display_name,
    vk_to_hotkey_identifier_and_label,
)
from .hud import VoiceInputHUD
from .inserter import TextInserter
from .sounds import SoundManager

logger = logging.getLogger("giga_dictate.menu_app")

_WORKER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="GigaWorker")

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


class CaptureWindowDelegate(NSObject):
    def initWithController_(self, controller):
        self = objc.super(CaptureWindowDelegate, self).init()
        if self is None:
            return None
        self._controller = controller
        return self

    def windowWillClose_(self, notification):
        if self._controller:
            self._controller._clean_capture_monitors()


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
        self._status_menu_item = None
        self._toggle_menu_item = None

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

        # Key capture window state
        self._capture_window = None
        self._capture_local_monitor = None
        self._capture_global_monitor = None
        self._is_capturing_key = False
        self._capture_delegate = None

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
        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory
        )

        # Load GigaAM model on background worker
        _WORKER_EXECUTOR.submit(self._initial_model_load)

    def onCheckPermissionsTick_(self, timer):
        """Periodically checks if user granted Accessibility and connects event tap automatically."""
        if TextInserter.is_accessibility_granted():
            if self.hotkey_listener._tap is None:
                logger.info(
                    "Accessibility granted by user! Activating hotkey listener..."
                )
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
            AppHelper.callAfter(
                self.onModelError_, self.engine.load_error or "Unknown error"
            )

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
            mode_desc = (
                "Удерживайте"
                if self.config.hotkey_mode == "push_to_talk"
                else "Нажмите"
            )
            status_text = (
                f"🟢 Готов к вводу ({mode_desc} {get_hotkey_display_name(self.config.hotkey)})"
            )

        status_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            status_text, None, ""
        )
        status_item.setEnabled_(False)
        menu.addItem_(status_item)
        self._status_menu_item = status_item

        # 2. Manual toggle recording
        action_title = (
            "⏹️ Остановить запись" if self.recorder.is_recording else "🎙️ Начать запись"
        )
        toggle_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            action_title, "onMenuToggleRecord:", ""
        )
        toggle_item.setTarget_(self)
        toggle_item.setEnabled_(self.engine.is_loaded)
        menu.addItem_(toggle_item)
        self._toggle_menu_item = toggle_item

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
        curr_label = get_hotkey_display_name(self.config.hotkey)
        hotkey_menu_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            f"Горячая клавиша: {curr_label}", None, ""
        )
        hotkey_sub = NSMenu.alloc().init()

        is_default = self.config.hotkey == "right_cmd"

        # 1. Default hotkey: Right Command
        def_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "По умолчанию: Правый Command (Right Cmd)", "onSelectDefaultHotkey:", ""
        )
        def_item.setTarget_(self)
        if is_default:
            def_item.setState_(1)
        hotkey_sub.addItem_(def_item)

        # 2. Selected custom hotkey (added in the middle ONLY if differs from default)
        if not is_default:
            cust_label = f"Выбранная клавиша: {get_hotkey_display_name(self.config.hotkey)}"
            cust_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                cust_label, "onSelectCustomHotkey:", ""
            )
            cust_item.setTarget_(self)
            cust_item.setState_(1)
            hotkey_sub.addItem_(cust_item)

        hotkey_sub.addItem_(NSMenuItem.separatorItem())

        # 3. Configure / change hotkey
        record_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "✏️ Изменить клавишу...", "onCustomHotkeyDialog:", ""
        )
        record_item.setTarget_(self)
        hotkey_sub.addItem_(record_item)

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
        perm_label = (
            "🛡️ Универсальный доступ: Разрешено"
            if perm_has
            else "⚠️ Разрешить Универсальный доступ (Вставка)"
        )
        perm_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            perm_label, "onCheckPermissions:", ""
        )
        perm_item.setTarget_(self)
        menu.addItem_(perm_item)

        menu.addItem_(NSMenuItem.separatorItem())

        # 9. About, Uninstall & Quit
        about_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "О программе GigaDictate...", "onAboutDialog:", ""
        )
        about_item.setTarget_(self)
        menu.addItem_(about_item)

        uninstall_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "🗑️ Полное удаление...", "onUninstallAppDialog:", ""
        )
        uninstall_item.setTarget_(self)
        menu.addItem_(uninstall_item)

        menu.addItem_(NSMenuItem.separatorItem())

        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Завершить GigaDictate", "onQuitApp:", "q"
        )
        quit_item.setTarget_(self)
        menu.addItem_(quit_item)

        self.menu = menu
        if self.status_item:
            self.status_item.setMenu_(menu)

    def update_menu_status(self):
        """Updates dynamic status and toggle items in-place without rebuilding entire menu."""
        if self._status_menu_item is None or self._toggle_menu_item is None:
            self.rebuild_menu()
            return

        if self.recorder.is_recording:
            status_text = "🔴 Идет запись голоса..."
            action_title = "⏹️ Остановить запись"
        elif not self.engine.is_loaded:
            status_text = "⏳ Загрузка модели GigaAM-v3..."
            action_title = "🎙️ Начать запись"
        else:
            mode_desc = (
                "Удерживайте"
                if self.config.hotkey_mode == "push_to_talk"
                else "Нажмите"
            )
            status_text = (
                f"🟢 Готов к вводу ({mode_desc} {get_hotkey_display_name(self.config.hotkey)})"
            )
            action_title = "🎙️ Начать запись"

        self._status_menu_item.setTitle_(status_text)
        self._toggle_menu_item.setTitle_(action_title)
        self._toggle_menu_item.setEnabled_(self.engine.is_loaded)

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
            logger.info(
                "Model still warming up in background. Pre-roll recording started."
            )

        if self.recorder.start():
            logger.info("Audio recording started! Showing HUD.")
            self.update_status_title("Запись...", recording=True)
            self.sounds.play_start()
            self.hud.showRecording()
            self.update_menu_status()
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
        self.update_menu_status()

        # Stop recording and run transcription in dedicated worker executor
        audio, duration = self.recorder.stop()

        def _worker():
            if duration < self.config.min_duration or len(audio) == 0:
                logger.info(f"Audio too short ({duration:.2f}s), skipping.")
                self.sounds.play_cancel()
                AppHelper.callAfter(self._finishTranscription_, "")
                return

            if not self.engine.is_loaded:
                logger.info(
                    "Waiting for model warmup to finish before transcription..."
                )
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

        _WORKER_EXECUTOR.submit(_worker)

    def _finishTranscription_(self, text):
        self.hud.hideHUD()
        self.update_status_title("")
        self.update_menu_status()

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

        _WORKER_EXECUTOR.submit(_reload)

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

        _WORKER_EXECUTOR.submit(_reload)

    def onSelectMode_(self, sender):
        new_mode = sender.representedObject()
        self.config.hotkey_mode = new_mode
        self.config.save()
        self.hotkey_listener.update_hotkey(self.config.hotkey, mode=new_mode)
        self.rebuild_menu()

    def onSelectDefaultHotkey_(self, sender):
        if self.config.hotkey == "right_cmd":
            return
        self.config.hotkey = "right_cmd"
        self.config.custom_hotkey = None
        self.config.save()
        self.hotkey_listener.update_hotkey("right_cmd")
        self.rebuild_menu()

    def onSelectCustomHotkey_(self, sender):
        if not self.config.custom_hotkey:
            self.onCustomHotkeyDialog_(sender)
            return
        if self.config.hotkey == self.config.custom_hotkey:
            return
        self.config.hotkey = self.config.custom_hotkey
        self.config.save()
        self.hotkey_listener.update_hotkey(self.config.custom_hotkey)
        self.rebuild_menu()

    def onCustomHotkeyDialog_(self, sender):
        """Opens automatic key recorder window where pressing any key captures it."""
        self._open_key_capture_window()

    def _open_key_capture_window(self):
        # If already open, bring to front
        if self._capture_window is not None:
            self._capture_window.makeKeyAndOrderFront_(None)
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
            return

        rect = NSRect(NSPoint(0, 0), NSSize(380, 205))
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable,
            NSBackingStoreBuffered,
            False,
        )
        window.setTitle_("Назначение горячей клавиши")
        window.center()
        window.setLevel_(NSFloatingWindowLevel)
        window.setReleasedWhenClosed_(False)

        self._capture_delegate = CaptureWindowDelegate.alloc().initWithController_(self)
        window.setDelegate_(self._capture_delegate)

        content = window.contentView()

        # Title
        title_label = NSTextField.alloc().initWithFrame_(
            NSRect(NSPoint(20, 150), NSSize(340, 26))
        )
        title_label.setStringValue_("🎙️ Назначение клавиши записи")
        title_label.setFont_(NSFont.boldSystemFontOfSize_(15.0))
        title_label.setAlignment_(NSTextAlignmentCenter)
        title_label.setBezeled_(False)
        title_label.setDrawsBackground_(False)
        title_label.setEditable_(False)
        title_label.setSelectable_(False)
        content.addSubview_(title_label)

        # Key badge / placeholder
        badge = NSTextField.alloc().initWithFrame_(
            NSRect(NSPoint(40, 92), NSSize(300, 46))
        )
        badge.setStringValue_("[ Нажмите любую клавишу ]")
        badge.setFont_(NSFont.boldSystemFontOfSize_(16.0))
        badge.setAlignment_(NSTextAlignmentCenter)
        badge.setBezeled_(True)
        badge.setDrawsBackground_(True)
        badge.setEditable_(False)
        badge.setSelectable_(False)
        content.addSubview_(badge)

        # Hint text
        hint = NSTextField.alloc().initWithFrame_(
            NSRect(NSPoint(20, 48), NSSize(340, 34))
        )
        hint.setStringValue_(
            "Нажмите клавишу на клавиатуре — система сразу зафиксирует её.\n(Для отмены нажмите кнопку ниже или Escape)"
        )
        hint.setFont_(NSFont.systemFontOfSize_(11.0))
        hint.setTextColor_(NSColor.secondaryLabelColor())
        hint.setAlignment_(NSTextAlignmentCenter)
        hint.setBezeled_(False)
        hint.setDrawsBackground_(False)
        hint.setEditable_(False)
        hint.setSelectable_(False)
        content.addSubview_(hint)

        # Cancel button
        cancel_btn = NSButton.alloc().initWithFrame_(
            NSRect(NSPoint(140, 14), NSSize(100, 28))
        )
        cancel_btn.setTitle_("Отмена")
        cancel_btn.setTarget_(self)
        cancel_btn.setAction_("onCancelHotkeyCapture:")
        cancel_btn.setBezelStyle_(NSRoundedBezelStyle)
        content.addSubview_(cancel_btn)

        self._capture_window = window
        self._is_capturing_key = True

        def _on_key_event(event):
            if not getattr(self, "_is_capturing_key", False):
                return event

            evt_type = event.type()
            vk = event.keyCode()

            is_down = False
            if evt_type == 10:  # KeyDown
                is_down = True
            elif evt_type == 12:  # FlagsChanged
                if vk in MODIFIER_MASKS:
                    mask = MODIFIER_MASKS[vk]
                    is_down = bool(event.modifierFlags() & mask)
                else:
                    is_down = True

            if not is_down:
                return event

            # Escape cancels capture
            if vk == 53:  # Escape
                AppHelper.callAfter(self._close_key_capture_window)
                return None

            self._is_capturing_key = False
            code_str, label = vk_to_hotkey_identifier_and_label(vk)
            logger.info(f"Captured hotkey: vk={vk} -> {code_str} ({label})")

            def finish_capture():
                if code_str == "right_cmd":
                    self.config.hotkey = "right_cmd"
                    self.config.custom_hotkey = None
                else:
                    self.config.custom_hotkey = code_str
                    self.config.hotkey = code_str
                self.config.save()
                self.hotkey_listener.update_hotkey(code_str)
                self._close_key_capture_window()
                self.rebuild_menu()

            AppHelper.callAfter(finish_capture)
            return None

        event_mask = NSEventMaskKeyDown | NSEventMaskFlagsChanged
        self._capture_local_monitor = (
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                event_mask, _on_key_event
            )
        )
        self._capture_global_monitor = (
            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                event_mask, _on_key_event
            )
        )

        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        window.makeKeyAndOrderFront_(None)

    def _clean_capture_monitors(self):
        self._is_capturing_key = False
        if getattr(self, "_capture_local_monitor", None) is not None:
            try:
                NSEvent.removeMonitor_(self._capture_local_monitor)
            except Exception as e:
                logger.debug(f"Local monitor removal error: {e}")
            self._capture_local_monitor = None
        if getattr(self, "_capture_global_monitor", None) is not None:
            try:
                NSEvent.removeMonitor_(self._capture_global_monitor)
            except Exception as e:
                logger.debug(f"Global monitor removal error: {e}")
            self._capture_global_monitor = None

    def _close_key_capture_window(self):
        self._clean_capture_monitors()
        win = getattr(self, "_capture_window", None)
        if win is not None:
            self._capture_window = None
            try:
                win.close()
            except Exception as e:
                logger.debug(f"Capture window close error: {e}")

    def onCancelHotkeyCapture_(self, sender):
        self._close_key_capture_window()

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
            alert.setInformativeText_(
                "Разрешение 'Универсальный доступ' уже предоставлено. Автовставка работает!"
            )
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

    def onUninstallAppDialog_(self, sender):
        alert = NSAlert.alloc().init()
        alert.setMessageText_("Вы действительно хотите полностью удалить GigaDictate?")
        alert.setInformativeText_(
            "Будет удалено приложение, сохраненная конфигурация, кэш моделей нейросети (~900 МБ) "
            "и сброшены системные разрешения (микрофон, управление вводом)."
        )
        alert.setAlertStyle_(2)  # NSAlertStyleCritical
        alert.addButtonWithTitle_("Да")
        alert.addButtonWithTitle_("Нет")

        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        response = alert.runModal()

        # Если пользователь выбрал "Нет" (вторая кнопка) или закрыл диалог
        if response != NSAlertFirstButtonReturn:
            logger.info("Uninstallation cancelled by user.")
            return

        self.execute_complete_uninstallation()

    def execute_complete_uninstallation(self):
        logger.info("Executing complete uninstallation...")

        # 1. Stop background components
        try:
            self.hotkey_listener.stop()
            self.hud.hideHUD()
            self.recorder.close()
            _WORKER_EXECUTOR.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"Error stopping components during uninstall: {e}")

        # 2. Determine running bundle path
        bundle = NSBundle.mainBundle()
        bundle_path = bundle.bundlePath() if bundle else ""
        safe_bundle = shlex.quote(str(bundle_path))
        pid = int(os.getpid())

        # 3. Spawn detached uninstallation script safely
        cleanup_script = f"""
sleep 0.5
while kill -0 {pid} 2>/dev/null; do sleep 0.1; done

# Remove running app bundle if applicable
if [ -n {safe_bundle} ] && [ -d {safe_bundle} ] && [[ {safe_bundle} == *".app"* ]]; then
    rm -rf {safe_bundle} 2>/dev/null || true
fi
rm -rf "/Applications/GigaDictate.app" 2>/dev/null || true
rm -rf "$HOME/Applications/GigaDictate.app" 2>/dev/null || true

# Unregister from LaunchServices
LSREG="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
if [ -x "$LSREG" ]; then
    "$LSREG" -u "/Applications/GigaDictate.app" 2>/dev/null || true
    if [ -n {safe_bundle} ]; then
        "$LSREG" -u {safe_bundle} 2>/dev/null || true
    fi
    "$LSREG" -gc 2>/dev/null || true
fi

# Reset macOS TCC Permissions
BUNDLE_ID="com.salute.gigadictate"
tccutil reset Microphone "$BUNDLE_ID" 2>/dev/null || true
tccutil reset Accessibility "$BUNDLE_ID" 2>/dev/null || true
tccutil reset ListenEvent "$BUNDLE_ID" 2>/dev/null || true
tccutil reset SpeechRecognition "$BUNDLE_ID" 2>/dev/null || true
tccutil reset All "$BUNDLE_ID" 2>/dev/null || true

# Purge config, logs and ~900MB neural network cache
rm -rf "$HOME/.config/giga_dictate"
rm -rf "$HOME/.cache/gigaam"
rm -f "$HOME/Library/Logs/DiagnosticReports/GigaDictate"* 2>/dev/null || true
rm -f "$HOME/Library/Application Support/CrashReporter/GigaDictate"* 2>/dev/null || true
"""

        try:
            subprocess.Popen(
                ["/bin/bash", "-c", cleanup_script],
                start_new_session=True,
                close_fds=True,
            )
        except Exception as e:
            logger.error(f"Failed to spawn uninstall script: {e}")

        # 4. Terminate process immediately
        NSApplication.sharedApplication().terminate_(self)

    def onRestartApp_(self, sender):
        self.restart_application()

    def restart_application(self):
        logger.info("Restarting application...")
        self.hotkey_listener.stop()
        self.hud.hideHUD()
        self.recorder.close()
        _WORKER_EXECUTOR.shutdown(wait=False)
        from AppKit import NSURL, NSWorkspace

        app_url = NSURL.fileURLWithPath_("/Applications/GigaDictate.app")
        if app_url:
            NSWorkspace.sharedWorkspace().openURL_(app_url)
        NSApplication.sharedApplication().terminate_(self)

    def onQuitApp_(self, sender):
        self.hotkey_listener.stop()
        self.hud.hideHUD()
        self.recorder.close()
        _WORKER_EXECUTOR.shutdown(wait=False)
        NSApplication.sharedApplication().terminate_(self)

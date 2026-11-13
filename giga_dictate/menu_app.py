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
    NSBitmapImageRep,
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
    NSURL,
    NSVariableStatusItemLength,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskTitled,
    NSWorkspace,
)
from Foundation import NSBundle, NSObject, NSRunLoop, NSRunLoopCommonModes, NSTimer
from PyObjCTools import AppHelper

from . import __version__
from .audio import AudioRecorder
from .config import AppConfig, is_launch_at_login_enabled, set_launch_at_login
from .ducking import AudioDucker
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
from .session import DictationSession, DictationState, RecoveryText

logger = logging.getLogger("giga_dictate.menu_app")


def _set_thread_qos_user_interactive():
    """Sets current thread QoS to Darwin QOS_CLASS_USER_INTERACTIVE (0x21) for Performance cores scheduling."""
    try:
        import ctypes
        import ctypes.util

        libc_name = ctypes.util.find_library("c")
        if libc_name:
            libc = ctypes.CDLL(libc_name)
            # QOS_CLASS_USER_INTERACTIVE = 0x21
            libc.pthread_set_qos_class_self_np(0x21, 0)
    except Exception as e:
        logger.debug(f"Could not set thread QoS: {e}")


_WORKER_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="GigaWorker",
    initializer=_set_thread_qos_user_interactive,
)
_PREVIEW_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="GigaPreview",
    initializer=_set_thread_qos_user_interactive,
)

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


def _load_menu_icon(path_1x: str, is_template: bool = False) -> NSImage | None:
    """Loads a menu icon supporting both 1x and @2x Retina representations."""
    if not os.path.exists(path_1x):
        return None

    rep_1x = NSBitmapImageRep.imageRepWithContentsOfFile_(path_1x)
    if not rep_1x:
        return None

    width_pt = rep_1x.pixelsWide()
    height_pt = rep_1x.pixelsHigh()
    rep_1x.setSize_(NSSize(width_pt, height_pt))

    img = NSImage.alloc().initWithSize_(NSSize(width_pt, height_pt))
    img.addRepresentation_(rep_1x)

    base, ext = os.path.splitext(path_1x)
    path_2x = f"{base}@2x{ext}"
    if os.path.exists(path_2x):
        rep_2x = NSBitmapImageRep.imageRepWithContentsOfFile_(path_2x)
        if rep_2x:
            rep_2x.setSize_(NSSize(width_pt, height_pt))
            img.addRepresentation_(rep_2x)

    if is_template:
        img.setTemplate_(True)
    return img


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

        self._shutting_down = False
        self.session = DictationSession()
        self.recovery = RecoveryText()
        self._recovery_timer = None
        self._preview_future = None
        self.config = AppConfig.load()
        self.sounds = SoundManager(enabled=self.config.sound_feedback)
        self.recorder = AudioRecorder(device=self.config.audio_device)
        self.engine = GigaAMEngine(
            model_name=self.config.model_name,
            device=self.config.device,
        )
        self.inserter = TextInserter()
        self.ducker = AudioDucker()
        from .meeting_ui import MeetingUI
        self.meeting_ui = MeetingUI.alloc().initWithController_(self)

        self.status_item = None
        self.menu = None
        self.menu_icon = None
        self.menu_icon_rec = None
        self._status_menu_item = None
        self._toggle_menu_item = None

        # Voice Input HUD
        self.hud = VoiceInputHUD.alloc().init()
        self.hud.setAudioRecorder_(self.recorder)
        if hasattr(self.hud, "setLivePreviewEnabled_"):
            self.hud.setLivePreviewEnabled_(self.config.live_hud_preview)
        if hasattr(self.hud, "setGlassStyle_"):
            self.hud.setGlassStyle_(self.config.hud_glass_style)
        self._perm_check_timer = None
        self._preview_timer = None
        self._is_preview_running = False
        self._preview_session_id = 0

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

        # Model download and loading tracking
        self._model_progress_str = ""
        self._model_error = None
        self._model_reloading = False

        return self

    def applicationDidFinishLaunching_(self, notification):
        # Configure Menu Bar item
        status_bar = NSStatusBar.systemStatusBar()
        self.status_item = status_bar.statusItemWithLength_(NSVariableStatusItemLength)
        self.status_item.setVisible_(True)
        _GLOBAL_STATUS_ITEMS.append(self.status_item)

        # Load menu icons from resources (multi-representation for sharp Retina rendering)
        res_dir = os.path.join(os.path.dirname(__file__), "resources")
        self.menu_icon = _load_menu_icon(
            os.path.join(res_dir, "menu_icon.png"), is_template=True
        )
        self.menu_icon_rec = _load_menu_icon(
            os.path.join(res_dir, "menu_icon_rec.png"), is_template=False
        )

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
        if self.hotkey_listener._tap is None:
            if self.hotkey_listener.start():
                logger.info(
                    "Accessibility active: Event tap successfully connected! Rebuilding menu..."
                )
                self.rebuild_menu()

    def _on_model_progress(self, downloaded, total):
        pct = int(downloaded / total * 100) if total > 0 else 0
        mb_down = downloaded / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        self._model_progress_str = f"Скачивание модели: {pct}% ({int(mb_down)}/{int(mb_total)} МБ)"
        AppHelper.callAfter(
            self.update_status_title, f" {pct}%"
        )
        if self._status_menu_item and not self.engine.is_loaded and not self._model_error:
            AppHelper.callAfter(
                self._status_menu_item.setTitle_, f"⏳ {self._model_progress_str}"
            )

    def _initial_model_load(self):
        self._model_error = None
        self._model_progress_str = "Подготовка к загрузке..."
        AppHelper.callAfter(self.update_status_title, " ⏳")
        AppHelper.callAfter(self.rebuild_menu)

        try:
            import gigaam
            if hasattr(gigaam, "set_download_progress_callback"):
                gigaam.set_download_progress_callback(self._on_model_progress)
        except Exception as e:
            logger.debug(f"Progress callback setup error: {e}")

        success = self.engine.load()
        if success:
            self._model_error = None
            self._model_progress_str = ""
            AppHelper.callAfter(self.onModelLoaded_, None)
        else:
            self._model_error = self.engine.load_error or "Неизвестная ошибка загрузки"
            AppHelper.callAfter(
                self.onModelError_, self._model_error
            )

    def onModelLoaded_(self, _):
        self._model_reloading = False
        self._model_error = None
        self._model_progress_str = ""
        self.update_status_title("", recording=self.session.state == DictationState.RECORDING)
        self.rebuild_menu()
        logger.info("Ready for dictation.")

    def onModelError_(self, error_str):
        self._model_reloading = False
        self._model_error = error_str
        self.update_status_title(" ⚠️")
        self.rebuild_menu()
        logger.error(f"Model load error: {error_str}")

    @objc.IBAction
    def onRetryModelLoad_(self, sender):
        if self.session.busy or self._model_reloading or self._meeting_busy():
            return
        self._model_reloading = True
        logger.info("Retrying model load by user request...")
        self._model_error = None
        self._model_progress_str = "Перезапуск загрузки..."
        self.update_status_title(" ⏳")
        self.rebuild_menu()
        _WORKER_EXECUTOR.submit(self._initial_model_load)

    def update_status_title(self, title: str = "", recording: bool = False):
        if self.status_item and self.status_item.button():
            btn = self.status_item.button()
            if recording:
                if self.menu_icon_rec:
                    btn.setImage_(self.menu_icon_rec)
                btn.setImagePosition_(NSImageOnly)
                btn.setTitle_("")
            elif self._model_error or self.session.state == DictationState.ERROR:
                if self.menu_icon:
                    btn.setImage_(self.menu_icon)
                btn.setImagePosition_(NSImageLeft)
                btn.setTitle_(" ⚠️")
            elif not self.engine.is_loaded:
                if self.menu_icon:
                    btn.setImage_(self.menu_icon)
                btn.setImagePosition_(NSImageLeft)
                btn.setTitle_(title if title else " ⏳")
            else:
                # Ready state: clean icon only, takes ~24px, never hidden by MacBook notch!
                if self.menu_icon:
                    btn.setImage_(self.menu_icon)
                btn.setImagePosition_(NSImageOnly)
                btn.setTitle_("")
            btn.setToolTip_(self.session.error if self.session.state == DictationState.ERROR else "GigaDictate — Голосовой ввод GigaAM-v3")
            self.status_item.setVisible_(True)

    def rebuild_menu(self):
        menu = NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)

        # 1. Header / Status
        meeting_busy = self._meeting_busy()
        if meeting_busy:
            status_text = "🔴 Встреча: запись или обработка"
        elif self.session.state == DictationState.TRANSCRIBING:
            status_text = "⏳ Распознавание..."
        elif self.session.state == DictationState.INSERTING:
            status_text = "⏳ Отправка вставки..."
        elif self.session.state == DictationState.ERROR:
            status_text = f"⚠️ {self.session.error}"
        elif self.session.state == DictationState.RECORDING:
            status_text = "🔴 Идет запись голоса..."
        elif self._model_error:
            status_text = f"⚠️ Ошибка: {self._model_error[:40]}"
        elif not self.engine.is_loaded:
            status_text = f"⏳ {self._model_progress_str or 'Загрузка модели GigaAM-v3...'}"
        elif self.hotkey_listener._tap is None:
            status_text = "⚠️ Клавиша не активна (нужен доступ в Настройках)"
        else:
            mode_desc = (
                "Удерживайте"
                if self.config.hotkey_mode == "push_to_talk"
                else "Нажмите"
            )
            hk_label = get_hotkey_display_name(self.config.hotkey)
            if self.config.hotkey == "right_cmd":
                hk_label = "ПРАВЫЙ Command (кнопка ⌘ справа от пробела)"
            status_text = (
                f"🟢 Готов к вводу ({mode_desc} {hk_label})"
            )

        status_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            status_text, None, ""
        )
        status_item.setEnabled_(False)
        menu.addItem_(status_item)
        self._status_menu_item = status_item

        if self._model_error:
            retry_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "🔄 Повторить загрузку модели", "onRetryModelLoad:", ""
            )
            retry_item.setTarget_(self)
            menu.addItem_(retry_item)

        if self.hotkey_listener._tap is None and self.engine.is_loaded and not self._model_error:
            fix_perm_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "👉 Активировать перехват клавиши в Настройках macOS...", "onFixAccessibility:", ""
            )
            fix_perm_item.setTarget_(self)
            menu.addItem_(fix_perm_item)

        # 2. Manual toggle recording
        action_title = (
            "⏹️ Остановить запись" if self.session.state == DictationState.RECORDING else "🎙️ Начать запись"
        )
        toggle_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            action_title, "onMenuToggleRecord:", ""
        )
        toggle_item.setTarget_(self)
        toggle_item.setEnabled_(not meeting_busy and self.engine.is_loaded and not self._model_reloading and self.session.state not in (DictationState.TRANSCRIBING, DictationState.INSERTING))
        menu.addItem_(toggle_item)
        self._toggle_menu_item = toggle_item

        meeting = getattr(self, "meeting_ui", None)
        if meeting is not None:
            for title, action, enabled in (
                ("⏹ Завершить встречу" if meeting.session.busy else "🎙 Начать встречу / созвон…",
                 "onStart:", not self.session.busy and not self._model_reloading and not meeting.preparing_assets),
                ("Открыть стенограмму встречи", "onShowResult:", meeting.result is not None),
                ("Подготовить модели встреч…", "onPrepareModels:", not meeting_busy and not self.session.busy),
                ("Записать мой голос (20 сек)…", "onEnroll:", not meeting_busy and not self.session.busy and not self._model_reloading),
                ("Удалить мой голосовой профиль", "onDeleteProfile:", not meeting_busy),
            ):
                item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, "")
                item.setTarget_(meeting)
                item.setEnabled_(enabled)
                menu.addItem_(item)

        menu.addItem_(NSMenuItem.separatorItem())

        for title, action in (
            ("Повторить вставку", "onRetryInsertion:"),
            ("Скопировать последний текст", "onCopyLastText:"),
            ("Очистить последний текст", "onClearLastText:"),
        ):
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, "")
            item.setTarget_(self)
            item.setEnabled_(bool(self.recovery.get()) and not self.session.busy and not meeting_busy)
            menu.addItem_(item)
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
        model_menu_item.setEnabled_(not self.session.busy and not self._model_reloading and not meeting_busy)
        model_menu_item.setSubmenu_(model_sub)
        menu.addItem_(model_menu_item)

        # 4. Hotkey Mode Submenu
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
        mode_menu_item.setEnabled_(not self.session.busy and not meeting_busy)
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
            "Изменить клавишу...", "onCustomHotkeyDialog:", ""
        )
        record_item.setTarget_(self)
        hotkey_sub.addItem_(record_item)

        hotkey_menu_item.setEnabled_(not self.session.busy and not meeting_busy)
        hotkey_menu_item.setSubmenu_(hotkey_sub)
        menu.addItem_(hotkey_menu_item)

        menu.addItem_(NSMenuItem.separatorItem())

        # 6. Preferences Checkboxes
        preview_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Живой предпросмотр в HUD", "onToggleLiveHudPreview:", ""
        )
        preview_item.setTarget_(self)
        preview_item.setState_(1 if self.config.live_hud_preview else 0)
        menu.addItem_(preview_item)

        # 6.1 HUD Glass Transparency Submenu
        glass_menu_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Стиль стекла HUD", None, ""
        )
        glass_sub = NSMenu.alloc().init()
        glass_options = [
            ("glass", "Стеклянный (Высокая прозрачность)"),
            ("balanced", "Сбалансированный (Матовое стекло)"),
            ("matte", "Матовый (Плотный контраст)"),
        ]
        for style_key, title in glass_options:
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                title, "onSelectGlassStyle:", ""
            )
            item.setTarget_(self)
            item.setRepresentedObject_(style_key)
            item.setState_(1 if self.config.hud_glass_style == style_key else 0)
            glass_sub.addItem_(item)

        glass_menu_item.setSubmenu_(glass_sub)
        glass_menu_item.setEnabled_(self.config.live_hud_preview)
        menu.addItem_(glass_menu_item)

        sound_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Звуковые сигналы", "onToggleSound:", ""
        )
        sound_item.setTarget_(self)
        sound_item.setState_(1 if self.config.sound_feedback else 0)
        menu.addItem_(sound_item)

        ducking_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Приглушать звук Mac при записи", "onToggleAudioDucking:", ""
        )
        ducking_item.setTarget_(self)
        ducking_item.setState_(1 if getattr(self.config, "audio_ducking", True) else 0)
        menu.addItem_(ducking_item)

        space_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Добавлять пробел после текста", "onToggleSpace:", ""
        )
        space_item.setTarget_(self)
        space_item.setState_(1 if self.config.add_trailing_space else 0)
        menu.addItem_(space_item)

        login_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Запуск при старте системы", "onToggleLaunchAtLogin:", ""
        )
        login_item.setTarget_(self)
        login_item.setState_(1 if is_launch_at_login_enabled() else 0)
        menu.addItem_(login_item)

        # 7. Check Permissions
        perm_has = TextInserter.is_accessibility_granted() and self.hotkey_listener._tap is not None
        perm_label = (
            "Универсальный доступ: Активен ✅"
            if perm_has
            else "Проверить Универсальный доступ ⚠️"
        )
        perm_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            perm_label, "onCheckPermissions:", ""
        )
        perm_item.setTarget_(self)
        menu.addItem_(perm_item)

        menu.addItem_(NSMenuItem.separatorItem())

        # 8. About, Uninstall & Quit
        about_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "О программе GigaDictate...", "onAboutDialog:", ""
        )
        about_item.setTarget_(self)
        menu.addItem_(about_item)

        uninstall_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Полное удаление...", "onUninstallAppDialog:", ""
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
        # Also updates availability of recovery commands and model/hotkey changes.
        self.rebuild_menu()

    def _meeting_busy(self):
        meeting = getattr(self, "meeting_ui", None)
        return meeting is not None and meeting.busy

    # --- Actions ---

    def onMenuToggleRecord_(self, sender):
        if self.session.state == DictationState.RECORDING:
            self.stop_and_transcribe()
        else:
            self.start_recording(manual=True)

    def on_hotkey_start(self):
        """Called by hotkey listener on background thread."""
        logger.info("on_hotkey_start -> dispatching start_recording to main thread")
        AppHelper.callAfter(self.start_recording)

    def on_hotkey_stop(self):
        """Called by hotkey listener on background thread."""
        logger.info("on_hotkey_stop -> dispatching stop_and_transcribe to main thread")
        AppHelper.callAfter(self.stop_and_transcribe)

    def start_recording(self, manual: bool = False):
        if self._meeting_busy():
            self.hotkey_listener.reset_state()
            return
        if self._model_reloading:
            self.hotkey_listener.reset_state()
            self.sounds.play_cancel()
            return
        if not self.session.begin():
            self.sounds.play_cancel()
            # A rejected hotkey START must not leave toggle mode logically active.
            if self.session.state != DictationState.RECORDING:
                self.hotkey_listener.reset_state()
            return
        generation = self.session.generation
        try:
            if not self.recorder.start():
                self._finish_session(generation, "Не удалось включить микрофон")
                return
            self.hotkey_listener.set_recording(True)
            self._preview_session_id += 1
            self.engine.reset_preview_cache()
            if self.config.audio_ducking:
                self.ducker.duck(target_volume=self.config.ducking_volume)
            self.update_status_title("Запись...", recording=True)
            self.sounds.play_start()
            self.hud.showRecording()
            self.update_menu_status()
            if self.config.live_hud_preview:
                self._start_preview_timer()
            logger.info("session_started session=%s", generation)
        except Exception:
            logger.warning("recording_start_failed session=%s", generation)
            self._finish_session(generation, "Ошибка запуска записи")

    def stop_and_transcribe(self):
        if self.session.state != DictationState.RECORDING:
            return
        generation = self.session.generation
        self.session.state = DictationState.TRANSCRIBING
        self._preview_session_id += 1
        self._stop_preview_timer()
        self.ducker.unduck()
        self.hotkey_listener.reset_state()
        try:
            audio, duration = self.recorder.stop()
            self.sounds.play_stop()
            self.hud.showTranscribing()
            self.update_status_title("Обработка...")
            self.update_menu_status()
        except Exception:
            self._finish_session(generation, "Ошибка остановки записи")
            return
        logger.info("recording_stopped session=%s duration_s=%.3f", generation, duration)

        def worker():
            if len(audio) == 0:
                AppHelper.callAfter(self._transcription_ready, generation, "",
                    "Микрофон не передаёт звук. Повторите запись и проверьте устройство ввода macOS.")
                return
            if duration < self.config.min_duration:
                AppHelper.callAfter(self._transcription_ready, generation, "", "Запись слишком короткая")
                return
            t0 = time.monotonic()
            try:
                if not self.engine.is_loaded:
                    self.engine.wait_loaded(timeout=15.0)
                logger.info("model_wait session=%s elapsed_ms=%.1f", generation, (time.monotonic() - t0) * 1000)
                if not self.engine.is_loaded:
                    raise RuntimeError("model_unavailable")
                t0 = time.monotonic()
                text = self.engine.transcribe(audio)
                logger.info("transcription_done session=%s elapsed_ms=%.1f chars=%s",
                            generation, (time.monotonic() - t0) * 1000, len(text))
                AppHelper.callAfter(self._transcription_ready, generation, text, "" if text else "Речь не распознана")
            except Exception:
                logger.warning("transcription_failed session=%s", generation)
                AppHelper.callAfter(self._transcription_ready, generation, "", "Ошибка распознавания")
        try:
            _WORKER_EXECUTOR.submit(worker)
        except RuntimeError:
            self._finish_session(generation, "Обработка недоступна")

    def _transcription_ready(self, generation, text, error):
        if not self.session.accepts(generation) or self.session.state != DictationState.TRANSCRIBING:
            return
        if error:
            self._finish_session(generation, error)
            return
        self.recovery.remember(text)
        self._arm_recovery_expiry()
        self._insert_result(generation, text)

    def _insert_result(self, generation, text):
        if not self.session.accepts(generation):
            return
        self.session.state = DictationState.INSERTING
        self.update_menu_status()
        try:
            future = self.inserter.insert_text(
                text, add_trailing_space=self.config.add_trailing_space,
                restore_clipboard=self.config.restore_clipboard,
                conceal_clipboard=self.config.conceal_clipboard, delay=self.config.paste_delay,
            )
            def done(completed):
                AppHelper.callAfter(self._insertion_ready, generation, completed.result())
            future.add_done_callback(done)
        except Exception:
            self._finish_session(generation, "Не удалось отправить вставку; текст доступен в меню")

    def _insertion_ready(self, generation, result):
        if not self.session.accepts(generation) or self.session.state != DictationState.INSERTING:
            return
        if result.event_posted:
            self.sounds.play_success()
            self._finish_session(generation)
        else:
            reasons = {
                "accessibility_denied": "Нужен Универсальный доступ",
                "modifiers_held": "Отпустите клавиши-модификаторы",
                "target_changed": "Активное приложение изменилось",
                "clipboard_changed": "Буфер обмена изменился",
            }
            message = reasons.get(result.reason, "Не удалось завершить вставку")
            self._finish_session(generation, message)

    def _finish_session(self, generation, error=""):
        if not self.session.finish(generation, error):
            return
        self._stop_preview_timer()
        self.ducker.unduck()
        if self.recorder.is_recording:
            try:
                self.recorder.stop()
            except Exception:
                logger.warning("recording_cleanup_failed session=%s", generation)
        self.hotkey_listener.reset_state()
        self.engine.reset_preview_cache()
        self.hud.hideHUD()
        self.update_status_title("")
        self.update_menu_status()
        if error:
            self.sounds.play_cancel()
        logger.info("session_finished session=%s status=%s elapsed_ms=%.1f", generation,
                    "error" if error else "event_posted", (time.monotonic() - self.session.started_at) * 1000)

    def _arm_recovery_expiry(self):
        if self._recovery_timer is not None:
            self._recovery_timer.invalidate()
        self._recovery_timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            self.recovery.ttl, self, "onRecoveryExpiry:", None, False)
        NSRunLoop.mainRunLoop().addTimer_forMode_(self._recovery_timer, NSRunLoopCommonModes)

    def onRecoveryExpiry_(self, timer):
        if timer is not self._recovery_timer:
            return
        self.recovery.get()
        self._recovery_timer = None
        self.update_menu_status()

    def onRetryInsertion_(self, sender):
        if self._meeting_busy():
            return
        text = self.recovery.get()
        if not text or not self.session.begin():
            self.sounds.play_cancel()
            return
        # Let the menu close before capturing the active target and sending keys.
        generation = self.session.generation
        self.session.state = DictationState.INSERTING
        AppHelper.callLater(0.15, self._insert_result, generation, text)

    def onCopyLastText_(self, sender):
        if self.session.busy or self._meeting_busy():
            return
        result = self.inserter.copy_text(self.recovery.get(), self.config.conceal_clipboard)
        if not result:
            self.session.error = "Не удалось скопировать текст"
            self.session.state = DictationState.ERROR
        else:
            self.session.error = ""
            self.session.state = DictationState.IDLE
        self.update_status_title("")
        self.update_menu_status()

    def onClearLastText_(self, sender):
        self.recovery.clear()
        if self._recovery_timer is not None:
            self._recovery_timer.invalidate()
            self._recovery_timer = None
        self.update_menu_status()

    def _start_preview_timer(self):
        self._stop_preview_timer()
        self._is_preview_running = False
        # Tick every 0.35s (~2.8 times per second) for real-time live preview
        self._preview_timer = (
            NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
                0.35, self, "onPreviewTimerTick:", None, True
            )
        )
        NSRunLoop.mainRunLoop().addTimer_forMode_(
            self._preview_timer, NSRunLoopCommonModes
        )

    def _stop_preview_timer(self):
        if getattr(self, "_preview_timer", None) is not None:
            try:
                self._preview_timer.invalidate()
            except Exception as e:
                logger.debug(f"Preview timer invalidation error: {e}")
            self._preview_timer = None
        self._is_preview_running = False

    def onPreviewTimerTick_(self, timer):
        if not self.recorder.is_recording or not self.config.live_hud_preview:
            return
        if not self.engine.is_loaded or getattr(self, "_is_preview_running", False):
            return

        if self._preview_future is not None and not self._preview_future.done():
            return
        self._is_preview_running = True
        session_id = self._preview_session_id

        def _preview_worker():
            try:
                _set_thread_qos_user_interactive()
                if not self.recorder.is_recording or session_id != getattr(self, "_preview_session_id", 0):
                    return
                t0 = time.monotonic()
                offset = self.engine.preview_offset
                snapshot = self.recorder.get_stream_snapshot_from(offset)
                if len(snapshot) < int(0.4 * self.recorder.sample_rate):
                    return
                text = self.engine.transcribe_preview(
                    snapshot, offset=offset,
                    cancelled=lambda: session_id != self._preview_session_id or not self.recorder.is_recording,
                )
                logger.debug("preview_done session=%s elapsed_ms=%.1f", session_id, (time.monotonic() - t0) * 1000)
                if (
                    text
                    and self.recorder.is_recording
                    and session_id == getattr(self, "_preview_session_id", 0)
                ):
                    AppHelper.callAfter(self._deliver_preview_text, text, session_id)
            except Exception:
                logger.debug("preview_worker_failed")
            finally:
                if session_id == getattr(self, "_preview_session_id", 0):
                    self._is_preview_running = False

        try:
            self._preview_future = _PREVIEW_EXECUTOR.submit(_preview_worker)
        except Exception as e:
            logger.debug(f"Failed to submit preview worker: {e}")
            self._is_preview_running = False

    def _deliver_preview_text(self, text: str, session_id: int):
        if (
            session_id == getattr(self, "_preview_session_id", 0)
            and self.recorder.is_recording
            and getattr(self.hud, "state", None) == "recording"
        ):
            self.hud.setPreviewText_(text)

    def onSelectModel_(self, sender):
        if self.session.busy or self._model_reloading or self._meeting_busy():
            return
        new_model = sender.representedObject()
        if new_model == self.config.model_name:
            return
        self._model_reloading = True
        self.config.model_name = new_model
        self.config.save()
        self.update_status_title(f"⏳ Загрузка {new_model}...")
        self.rebuild_menu()

        def _reload():
            success = self.engine.load(model_name=new_model)
            AppHelper.callAfter(self.onModelLoaded_ if success else self.onModelError_,
                                None if success else "Не удалось загрузить модель")

        _WORKER_EXECUTOR.submit(_reload)

    def onSelectDevice_(self, sender):
        if self.session.busy or self._model_reloading or self._meeting_busy():
            return
        new_device = sender.representedObject()
        if new_device == self.config.device:
            return
        self._model_reloading = True
        self.config.device = new_device
        self.config.save()
        self.update_status_title(f"⏳ Переключение на {new_device}...")
        self.rebuild_menu()

        def _reload():
            success = self.engine.load(device=new_device)
            AppHelper.callAfter(self.onModelLoaded_ if success else self.onModelError_,
                                None if success else "Не удалось переключить устройство")

        _WORKER_EXECUTOR.submit(_reload)

    def onSelectMode_(self, sender):
        if self.session.busy or self._meeting_busy():
            return
        new_mode = sender.representedObject()
        self.config.hotkey_mode = new_mode
        self.config.save()
        self.hotkey_listener.update_hotkey(self.config.hotkey, mode=new_mode)
        self.rebuild_menu()

    def onSelectDefaultHotkey_(self, sender):
        if self.session.busy or self._meeting_busy():
            return
        if self.config.hotkey == "right_cmd":
            return
        self.config.hotkey = "right_cmd"
        self.config.custom_hotkey = None
        self.config.save()
        self.hotkey_listener.update_hotkey("right_cmd")
        self.rebuild_menu()

    def onSelectCustomHotkey_(self, sender):
        if self.session.busy or self._meeting_busy():
            return
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
        if self.session.busy or self._meeting_busy():
            return
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

    def onToggleLiveHudPreview_(self, sender):
        self.config.live_hud_preview = not self.config.live_hud_preview
        self.config.save()
        if hasattr(self.hud, "setLivePreviewEnabled_"):
            self.hud.setLivePreviewEnabled_(self.config.live_hud_preview)
        if self.recorder.is_recording:
            if self.config.live_hud_preview:
                self._start_preview_timer()
            else:
                self._stop_preview_timer()
        self.rebuild_menu()

    def onSelectGlassStyle_(self, sender):
        style_key = sender.representedObject()
        if style_key and style_key in ("glass", "balanced", "matte"):
            self.config.hud_glass_style = str(style_key)
            self.config.save()
            if hasattr(self.hud, "setGlassStyle_"):
                self.hud.setGlassStyle_(self.config.hud_glass_style)
            self.rebuild_menu()

    def onToggleSound_(self, sender):
        self.config.sound_feedback = not self.config.sound_feedback
        self.sounds.enabled = self.config.sound_feedback
        self.config.save()
        self.rebuild_menu()

    def onToggleAudioDucking_(self, sender):
        self.config.audio_ducking = not getattr(self.config, "audio_ducking", True)
        self.config.save()
        if not self.config.audio_ducking and self.ducker.is_ducked:
            self.ducker.unduck()
        self.rebuild_menu()

    def onToggleSpace_(self, sender):
        self.config.add_trailing_space = not self.config.add_trailing_space
        self.config.save()
        self.rebuild_menu()

    def onToggleLaunchAtLogin_(self, sender):
        current = is_launch_at_login_enabled()
        new_val = not current
        set_launch_at_login(new_val)
        self.config.launch_at_login = new_val
        self.config.save()
        self.rebuild_menu()

    def onFixAccessibility_(self, sender):
        alert = NSAlert.alloc().init()
        alert.setMessageText_("Активация горячей клавиши")
        alert.setInformativeText_(
            "macOS требует активный 'Универсальный доступ' для работы глобальной клавиши.\n\n"
            "Если тумблер для GigaDictate уже включен в настройках:\n"
            "1. Выключите тумблер GigaDictate и включите его обратно (или удалите кнопкой '-' и добавьте снова).\n"
            "2. Перехват клавиши подключится автоматически без перезапуска приложения."
        )
        alert.addButtonWithTitle_("Открыть Настройки macOS")
        alert.addButtonWithTitle_("Позже")
        if alert.runModal() == NSAlertFirstButtonReturn:
            TextInserter.request_accessibility()

    def onCheckPermissions_(self, sender):
        if not TextInserter.is_accessibility_granted():
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Требуется Универсальный доступ")
            alert.setInformativeText_(
                "macOS требует разрешение 'Универсальный доступ' (Accessibility) "
                "для симуляции нажатия Cmd+V и автовставки текста под курсор.\n\n"
                "Если разрешение не выдано, результат сохраняется на пять минут в памяти. "
                "Выберите «Скопировать последний текст» в меню и вставьте его вручную."
            )
            alert.addButtonWithTitle_("Открыть Настройки macOS")
            alert.addButtonWithTitle_("Позже")
            if alert.runModal() == NSAlertFirstButtonReturn:
                TextInserter.request_accessibility()
        else:
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Разрешения в порядке")
            alert.setInformativeText_(
                "Разрешение 'Универсальный доступ' предоставлено. Отправка событий ввода доступна."
            )
            alert.addButtonWithTitle_("OK")
            alert.runModal()

    def onAboutDialog_(self, sender):
        alert = NSAlert.alloc().init()
        alert.setMessageText_("О программе GigaDictate")
        alert.setInformativeText_(
            f"GigaDictate {__version__} — локальное высокоскоростное голосовое распознавание для macOS.\n\n"
            "• Основано на моделях Salute Developers GigaAM-v3\n"
            "• Ускорение на Apple Silicon (MPS / Metal)\n"
            "• Автоматическая пунктуация и капитализация\n"
            "• Работа в фоне и вставка под курсор"
        )
        alert.addButtonWithTitle_("OK")
        alert.runModal()

    def onUninstallAppDialog_(self, sender):
        meeting = getattr(self, "meeting_ui", None)
        if meeting and not meeting.request_quit():
            return
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
        meeting = getattr(self, "meeting_ui", None)
        if meeting and not meeting.request_quit():
            return
        logger.info("Executing complete uninstallation...")
        self._shutdown()

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

# Purge config, logs, launch agents and ~900MB neural network cache
rm -f "$HOME/Library/LaunchAgents/com.salute.gigadictate.plist" 2>/dev/null || true
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
        meeting = getattr(self, "meeting_ui", None)
        if meeting and not meeting.request_quit():
            return
        logger.info("Restarting application...")
        self._shutdown()
        from AppKit import NSURL, NSWorkspace

        app_url = NSURL.fileURLWithPath_("/Applications/GigaDictate.app")
        if app_url:
            NSWorkspace.sharedWorkspace().openURL_(app_url)
        NSApplication.sharedApplication().terminate_(self)

    def _shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        meeting = getattr(self, "meeting_ui", None)
        if meeting:
            meeting.close()
        self.session.invalidate()
        self._preview_session_id += 1
        self.recovery.clear()
        if self._recovery_timer is not None:
            self._recovery_timer.invalidate()
            self._recovery_timer = None
        for name, cleanup in (
            ("preview", self._stop_preview_timer),
            ("clipboard", self.inserter.cancel_pending),
            ("hud", self.hud.close),
            ("volume", self.ducker.unduck),
            ("hotkey", self.hotkey_listener.stop),
            ("microphone", self.recorder.close),
        ):
            try:
                cleanup()
            except Exception:
                logger.warning("shutdown_cleanup_failed component=%s", name)
        self.engine.reset_preview_cache()
        _WORKER_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _PREVIEW_EXECUTOR.shutdown(wait=False, cancel_futures=True)

    def applicationWillTerminate_(self, notification):
        self._shutdown()

    def onQuitApp_(self, sender):
        NSApplication.sharedApplication().terminate_(self)

    def applicationShouldTerminate_(self, sender):
        from AppKit import NSTerminateCancel, NSTerminateNow
        meeting = getattr(self, "meeting_ui", None)
        if meeting and not meeting.request_quit():
            return NSTerminateCancel
        return NSTerminateNow

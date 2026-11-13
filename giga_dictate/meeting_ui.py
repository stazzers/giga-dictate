"""Native meeting controls. All entry points and event delivery run on AppKit."""
from pathlib import Path
import os
import tempfile
import threading

import objc
import AppKit as AK
from Foundation import NSObject, NSTimer, NSRunLoop, NSRunLoopCommonModes, NSNotificationCenter
from PyObjCTools import AppHelper

from .meeting_session import MeetingSession
from .meeting_types import MeetingOptions, MeetingState, timestamp
from .hud import NonactivatingHUDPanel


def label(text, frame):
    field = AK.NSTextField.labelWithString_(text)
    field.setFrame_(frame)
    return field


def button(title, action, target, frame):
    item = AK.NSButton.alloc().initWithFrame_(frame)
    item.setTitle_(title)
    item.setBezelStyle_(AK.NSRoundedBezelStyle)
    item.setTarget_(target)
    item.setAction_(action)
    return item


class MeetingUI(NSObject):
    def initWithController_(self, controller):
        self = objc.super(MeetingUI, self).init()
        if self is None:
            return None
        self.controller = controller
        from .config import CONFIG_DIR
        self.session = MeetingSession(controller.engine, Path(CONFIG_DIR) / "owner_voice_profile.json",
            notify=lambda event: AppHelper.callAfter(self.deliver, event))
        self.window = self.hud = self.timer = None
        self.result = None
        self.exported = False
        self.preparing_assets = False
        self.closed = False
        self._sharing = None
        AK.NSWorkspace.sharedWorkspace().notificationCenter().addObserver_selector_name_object_(
            self, "onSleep:", AK.NSWorkspaceWillSleepNotification, None)
        AK.NSWorkspace.sharedWorkspace().notificationCenter().addObserver_selector_name_object_(
            self, "onScreenChange:", AK.NSWorkspaceActiveSpaceDidChangeNotification, None)
        NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
            self, "onScreenChange:", AK.NSApplicationDidChangeScreenParametersNotification, None)
        return self

    @property
    def busy(self):
        return self.session.busy or self.preparing_assets

    @objc.python_method
    def alert(self, title, message):
        alert = AK.NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(message)
        alert.addButtonWithTitle_("OK")
        alert.runModal()

    @objc.python_method
    def available(self):
        if self.busy or self.controller.session.busy or self.controller._model_reloading:
            return False
        if not self.controller.engine.is_loaded:
            self.alert("Модель ещё не готова", "Дождитесь загрузки GigaAM.")
            return False
        if self.result is None and self.session.result is not None and self.session.options.mode != "enrollment":
            self.result = self.session.result
            self.exported = False
        if self.result is not None:
            self.onShowResult_(None)
            return False
        return True

    def onStart_(self, sender):
        if self.session.busy:
            self.session.stop()
            return
        if not self.available():
            return
        import sounddevice as sd
        from .meeting_models import find_models
        try:
            find_models()
            devices = [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
        except Exception:
            self.alert("Подготовка встреч", "Сначала выберите «Подготовить модели встреч» в меню. Проверьте доступность микрофона.")
            return
        alert = AK.NSAlert.alloc().init()
        alert.setMessageText_("Новая встреча")
        alert.setInformativeText_("Созвон захватывает весь системный звук. Используйте наушники: через динамики возможно эхо и дублирование реплик.")
        view = AK.NSView.alloc().initWithFrame_(AK.NSMakeRect(0, 0, 500, 160))
        mode = AK.NSPopUpButton.alloc().initWithFrame_pullsDown_(AK.NSMakeRect(0, 120, 500, 28), False)
        mode.addItemsWithTitles_(["Онлайн-созвон — микрофон + звук системы", "Живая встреча — один микрофон"])
        view.addSubview_(mode)
        mic = AK.NSPopUpButton.alloc().initWithFrame_pullsDown_(AK.NSMakeRect(0, 75, 500, 28), False)
        mic.addItemsWithTitles_(["Системный микрофон по умолчанию", *[name for _, name in devices]])
        view.addSubview_(mic)
        save = AK.NSButton.checkboxWithTitle_target_action_("Сохранять аудио в .m4a / .wav", None, None)
        save.setFrame_(AK.NSMakeRect(0, 35, 500, 28))
        view.addSubview_(save)
        view.addSubview_(label("Без сохранения аудио остаётся только в ограниченном буфере RAM.", AK.NSMakeRect(0, 0, 500, 25)))
        alert.setAccessoryView_(view)
        alert.addButtonWithTitle_("Начать")
        alert.addButtonWithTitle_("Отмена")
        if alert.runModal() != AK.NSAlertFirstButtonReturn:
            return
        online = mode.indexOfSelectedItem() == 0
        if online:
            from Quartz import CGPreflightScreenCaptureAccess, CGRequestScreenCaptureAccess
            if not CGPreflightScreenCaptureAccess():
                CGRequestScreenCaptureAccess()
                if not CGPreflightScreenCaptureAccess():
                    self.permission_error()
                    return
        path = None
        if save.state():
            panel = AK.NSSavePanel.savePanel()
            panel.setAllowedFileTypes_(["m4a", "wav"])
            panel.setNameFieldStringValue_("Встреча.m4a")
            if panel.runModal() != AK.NSModalResponseOK:
                return
            path = str(panel.URL().path())
            if Path(path).exists():
                self.alert("Выберите новое имя файла", "Аудиофайл встречи не должен перезаписывать существующий файл.")
                return
        index = mic.indexOfSelectedItem()
        device = devices[index - 1][0] if index else None
        self.controller.recorder.close()
        self.session.start(MeetingOptions("online" if online else "room", device, path))
        self.controller.hotkey_listener.reset_state()
        self.controller.rebuild_menu()

    @objc.python_method
    def permission_error(self):
        alert = AK.NSAlert.alloc().init()
        alert.setMessageText_("Нужен доступ к системному звуку")
        alert.setInformativeText_("Разрешите GigaDictate запись экрана и системного звука в настройках конфиденциальности. После выдачи разрешения может потребоваться перезапуск приложения.")
        alert.addButtonWithTitle_("Открыть настройки")
        alert.addButtonWithTitle_("Закрыть")
        if alert.runModal() == AK.NSAlertFirstButtonReturn:
            AK.NSWorkspace.sharedWorkspace().openURL_(AK.NSURL.URLWithString_(
                "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"))

    def onPrepareModels_(self, sender):
        if self.busy or self.controller.session.busy:
            return
        self.preparing_assets = True
        self.controller.rebuild_menu()
        self.controller.update_status_title("Подготовка моделей встреч…")

        def worker():
            try:
                from .meeting_models import prepare_models
                prepare_models()
                error = False
            except Exception:
                error = True
            AppHelper.callAfter(self._models_ready, error)
        threading.Thread(target=worker, name="GigaMeetingModels", daemon=True).start()

    @objc.python_method
    def _models_ready(self, error):
        self.preparing_assets = False
        if self.closed:
            return
        self.controller.update_status_title("")
        self.controller.rebuild_menu()
        self.alert("Модели встреч", "Не удалось подготовить модели. Проверьте интернет и зависимости; инструкция — docs/MEETING_MODE.md."
                   if error else "Модели готовы. Во время встреч сеть не используется.")

    def onEnroll_(self, sender):
        if not self.available():
            return
        alert = AK.NSAlert.alloc().init()
        alert.setMessageText_("Запись моего голоса — 20 секунд")
        alert.setInformativeText_("В тихой комнате говорите обычным голосом. Например, расскажите о планах на неделю. Приложение сохранит только голосовые векторы; запись заменит предыдущий профиль.")
        alert.addButtonWithTitle_("Записать")
        alert.addButtonWithTitle_("Отмена")
        if alert.runModal() == AK.NSAlertFirstButtonReturn:
            self.controller.recorder.close()
            self.session.start(MeetingOptions("enrollment", self.controller.config.audio_device))

    def onDeleteProfile_(self, sender):
        if not self.busy:
            try:
                self.session.profile.delete()
                self.alert("Профиль удалён", "Для комнатного режима будет использоваться обычная нумерация спикеров.")
            except OSError:
                self.alert("Не удалось удалить профиль", "Проверьте права на каталог конфигурации приложения.")

    def onSleep_(self, notification):
        if self.session.busy:
            self.session.fail("Mac переходит в сон; встреча завершена. После пробуждения начните новую запись.")

    def onScreenChange_(self, notification):
        if self.session.busy and self.hud is not None:
            self._show_hud()

    @objc.python_method
    def deliver(self, event):
        if self.closed or event["session_id"] != self.session.generation:
            return
        if event["kind"] == "state":
            state = event["state"]
            if state in (MeetingState.PREPARING, MeetingState.RECORDING, MeetingState.FINALIZING):
                self._show_hud()
            self.controller.rebuild_menu()
        elif event["kind"] == "complete":
            self._hide_hud()
            self.controller.update_status_title("")
            self.controller.hotkey_listener.reset_state()
            self.controller.rebuild_menu()
            if event["cancelled"]:
                return
            result = event["result"]
            if self.session.options.mode == "enrollment":
                self.alert("Голосовой профиль", "Профиль сохранён" if result.enrollment_complete else
                           "\n".join(result.errors or ["Запись профиля не завершена"]))
                result.clear()
                self.session.result = None
            else:
                self.result = result
                self.exported = False
                self.onShowResult_(None)
                self.controller.rebuild_menu()

    @objc.python_method
    def _show_hud(self):
        if self.hud is None:
            self.hud = NonactivatingHUDPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                AK.NSMakeRect(0, 0, 370, 68), AK.NSWindowStyleMaskBorderless |
                AK.NSWindowStyleMaskNonactivatingPanel, AK.NSBackingStoreBuffered, False)
            self.hud.setReleasedWhenClosed_(False)
            self.hud.setLevel_(AK.NSStatusWindowLevel)
            self.hud.setCollectionBehavior_(AK.NSWindowCollectionBehaviorCanJoinAllSpaces |
                                           AK.NSWindowCollectionBehaviorFullScreenAuxiliary)
            self.hud_text = label("Подготовка…", AK.NSMakeRect(15, 24, 265, 24))
            self.hud.contentView().addSubview_(self.hud_text)
            self.hud.contentView().addSubview_(button("Стоп", "onStop:", self, AK.NSMakeRect(285, 19, 75, 32)))
        screen = AK.NSScreen.mainScreen()
        if screen:
            frame = screen.visibleFrame()
            self.hud.setFrameOrigin_(AK.NSMakePoint(frame.origin.x + frame.size.width - 390,
                                                  frame.origin.y + 25))
        self.hud.orderFrontRegardless()
        if self.timer is None:
            self.timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
                .5, self, "onTick:", None, True)
            NSRunLoop.mainRunLoop().addTimer_forMode_(self.timer, NSRunLoopCommonModes)
        self.onTick_(None)

    def onTick_(self, timer):
        if self.closed or not self.session.busy:
            return
        state = self.session.state
        text = {MeetingState.PREPARING: "Подготовка встречи…",
                MeetingState.RECORDING: f"🔴 Запись встречи: {timestamp(self.session.elapsed)}",
                MeetingState.FINALIZING: "Обработка оставшихся реплик…"}.get(state, "Завершение…")
        self.hud_text.setStringValue_(text)
        self.controller.update_status_title(text, recording=state == MeetingState.RECORDING)

    @objc.python_method
    def _hide_hud(self):
        if self.timer:
            self.timer.invalidate()
            self.timer = None
        if self.hud:
            self.hud.orderOut_(None)

    def onStop_(self, sender):
        self.session.stop()

    def onShowResult_(self, sender):
        if self.result is None:
            return
        if self.window:
            self.window.makeKeyAndOrderFront_(None)
            return
        self.window = AK.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            AK.NSMakeRect(0, 0, 850, 640), AK.NSWindowStyleMaskTitled | AK.NSWindowStyleMaskClosable,
            AK.NSBackingStoreBuffered, False)
        self.window.setReleasedWhenClosed_(False)
        self.window.setTitle_("Стенограмма встречи")
        self.window.setDelegate_(self)
        content = self.window.contentView()
        scroll = AK.NSScrollView.alloc().initWithFrame_(AK.NSMakeRect(15, 180, 820, 445))
        scroll.setHasVerticalScroller_(True)
        self.text = AK.NSTextView.alloc().initWithFrame_(AK.NSMakeRect(0, 0, 800, 445))
        self.text.setEditable_(False)
        self.text.setFont_(AK.NSFont.systemFontOfSize_(14))
        self.text.setVerticallyResizable_(True)
        self.text.textContainer().setWidthTracksTextView_(True)
        scroll.setDocumentView_(self.text)
        content.addSubview_(scroll)
        self.speaker_keys = list(self.result.speakers)
        self.speaker_popup = AK.NSPopUpButton.alloc().initWithFrame_pullsDown_(AK.NSMakeRect(15, 135, 235, 28), False)
        content.addSubview_(self.speaker_popup)
        self.name_field = AK.NSTextField.alloc().initWithFrame_(AK.NSMakeRect(260, 136, 235, 26))
        self.name_field.setPlaceholderString_("Имя спикера")
        content.addSubview_(self.name_field)
        content.addSubview_(button("Переименовать", "onRename:", self, AK.NSMakeRect(505, 133, 155, 32)))
        content.addSubview_(label("Номер реплики:", AK.NSMakeRect(15, 98, 125, 24)))
        self.turn_field = AK.NSTextField.alloc().initWithFrame_(AK.NSMakeRect(145, 98, 70, 26))
        content.addSubview_(self.turn_field)
        content.addSubview_(button("Назначить выбранного спикера", "onAssign:", self, AK.NSMakeRect(225, 94, 280, 32)))
        for title, action, x, width in [("Копировать", "onCopy:", 15, 140),
                                      ("Сохранить .md / .txt", "onExport:", 165, 200),
                                      ("В Заметки…", "onShare:", 375, 145),
                                      ("Закрыть", "onClose:", 695, 140)]:
            content.addSubview_(button(title, action, self, AK.NSMakeRect(x, 25, width, 34)))
        self._refresh_result()
        self.window.center()
        AK.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)

    @objc.python_method
    def _refresh_result(self):
        # Numbers are editing handles only; exported transcript uses the agreed format.
        lines = [self.result.render().split("\n\n", 1)[0], ""]
        for turn in self.result.ordered_turns():
            speaker = self.result.speakers[turn.effective_speaker_id].name
            lines.append(f"{turn.id}. [{timestamp(turn.start)}] {speaker}: {turn.text}")
        if self.result.errors:
            lines.extend(["", *self.result.errors])
        self.text.setString_("\n".join(lines))
        selected = max(0, self.speaker_popup.indexOfSelectedItem())
        self.speaker_popup.removeAllItems()
        self.speaker_popup.addItemsWithTitles_([self.result.speakers[k].name for k in self.speaker_keys])
        self.speaker_popup.selectItemAtIndex_(selected)

    def onRename_(self, sender):
        try:
            key = self.speaker_keys[self.speaker_popup.indexOfSelectedItem()]
            self.result.rename_speaker(key, self.name_field.stringValue())
            self.exported = False
            self._refresh_result()
        except (ValueError, IndexError):
            self.alert("Имя спикера", "Выберите спикера и введите имя.")

    def onAssign_(self, sender):
        try:
            key = self.speaker_keys[self.speaker_popup.indexOfSelectedItem()]
            self.result.assign_speaker(int(self.turn_field.stringValue()), key)
            self.exported = False
            self._refresh_result()
        except (ValueError, IndexError, StopIteration):
            self.alert("Автор реплики", "Введите номер существующей реплики и выберите спикера.")

    def onCopy_(self, sender):
        board = AK.NSPasteboard.generalPasteboard()
        board.clearContents()
        if board.setString_forType_(self.result.render(), AK.NSPasteboardTypeString):
            self.exported = True

    def onExport_(self, sender):
        panel = AK.NSSavePanel.savePanel()
        panel.setAllowedFileTypes_(["md", "txt"])
        panel.setNameFieldStringValue_("Стенограмма.md")
        if panel.runModal() == AK.NSModalResponseOK:
            temporary = None
            try:
                path = Path(str(panel.URL().path()))
                fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".giga-transcript-")
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(self.result.render(markdown=path.suffix.lower() == ".md"))
                os.replace(temporary, path)
                self.exported = True
            except OSError:
                self.alert("Ошибка сохранения", "Не удалось сохранить стенограмму. Результат остаётся в памяти.")
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)

    def onShare_(self, sender):
        services = AK.NSSharingService.sharingServicesForItems_([self.result.render()])
        notes = next((service for service in services if "note" in str(service.name()).lower()), None)
        if notes:
            self._sharing = notes
            notes.performWithItems_([self.result.render()])
        else:
            self.alert("Заметки недоступны", "Скопируйте стенограмму или сохраните её в файл.")

    def onClose_(self, sender):
        if self.windowShouldClose_(self.window):
            self.window.close()

    def windowShouldClose_(self, window):
        if self.result and self.result.turns and not self.exported:
            alert = AK.NSAlert.alloc().init()
            alert.setMessageText_("Закрыть несохранённую стенограмму?")
            alert.setInformativeText_("Текст хранится только в памяти и будет удалён.")
            alert.addButtonWithTitle_("Удалить и закрыть")
            alert.addButtonWithTitle_("Отмена")
            if alert.runModal() != AK.NSAlertFirstButtonReturn:
                return False
        return True

    def windowWillClose_(self, notification):
        self.text.setString_("")
        self.name_field.setStringValue_("")
        self.window = self.text = self.name_field = self.turn_field = self.speaker_popup = None
        if self.result:
            self.result.clear()
        self.result = self.session.result = None
        self._sharing = None
        self.controller.rebuild_menu()

    @objc.python_method
    def request_quit(self):
        if self.session.busy:
            alert = AK.NSAlert.alloc().init()
            alert.setMessageText_("Идёт запись или обработка встречи")
            alert.setInformativeText_("Завершите встречу и сохраните результат перед выходом.")
            alert.addButtonWithTitle_("Завершить встречу")
            alert.addButtonWithTitle_("Отмена")
            if alert.runModal() == AK.NSAlertFirstButtonReturn:
                self.session.stop()
            return False
        if self.result is None and self.session.result is not None and self.session.options.mode != "enrollment":
            self.result = self.session.result
            self.exported = False
        if self.result and not self.windowShouldClose_(self.window):
            self.onShowResult_(None)
            return False
        return True

    @objc.python_method
    def close(self):
        self.closed = True
        self.session.cancel()
        self._hide_hud()
        AK.NSWorkspace.sharedWorkspace().notificationCenter().removeObserver_(self)
        NSNotificationCenter.defaultCenter().removeObserver_(self)
        if self.result:
            self.result.clear()
        if self.window:
            self.text.setString_("")
            self.name_field.setStringValue_("")
            self.window.orderOut_(None)
        self.result = self.session.result = None

import heapq
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class Scheduler:
    """Deterministic main-loop clock: no sleeps or AppKit run loop required."""
    def __init__(self):
        self.now = 0.0
        self.queue = []
        self.order = itertools.count()

    def call_later(self, delay, callback, *args):
        heapq.heappush(self.queue, (self.now + delay, next(self.order), callback, args))

    def step(self):
        self.now, _, callback, args = heapq.heappop(self.queue)
        callback(*args)

    def run_all(self):
        for _ in range(2000):
            if not self.queue:
                return
            self.step()
        raise AssertionError("Unbounded scheduled callbacks")


class FakePasteboardItem:
    def __init__(self, values=None):
        self.values = dict(values or {})

    @classmethod
    def alloc(cls):
        return cls()

    def init(self):
        return self

    def types(self):
        return list(self.values)

    def dataForType_(self, kind):
        return self.values.get(kind)

    def setData_forType_(self, data, kind):
        self.values[kind] = bytes(data)
        return True


class FakePasteboard:
    def __init__(self):
        self.items = []
        self.count = 0
        self.fail_write = False

    def changeCount(self):
        return self.count

    def pasteboardItems(self):
        return self.items

    def clearContents(self):
        self.count += 1
        self.items = []
        return self.count

    def setString_forType_(self, text, kind):
        if self.fail_write:
            return False
        if not self.items:
            self.items.append(FakePasteboardItem())
        self.items[0].values[kind] = text.encode()
        return True

    def stringForType_(self, kind):
        value = self.items[0].values.get(kind) if self.items else None
        return value.decode() if value is not None else None

    def types(self):
        return self.items[0].types() if self.items else []

    def writeObjects_(self, items):
        self.count += 1
        self.items = list(items)
        return True


class FakeStream:
    def __init__(self, **kwargs):
        self.closed = False
        self.stopped = True
        self.active = False

    def start(self):
        self.active = True
        self.stopped = False

    def stop(self):
        self.active = False
        self.stopped = True

    def close(self):
        self.stop()
        self.closed = True


class IsolatedTestCase(unittest.TestCase):
    """Never touch real settings, clipboard, sound, input events or microphone."""
    def setUp(self):
        super().setUp()
        import os
        if os.environ.get("GIGA_UI_TESTS") == "1":
            from AppKit import NSApplication
            NSApplication.sharedApplication()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.scheduler = Scheduler()
        self.pasteboard = FakePasteboard()
        replacements = {
            "giga_dictate.config.CONFIG_DIR": str(root / "config"),
            "giga_dictate.config.CONFIG_PATH": str(root / "config/config.json"),
            "giga_dictate.config.LAUNCH_AGENT_DIR": str(root / "agents"),
            "giga_dictate.config.LAUNCH_AGENT_PLIST": str(root / "agents/app.plist"),
            "giga_dictate.audio.sd.InputStream": FakeStream,
            "giga_dictate.sounds.NSSound": MagicMock(),
            "giga_dictate.ducking.AudioDucker.get_system_volume": MagicMock(return_value=0.5),
            "giga_dictate.ducking.AudioDucker.set_system_volume": MagicMock(),
            "giga_dictate.inserter.NSPasteboard": MagicMock(generalPasteboard=MagicMock(return_value=self.pasteboard)),
            "giga_dictate.inserter.NSPasteboardItem": FakePasteboardItem,
            "giga_dictate.inserter.AppHelper": MagicMock(callLater=self.scheduler.call_later),
            "giga_dictate.inserter.AXIsProcessTrusted": lambda: True,
            "giga_dictate.inserter.TextInserter._front_pid": lambda _: 1234,
            "giga_dictate.inserter.CGEventCreateKeyboardEvent": MagicMock(side_effect=lambda *a: object()),
            "giga_dictate.inserter.CGEventSetFlags": MagicMock(),
            "giga_dictate.inserter.CGEventSetIntegerValueField": MagicMock(),
            "giga_dictate.inserter.CGEventSourceFlagsState": lambda _: 0,
            "giga_dictate.inserter.CGEventPost": MagicMock(),
        }
        for target, replacement in replacements.items():
            context = patch(target, replacement)
            context.start()
            self.addCleanup(context.stop)

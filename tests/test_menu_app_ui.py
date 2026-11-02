import unittest
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
)
from giga_dictate.menu_app import MenuAppController, MODELS, DEVICES, HOTKEY_PRESETS
from giga_dictate.config import AppConfig


class TestMenuAppUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = NSApplication.sharedApplication()

    def test_pure_menubar_mode(self):
        controller = MenuAppController.alloc().init()
        self.assertIsNone(getattr(controller, "settings_window", None))

        # Build menu and verify pure menubar items
        controller.rebuild_menu()
        menu = controller.menu
        self.assertIsNotNone(menu)

        titles = [item.title() for item in menu.itemArray() if item.title()]

        # Verify NO settings window / control center item
        for title in titles:
            self.assertNotIn("Центр управления", title)
            self.assertNotIn("Настройки...", title)

        # Verify essential menubar controls are present
        self.assertTrue(any("Готов" in t or "Загрузка" in t or "Запись" in t for t in titles))
        self.assertTrue(any("запись" in t.lower() for t in titles))
        self.assertTrue(any("Модель GigaAM" in t for t in titles))
        self.assertTrue(any("Устройство ускорения" in t for t in titles))
        self.assertTrue(any("Режим вызова клавиши" in t for t in titles))
        self.assertTrue(any("Горячая клавиша" in t for t in titles))
        self.assertTrue(any("Звуковые сигналы" in t for t in titles))
        self.assertTrue(any("Добавлять пробел" in t for t in titles))
        self.assertTrue(any("Универсальный доступ" in t for t in titles))
        self.assertTrue(any("Завершить" in t for t in titles))

    def test_recording_menubar_clean_icon_only(self):
        controller = MenuAppController.alloc().init()
        # Mock status item button
        from AppKit import NSButton, NSStatusBar, NSVariableStatusItemLength, NSImage
        sb = NSStatusBar.systemStatusBar()
        controller.status_item = sb.statusItemWithLength_(NSVariableStatusItemLength)
        controller.menu_icon = NSImage.alloc().init()
        controller.menu_icon_rec = NSImage.alloc().init()

        # 1. Recording state: only red icon, no "Запись", no "●"
        controller.update_status_title("Запись...", recording=True)
        btn = controller.status_item.button()
        self.assertEqual(btn.title(), "")
        self.assertNotIn("Запись", btn.title())
        self.assertNotIn("●", btn.title())
        self.assertEqual(btn.image(), controller.menu_icon_rec)

        # 2. Ready state: clean icon only
        controller.engine._is_loaded = True
        controller.update_status_title("", recording=False)
        self.assertEqual(btn.title(), "")
        self.assertEqual(btn.image(), controller.menu_icon)


if __name__ == "__main__":
    unittest.main()

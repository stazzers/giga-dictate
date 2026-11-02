import unittest

from AppKit import NSApplication

from giga_dictate.menu_app import MenuAppController


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
        self.assertFalse(any("Устройство ускорения" in t for t in titles))
        self.assertTrue(any("Режим вызова клавиши" in t for t in titles))
        self.assertTrue(any("Горячая клавиша" in t for t in titles))
        self.assertTrue(any("Звуковые сигналы" in t for t in titles))
        self.assertTrue(any("Добавлять пробел" in t for t in titles))
        self.assertTrue(any("Запуск при старте" in t for t in titles))
        self.assertTrue(any("Универсальный доступ" in t for t in titles))
        self.assertTrue(any("полное удаление" in t.lower() for t in titles))
        self.assertTrue(any("Завершить" in t for t in titles))
        self.assertTrue(hasattr(controller, "onUninstallAppDialog_"))
        self.assertTrue(hasattr(controller, "execute_complete_uninstallation"))
        self.assertTrue(hasattr(controller, "onToggleLaunchAtLogin_"))

    def test_hotkey_three_line_submenu(self):
        controller = MenuAppController.alloc().init()
        # 1. Default state: only 2 items (default + change key), NO middle custom key row
        controller.config.hotkey = "right_cmd"
        controller.config.custom_hotkey = None
        controller.rebuild_menu()

        menu = controller.menu
        hotkey_item = next(item for item in menu.itemArray() if "Горячая клавиша" in (item.title() or ""))
        submenu = hotkey_item.submenu()
        self.assertIsNotNone(submenu)

        sub_titles = [item.title() for item in submenu.itemArray() if item.title()]
        self.assertEqual(len(sub_titles), 2)
        self.assertTrue("По умолчанию" in sub_titles[0])
        self.assertTrue("Правый Command" in sub_titles[0])
        self.assertTrue("Изменить клавишу" in sub_titles[1])
        # Default key is checked
        items = [item for item in submenu.itemArray() if item.title()]
        self.assertEqual(items[0].state(), 1)

        # 2. Custom hotkey selected (differs from default): middle row appears
        controller.config.hotkey = "f8"
        controller.config.custom_hotkey = "f8"
        controller.rebuild_menu()

        items_custom = next(item for item in controller.menu.itemArray() if "Горячая клавиша" in (item.title() or "")).submenu().itemArray()
        sub_custom = [item for item in items_custom if item.title()]
        self.assertEqual(len(sub_custom), 3)
        self.assertTrue("По умолчанию" in sub_custom[0].title())
        self.assertEqual(sub_custom[0].state(), 0)  # unselected but clickable
        self.assertTrue("Выбранная клавиша" in sub_custom[1].title())
        self.assertTrue("F8" in sub_custom[1].title())
        self.assertEqual(sub_custom[1].state(), 1)  # selected
        self.assertTrue("Изменить клавишу" in sub_custom[2].title())

        # 3. Switching back to default: middle row disappears immediately!
        controller.onSelectDefaultHotkey_(None)
        self.assertEqual(controller.config.hotkey, "right_cmd")
        items_def = next(item for item in controller.menu.itemArray() if "Горячая клавиша" in (item.title() or "")).submenu().itemArray()
        sub_def = [item for item in items_def if item.title()]
        self.assertEqual(len(sub_def), 2)  # middle row is gone!
        self.assertTrue("По умолчанию" in sub_def[0].title())
        self.assertEqual(sub_def[0].state(), 1)
        self.assertTrue("Изменить клавишу" in sub_def[1].title())

    def test_recording_menubar_clean_icon_only(self):
        controller = MenuAppController.alloc().init()
        # Mock status item button
        from AppKit import NSImage, NSStatusBar, NSVariableStatusItemLength
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

    def test_launch_at_login_toggle(self):
        controller = MenuAppController.alloc().init()
        from giga_dictate.config import is_launch_at_login_enabled, set_launch_at_login

        try:
            # Ensure starts clean
            set_launch_at_login(False)
            self.assertFalse(is_launch_at_login_enabled())

            # Toggle on
            controller.onToggleLaunchAtLogin_(None)
            self.assertTrue(is_launch_at_login_enabled())
            self.assertTrue(controller.config.launch_at_login)

            # Toggle off
            controller.onToggleLaunchAtLogin_(None)
            self.assertFalse(is_launch_at_login_enabled())
            self.assertFalse(controller.config.launch_at_login)
        finally:
            set_launch_at_login(False)


    def test_load_menu_icon_retina_multi_representation(self):
        import os
        from giga_dictate.menu_app import _load_menu_icon

        res_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "giga_dictate",
            "resources",
        )
        icon_path = os.path.join(res_dir, "menu_icon.png")
        rec_path = os.path.join(res_dir, "menu_icon_rec.png")

        # 1. Non-existent path returns None
        self.assertIsNone(_load_menu_icon("/non/existent/path.png"))

        # 2. Template icon loads multi-rep Retina representations
        img = _load_menu_icon(icon_path, is_template=True)
        self.assertIsNotNone(img)
        self.assertTrue(img.isTemplate())
        self.assertEqual(img.size().width, 11.0)
        self.assertEqual(img.size().height, 18.0)
        self.assertGreaterEqual(len(img.representations()), 2)

        # 3. Recording icon loads with full color and non-template
        rec_img = _load_menu_icon(rec_path, is_template=False)
        self.assertIsNotNone(rec_img)
        self.assertFalse(rec_img.isTemplate())
        self.assertEqual(rec_img.size().width, 11.0)
        self.assertEqual(rec_img.size().height, 18.0)
        self.assertGreaterEqual(len(rec_img.representations()), 2)


if __name__ == "__main__":
    unittest.main()

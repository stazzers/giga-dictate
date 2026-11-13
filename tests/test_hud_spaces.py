"""Native HUD lifecycle checks; they do not actually switch user desktops."""
import os
import unittest
from unittest.mock import MagicMock, patch

from AppKit import (NSStatusWindowLevel, NSWindowCollectionBehaviorCanJoinAllSpaces,
                    NSWindowCollectionBehaviorFullScreenAuxiliary,
                    NSWindowStyleMaskNonactivatingPanel)
from giga_dictate.hud import VoiceInputHUD
from tests.support import IsolatedTestCase


@unittest.skipUnless(os.environ.get("GIGA_UI_TESTS") == "1", "Native UI tests require GIGA_UI_TESTS=1")
class TestHUDSpaces(IsolatedTestCase):
    def hud(self):
        hud = VoiceInputHUD.alloc().init()
        self.addCleanup(hud.close)
        # No Accessibility queries against the user's currently focused app.
        hud._request_caret = MagicMock()
        return hud

    def test_panel_joins_spaces_and_fullscreen_without_becoming_key(self):
        hud = self.hud()
        flags = hud.panel.collectionBehavior()
        self.assertTrue(flags & NSWindowCollectionBehaviorCanJoinAllSpaces)
        self.assertTrue(flags & NSWindowCollectionBehaviorFullScreenAuxiliary)
        self.assertTrue(hud.panel.styleMask() & NSWindowStyleMaskNonactivatingPanel)
        self.assertFalse(hud.panel.canBecomeKeyWindow())
        self.assertFalse(hud.panel.canBecomeMainWindow())
        self.assertEqual(hud.panel.level(), NSStatusWindowLevel)

    def test_space_change_repositions_and_represents_active_panel(self):
        hud = self.hud()
        hud.showRecording()
        generation = hud._generation
        hud.panel.orderOut_(None)
        hud.panel.setAlphaValue_(0.0)
        hud.onWorkspaceChange_(None)
        self.assertEqual(hud._generation, generation + 1)
        self.assertTrue(hud.panel.isVisible())
        self.assertEqual(hud.panel.alphaValue(), 1.0)
        self.assertEqual(hud.state, "recording")
        self.assertFalse(hud.panel.isKeyWindow())

    def test_inactive_panel_is_not_shown_on_space_change(self):
        hud = self.hud()
        generation = hud._generation
        hud.onWorkspaceChange_(None)
        self.assertEqual(hud._generation, generation)
        self.assertFalse(hud.panel.isVisible())

    def test_old_fade_cannot_hide_new_recording(self):
        hud = self.hud()
        hud.showRecording()
        hud.hideHUD()
        old_timer = hud._fade_timer
        hud.showRecording()
        hud._onFadeDone_(old_timer)
        self.assertTrue(hud._is_visible)
        self.assertEqual(hud.state, "recording")
        self.assertIsNotNone(hud.timer)

    def test_tail_text_and_geometry_remain_bounded_with_cached_layout(self):
        hud = self.hud()
        hud.showRecording()
        text = "старый текст " * 1000 + "ПОСЛЕДНИЕ СЛОВА"
        for _ in range(2):
            hud.setPreviewText_(text)
            self.assertLessEqual(hud.panel.frame().size.height, 160)
            self.assertLessEqual(hud.panel.frame().size.width, 540)
            self.assertLess(len(hud.text_label.stringValue()), 602)
            self.assertTrue(hud.text_label.stringValue().endswith("ПОСЛЕДНИЕ СЛОВА"))
        hud.hideHUD()
        self.assertEqual(hud._current_text, "")
        self.assertFalse(hud._measure_cache)
        self.assertIsNone(hud.timer)

    def test_stale_caret_does_not_move_panel(self):
        from AppKit import NSPoint
        hud = self.hud()
        hud.showRecording()
        x = hud._anchor_center_x
        with patch("giga_dictate.hud.AppHelper.callLater"):
            hud._deliver_caret(hud._generation - 1, None, NSPoint(99999, 99999), 290, 46)
        self.assertEqual(hud._anchor_center_x, x)


@unittest.skipUnless(os.environ.get("GIGA_UI_TESTS") == "1", "Native UI tests require GIGA_UI_TESTS=1")
class TestPrivateNativePasteboard(IsolatedTestCase):
    def test_round_trip_real_pasteboard_preserves_binary_items(self):
        from AppKit import NSPasteboard, NSPasteboardItem, NSPasteboardTypeString
        from Foundation import NSData
        from giga_dictate.inserter import TextInserter
        pb = NSPasteboard.pasteboardWithUniqueName()
        self.addCleanup(pb.releaseGlobally)
        first, second = NSPasteboardItem.alloc().init(), NSPasteboardItem.alloc().init()
        payload = b"\x89PNG\x00\x01"
        self.assertTrue(first.setData_forType_(NSData.dataWithBytes_length_(payload, len(payload)), "public.png"))
        self.assertTrue(first.setString_forType_("caption", NSPasteboardTypeString))
        self.assertTrue(second.setString_forType_("second", NSPasteboardTypeString))
        self.assertTrue(pb.writeObjects_([first, second]))
        with patch("giga_dictate.inserter.NSPasteboardItem", NSPasteboardItem):
            inserter = TextInserter(pasteboard=pb, clock=lambda: self.scheduler.now)
            future = inserter.insert_text("speech", restore_clipboard=True)
            self.scheduler.run_all()
        self.assertTrue(future.result().event_posted)
        items = pb.pasteboardItems()
        self.assertEqual(len(items), 2)
        self.assertEqual(bytes(items[0].dataForType_("public.png")), payload)
        self.assertEqual(items[0].stringForType_(NSPasteboardTypeString), "caption")
        self.assertEqual(items[1].stringForType_(NSPasteboardTypeString), "second")

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from giga_dictate.config import AppConfig
from giga_dictate.ducking import AudioDucker
from giga_dictate.menu_app import MenuAppController


class TestAudioDucking(unittest.TestCase):
    def test_app_config_audio_ducking_defaults(self):
        """AppConfig must default audio_ducking to True and ducking_volume to 0.12."""
        config = AppConfig()
        self.assertTrue(config.audio_ducking)
        self.assertAlmostEqual(config.ducking_volume, 0.12, places=2)

    def test_app_config_audio_ducking_serialization(self):
        """AppConfig must preserve audio_ducking settings across serialization."""
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            with patch("giga_dictate.config.CONFIG_PATH", temp_path):
                config = AppConfig(audio_ducking=False, ducking_volume=0.08)
                config.save()

                loaded = AppConfig.load()
                self.assertFalse(loaded.audio_ducking)
                self.assertAlmostEqual(loaded.ducking_volume, 0.08, places=2)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_ducker_duck_and_unduck(self):
        """AudioDucker lowers volume on duck() and restores original volume on unduck()."""
        ducker = AudioDucker()

        with patch.object(ducker, "get_system_volume", return_value=0.75), \
             patch.object(ducker, "set_system_volume") as mock_set:

            ducker.duck(target_volume=0.12)
            self.assertTrue(ducker.is_ducked)
            self.assertEqual(ducker._orig_vol, 0.75)
            mock_set.assert_called_once_with(0.12)

            # Idempotent second duck call does not change recorded original volume
            mock_set.reset_mock()
            ducker.duck(target_volume=0.05)
            mock_set.assert_not_called()
            self.assertEqual(ducker._orig_vol, 0.75)

            # Unduck restores original 0.75
            ducker.unduck()
            self.assertFalse(ducker.is_ducked)
            self.assertIsNone(ducker._orig_vol)
            mock_set.assert_called_once_with(0.75)

            # Idempotent second unduck call is safe no-op
            mock_set.reset_mock()
            ducker.unduck()
            mock_set.assert_not_called()

    def test_ducker_already_below_target_volume(self):
        """AudioDucker does not alter volume if user already has it lower than ducking target."""
        ducker = AudioDucker()

        with patch.object(ducker, "get_system_volume", return_value=0.05), \
             patch.object(ducker, "set_system_volume") as mock_set:

            ducker.duck(target_volume=0.12)
            self.assertTrue(ducker.is_ducked)
            self.assertEqual(ducker._orig_vol, 0.05)
            # Should NOT increase volume to 0.12!
            mock_set.assert_not_called()

            ducker.unduck()
            self.assertFalse(ducker.is_ducked)
            mock_set.assert_called_once_with(0.05)

    def test_menu_app_ducking_enabled_triggers_duck_and_unduck(self):
        """MenuAppController ducks on start_recording and unducks on stop_and_transcribe."""
        controller = MenuAppController.alloc().init()
        controller.config.audio_ducking = True
        controller.config.ducking_volume = 0.12

        mock_ducker = MagicMock()
        controller.ducker = mock_ducker

        # Mock recorder start/stop to avoid hardware calls
        controller.recorder.start = MagicMock(return_value=True)
        controller.recorder.stop = MagicMock(return_value=(b"", 1.0))
        controller.recorder._recording = True

        controller.start_recording()
        mock_ducker.duck.assert_called_once_with(target_volume=0.12)

        controller.stop_and_transcribe()
        mock_ducker.unduck.assert_called_once()

    def test_menu_app_ducking_disabled_does_not_duck(self):
        """MenuAppController does not duck when audio_ducking config is False."""
        controller = MenuAppController.alloc().init()
        controller.config.audio_ducking = False

        mock_ducker = MagicMock()
        controller.ducker = mock_ducker

        controller.recorder.start = MagicMock(return_value=True)
        controller.start_recording()
        mock_ducker.duck.assert_not_called()

    def test_menu_app_toggle_audio_ducking(self):
        """Toggling audio ducking in menu flips config and unducks if currently ducked."""
        controller = MenuAppController.alloc().init()
        controller.config.audio_ducking = True

        mock_ducker = MagicMock()
        mock_ducker.is_ducked = True
        controller.ducker = mock_ducker
        controller.config.save = MagicMock()
        controller.rebuild_menu = MagicMock()

        # Toggle OFF
        controller.onToggleAudioDucking_(None)
        self.assertFalse(controller.config.audio_ducking)
        mock_ducker.unduck.assert_called_once()
        controller.config.save.assert_called_once()
        controller.rebuild_menu.assert_called_once()

        # Toggle ON
        mock_ducker.unduck.reset_mock()
        controller.onToggleAudioDucking_(None)
        self.assertTrue(controller.config.audio_ducking)
        mock_ducker.unduck.assert_not_called()


if __name__ == "__main__":
    unittest.main()

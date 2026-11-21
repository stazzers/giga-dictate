"""The retired profile is deleted narrowly; other local user data is preserved."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from giga_dictate.config import remove_legacy_voice_profile


class RetiredVoiceProfileCleanup(unittest.TestCase):
    def test_only_retired_profile_is_deleted_without_reading_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "owner_voice_profile.json"
            profile.write_bytes(b"legacy private data")
            settings = root / "config.json"
            settings.write_bytes(b"settings to preserve")
            with patch("giga_dictate.config.CONFIG_DIR", temporary), \
                 patch("builtins.open", side_effect=AssertionError("Profile must not be loaded")):
                remove_legacy_voice_profile()
                remove_legacy_voice_profile()
            self.assertFalse(profile.exists())
            self.assertEqual(settings.read_bytes(), b"settings to preserve")

    def test_absent_profile_is_an_idempotent_noop(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch("giga_dictate.config.CONFIG_DIR", temporary):
            remove_legacy_voice_profile()
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_failed_deletion_is_reported_without_content(self):
        with patch("giga_dictate.config.os.remove", side_effect=PermissionError("private data")), \
             self.assertLogs("giga_dictate.config", level="WARNING") as logs:
            remove_legacy_voice_profile()
        self.assertIn("retired_voice_profile_cleanup_failed", logs.output[0])
        self.assertNotIn("private data", logs.output[0])

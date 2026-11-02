import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field

CONFIG_DIR = os.path.expanduser("~/.config/giga_dictate")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
LAUNCH_AGENT_DIR = os.path.expanduser("~/Library/LaunchAgents")
LAUNCH_AGENT_PLIST = os.path.join(LAUNCH_AGENT_DIR, "com.salute.gigadictate.plist")


def is_launch_at_login_enabled() -> bool:
    """Checks whether the launch agent plist exists in user's Library."""
    return os.path.exists(LAUNCH_AGENT_PLIST)


def set_launch_at_login(enabled: bool) -> bool:
    """Enables or disables auto-launch at macOS login via user LaunchAgent."""
    if not enabled:
        try:
            if os.path.exists(LAUNCH_AGENT_PLIST):
                os.remove(LAUNCH_AGENT_PLIST)
            return True
        except OSError:
            return False

    app_path = "/Applications/GigaDictate.app"
    try:
        from Foundation import NSBundle

        bundle = NSBundle.mainBundle()
        if bundle:
            b_path = bundle.bundlePath()
            if b_path and b_path.endswith(".app") and os.path.exists(b_path):
                app_path = b_path
    except (ImportError, AttributeError):
        pass

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.salute.gigadictate</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/open</string>
        <string>{app_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>ProcessType</key>
    <string>Interactive</string>
</dict>
</plist>
"""
    try:
        os.makedirs(LAUNCH_AGENT_DIR, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(LAUNCH_AGENT_PLIST, flags, 0o644)
        with open(fd, "w", encoding="utf-8") as f:
            f.write(plist_content)
        return True
    except OSError:
        return False


def get_default_device() -> str:
    # On Apple Silicon macOS, MPS is natively supported
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"
    if "torch" in sys.modules:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


@dataclass
class AppConfig:
    hotkey: str = "right_cmd"
    hotkey_mode: str = "toggle"  # "toggle" or "push_to_talk"
    model_name: str = "v3_e2e_ctc"
    device: str = field(default_factory=get_default_device)
    sound_feedback: bool = True
    restore_clipboard: bool = (
        True  # Immediately erase dictated text from pasteboard after Cmd+V
    )
    conceal_clipboard: bool = (
        True  # Protect pasteboard from clipboard history managers and iCloud sync
    )
    add_trailing_space: bool = True
    paste_delay: float = 0.03
    min_duration: float = 0.25
    audio_device: int | None = None
    show_settings_on_launch: bool = False
    custom_hotkey: str | None = None
    launch_at_login: bool = False
    live_hud_preview: bool = True  # Real-time streaming transcription preview in floating HUD
    hud_glass_style: str = "glass"  # "glass" (vibrant translucent), "balanced" (frosted), "matte" (high-contrast opaque)
    audio_ducking: bool = True  # Automatically lower system volume during recording to prevent speaker bleed
    ducking_volume: float = 0.12  # Target volume level during ducking (0.0 to 1.0)

    @property
    def hud_glass_alpha(self) -> float:
        alphas = {
            "glass": 0.25,
            "balanced": 0.55,
            "matte": 0.85,
        }
        return alphas.get(self.hud_glass_style, 0.25)

    @classmethod
    def load(cls) -> "AppConfig":
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                config = cls(
                    **{k: v for k, v in data.items() if k in cls.__annotations__}
                )
                if config.custom_hotkey is None and config.hotkey != "right_cmd":
                    config.custom_hotkey = config.hotkey
                config.launch_at_login = is_launch_at_login_enabled()
                return config
            except Exception as e:
                print(
                    f"[Config] Error loading config from {CONFIG_PATH}: {e}, using defaults."
                )
        config = cls()
        config.save()
        return config

    def save(self) -> None:
        try:
            os.makedirs(CONFIG_DIR, mode=0o700, exist_ok=True)
            try:
                os.chmod(CONFIG_DIR, 0o700)
            except OSError:
                pass

            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            fd = os.open(CONFIG_PATH, flags, 0o600)
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
            try:
                os.chmod(CONFIG_PATH, 0o600)
            except OSError:
                pass
        except Exception as e:
            print(f"[Config] Error saving config: {e}")

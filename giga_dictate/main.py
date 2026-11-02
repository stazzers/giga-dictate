import logging
from logging.handlers import RotatingFileHandler
import os
import signal
import sys

# Ensure Homebrew path is in environment so ffmpeg and core utilities are found
brew_bin = "/opt/homebrew/bin"
if brew_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{brew_bin}:{os.environ.get('PATH', '')}"

from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

from giga_dictate.menu_app import MenuAppController

config_dir = os.path.expanduser("~/.config/giga_dictate")
try:
    os.makedirs(config_dir, mode=0o700, exist_ok=True)
    os.chmod(config_dir, 0o700)
    # Purge history from disk
    hp = os.path.join(config_dir, "history.json")
    if os.path.exists(hp):
        os.remove(hp)
except OSError:
    pass

log_path = os.path.join(config_dir, "app.log")
handlers = [
    logging.StreamHandler(sys.stdout),
    RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024, backupCount=1, encoding="utf-8"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=handlers,
)
logger = logging.getLogger("giga_dictate")


_main_controller = None


def main():
    global _main_controller
    logger.info("Starting GigaDictate...")
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    _main_controller = MenuAppController.alloc().init()
    app.setDelegate_(_main_controller)

    # Handle Ctrl+C in terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Exiting GigaDictate...")
        if _main_controller:
            _main_controller.onQuitApp_(None)


if __name__ == "__main__":
    main()

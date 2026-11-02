import logging
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
    # Purge any legacy log files and history from disk
    for f in ("app.log", "history.json"):
        fp = os.path.join(config_dir, f)
        if os.path.exists(fp):
            os.remove(fp)
except OSError:
    pass

# Zero-Disk Logging: Stream directly to stdout/stderr (zero log files on disk)
handlers = [
    logging.StreamHandler(sys.stdout),
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

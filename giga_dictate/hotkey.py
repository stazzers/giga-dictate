import logging
import time
from typing import Callable, Optional
from Quartz import (
    CGEventTapCreate,
    CGEventTapEnable,
    kCGSessionEventTap,
    kCGHeadInsertEventTap,
    kCGEventTapOptionDefault,
    kCGEventKeyDown,
    kCGEventKeyUp,
    kCGEventFlagsChanged,
    kCGEventTapDisabledByTimeout,
    kCGEventTapDisabledByUserInput,
    CGEventGetIntegerValueField,
    kCGKeyboardEventKeycode,
    CGEventGetFlags,
    kCGEventFlagMaskCommand,
    kCGEventFlagMaskShift,
    kCGEventFlagMaskAlternate,
    kCGEventFlagMaskControl,
)
from CoreFoundation import (
    CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource,
    CFRunLoopRemoveSource,
    CFRunLoopGetMain,
    kCFRunLoopCommonModes,
    kCFAllocatorDefault,
)
from PyObjCTools import AppHelper

logger = logging.getLogger("giga_dictate.hotkey")

# Virtual keycodes on macOS (Hardware ANSI)
VK_RIGHT_COMMAND = 54
VK_LEFT_COMMAND = 55
VK_RIGHT_SHIFT = 60
VK_LEFT_SHIFT = 56
VK_RIGHT_OPTION = 61
VK_LEFT_OPTION = 58
VK_RIGHT_CONTROL = 62
VK_LEFT_CONTROL = 59

MODIFIER_MASKS = {
    VK_RIGHT_COMMAND: kCGEventFlagMaskCommand,
    VK_LEFT_COMMAND: kCGEventFlagMaskCommand,
    VK_RIGHT_SHIFT: kCGEventFlagMaskShift,
    VK_LEFT_SHIFT: kCGEventFlagMaskShift,
    VK_RIGHT_OPTION: kCGEventFlagMaskAlternate,
    VK_LEFT_OPTION: kCGEventFlagMaskAlternate,
    VK_RIGHT_CONTROL: kCGEventFlagMaskControl,
    VK_LEFT_CONTROL: kCGEventFlagMaskControl,
}

MACOS_CHAR_TO_VK = {
    "/": 44,
    "\\": 42,
    ".": 47,
    ",": 43,
    ";": 41,
    "'": 39,
    "[": 33,
    "]": 30,
    "-": 27,
    "=": 24,
    "`": 50,
    "space": 49,
    # Russian layout keys
    "ю": 47,
    "б": 43,
    "ж": 41,
    "э": 39,
    "х": 33,
    "ъ": 30,
    "ё": 50,
    # Numbers
    "1": 18, "2": 19, "3": 20, "4": 21, "5": 23, "6": 22,
    "7": 26, "8": 28, "9": 25, "0": 29,
    # Letters
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7,
    "c": 8, "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15,
    "y": 16, "t": 17, "o": 31, "u": 32, "i": 34, "p": 35, "l": 37,
    "j": 38, "k": 40, "n": 45, "m": 46,
}

KEY_NAME_TO_VK = {
    # Command (SuperDictate default)
    "right_cmd": VK_RIGHT_COMMAND,
    "cmd_r": VK_RIGHT_COMMAND,
    "command_r": VK_RIGHT_COMMAND,
    "right_command": VK_RIGHT_COMMAND,
    "правый cmd": VK_RIGHT_COMMAND,
    "правый command": VK_RIGHT_COMMAND,
    "left_cmd": VK_LEFT_COMMAND,
    "cmd_l": VK_LEFT_COMMAND,
    "command_l": VK_LEFT_COMMAND,
    "left_command": VK_LEFT_COMMAND,
    "левый cmd": VK_LEFT_COMMAND,
    "cmd": VK_RIGHT_COMMAND,
    "command": VK_RIGHT_COMMAND,
    # Shift
    "right_shift": VK_RIGHT_SHIFT,
    "shift_r": VK_RIGHT_SHIFT,
    "правый shift": VK_RIGHT_SHIFT,
    "правый_shift": VK_RIGHT_SHIFT,
    "left_shift": VK_LEFT_SHIFT,
    "shift_l": VK_LEFT_SHIFT,
    "левый shift": VK_LEFT_SHIFT,
    "shift": VK_RIGHT_SHIFT,
    # Option / Alt
    "right_option": VK_RIGHT_OPTION,
    "alt_r": VK_RIGHT_OPTION,
    "option_r": VK_RIGHT_OPTION,
    "правый option": VK_RIGHT_OPTION,
    "правый alt": VK_RIGHT_OPTION,
    "left_option": VK_LEFT_OPTION,
    "alt_l": VK_LEFT_OPTION,
    "левый option": VK_LEFT_OPTION,
    "option": VK_RIGHT_OPTION,
    "alt": VK_RIGHT_OPTION,
    # Control
    "right_ctrl": VK_RIGHT_CONTROL,
    "ctrl_r": VK_RIGHT_CONTROL,
    "control_r": VK_RIGHT_CONTROL,
    "правый ctrl": VK_RIGHT_CONTROL,
    "left_ctrl": VK_LEFT_CONTROL,
    "ctrl_l": VK_LEFT_CONTROL,
    "левый ctrl": VK_LEFT_CONTROL,
    "ctrl": VK_RIGHT_CONTROL,
    # Keys
    "slash": 44,
    "/": 44,
    "слэш": 44,
    ".": 47,
    "space": 49,
    "пробел": 49,
    "tab": 48,
    "таб": 48,
    "enter": 36,
    "return": 36,
    "ввод": 36,
    "esc": 53,
    "escape": 53,
    # Function keys
    "f1": 122,
    "f2": 120,
    "f3": 99,
    "f4": 118,
    "f5": 96,
    "f6": 97,
    "f7": 98,
    "f8": 100,
    "f9": 101,
    "f10": 109,
    "f11": 103,
    "f12": 111,
}


def parse_hotkey(hotkey_str: str) -> int:
    clean = hotkey_str.strip().lower()
    if clean in KEY_NAME_TO_VK:
        return KEY_NAME_TO_VK[clean]
    if clean in MACOS_CHAR_TO_VK:
        return MACOS_CHAR_TO_VK[clean]
    raise ValueError(f"Unknown hotkey: '{hotkey_str}'")


class HotkeyListener:
    """
    100% Native macOS Quartz Event Tap listener attached directly to CFRunLoopGetMain
    in kCFRunLoopCommonModes (identical architecture to SuperDictate / Parakey).
    """

    def __init__(
        self,
        hotkey_str: str = "right_cmd",
        mode: str = "toggle",
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
    ):
        self.hotkey_str = hotkey_str
        self.mode = mode
        self.on_start = on_start
        self.on_stop = on_stop

        self.target_vk = VK_RIGHT_COMMAND
        self._update_target_vk(hotkey_str)

        self._tap = None
        self._run_loop_source = None
        self._is_active = False       # Currently recording
        self._is_key_down = False     # Physical key is held down
        self._last_toggle_time = 0.0

    def _update_target_vk(self, hotkey_str: str) -> None:
        try:
            self.target_vk = parse_hotkey(hotkey_str)
            self.hotkey_str = hotkey_str
            logger.info(f"Target hotkey set: '{hotkey_str}' -> virtual keycode {self.target_vk}")
        except Exception as e:
            logger.error(f"Invalid hotkey '{hotkey_str}': {e}, defaulting to Right Command (54)")
            self.target_vk = VK_RIGHT_COMMAND
            self.hotkey_str = "right_cmd"

    def update_hotkey(self, hotkey_str: str, mode: Optional[str] = None) -> bool:
        """
        Updates hotkey and/or mode instantly without stopping the tap.
        """
        if mode:
            self.mode = mode
        self._update_target_vk(hotkey_str)
        self._is_key_down = False

        if self._tap is None:
            self.start()

        return True

    def _tap_callback(self, proxy, event_type, event, refcon):
        # 1. Automatically re-enable if macOS WindowServer disables tap due to timeout
        if event_type in (kCGEventTapDisabledByTimeout, kCGEventTapDisabledByUserInput):
            if self._tap is not None:
                CGEventTapEnable(self._tap, True)
            return event

        vk = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
        flags = CGEventGetFlags(event)

        is_down = False
        is_up = False

        if event_type == kCGEventFlagsChanged:
            if vk in MODIFIER_MASKS:
                mask = MODIFIER_MASKS[vk]
                if flags & mask:
                    is_down = True
                else:
                    is_up = True
            else:
                return event
        elif event_type == kCGEventKeyDown:
            is_down = True
        elif event_type == kCGEventKeyUp:
            is_up = True
        else:
            return event

        # Check if the event matches our target virtual keycode
        if vk == self.target_vk:
            now = time.time()
            if self.mode == "toggle":
                if is_down:
                    if not self._is_key_down:
                        self._is_key_down = True
                        if now - self._last_toggle_time >= 0.20:
                            self._last_toggle_time = now
                            self._is_active = not self._is_active
                            if self._is_active:
                                logger.info(f"Native Tap Toggle: START recording (vk={vk})")
                                if self.on_start:
                                    AppHelper.callAfter(self.on_start)
                            else:
                                logger.info(f"Native Tap Toggle: STOP recording (vk={vk})")
                                if self.on_stop:
                                    AppHelper.callAfter(self.on_stop)
                elif is_up:
                    self._is_key_down = False

            elif self.mode == "push_to_talk":
                if is_down:
                    if not self._is_active:
                        self._is_active = True
                        logger.info(f"Native Tap PTT: START recording (vk={vk})")
                        if self.on_start:
                            AppHelper.callAfter(self.on_start)
                elif is_up:
                    if self._is_active:
                        self._is_active = False
                        logger.info(f"Native Tap PTT: STOP recording (vk={vk})")
                        if self.on_stop:
                            AppHelper.callAfter(self.on_stop)

        return event

    def start(self) -> bool:
        """Creates native event tap and attaches to CFRunLoopGetMain in common modes."""
        if self._tap is not None:
            return True

        self._is_active = False
        self._is_key_down = False

        mask = (1 << kCGEventKeyDown) | (1 << kCGEventKeyUp) | (1 << kCGEventFlagsChanged)
        try:
            self._tap = CGEventTapCreate(
                kCGSessionEventTap,
                kCGHeadInsertEventTap,
                kCGEventTapOptionDefault,
                mask,
                self._tap_callback,
                None,
            )
            if self._tap is None:
                logger.warning(
                    "CGEventTapCreate returned None! Accessibility or Input Monitoring permission needed."
                )
                return False

            self._run_loop_source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, self._tap, 0)
            CFRunLoopAddSource(CFRunLoopGetMain(), self._run_loop_source, kCFRunLoopCommonModes)
            CGEventTapEnable(self._tap, True)
            logger.info(
                f"Native Quartz event tap successfully registered for vk={self.target_vk} on CFRunLoopGetMain."
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create native event tap: {e}", exc_info=True)
            self._tap = None
            self._run_loop_source = None
            return False

    def stop(self) -> None:
        """Stops and removes the native event tap from runloop."""
        if self._tap is not None:
            try:
                CGEventTapEnable(self._tap, False)
                if self._run_loop_source is not None:
                    CFRunLoopRemoveSource(CFRunLoopGetMain(), self._run_loop_source, kCFRunLoopCommonModes)
            except Exception:
                pass
            self._tap = None
            self._run_loop_source = None

        self._is_active = False
        self._is_key_down = False

    def restart(self) -> None:
        """Restarts the native event tap."""
        self.stop()
        time.sleep(0.05)
        self.start()

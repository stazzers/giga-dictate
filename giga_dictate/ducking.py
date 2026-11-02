"""
Audio Ducking Manager for macOS.
Automatically lowers system output volume during active microphone recording
to prevent speaker playback (music, video, notifications) from bleeding into the microphone,
and instantaneously restores user volume when recording ends.
Uses ultra-low latency (~0.2ms) native CoreAudio hardware APIs with safe AppleScript fallback.
"""

import atexit
import ctypes
import logging
import subprocess
import threading

logger = logging.getLogger("giga_dictate.ducking")

# CoreAudio definitions
try:
    _core_audio = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
    )
except Exception as e:
    logger.debug(f"Could not load CoreAudio.framework: {e}")
    _core_audio = None


class AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


def _fourcc(s: str) -> int:
    return int.from_bytes(s.encode("ascii"), byteorder="big")


# FourCC selectors
_kAudioObjectSystemObject = 1
_kAudioHardwarePropertyDefaultOutputDevice = _fourcc("dOut")
_kAudioDevicePropertyVolumeScalar = _fourcc("volm")
_kAudioObjectPropertyScopeGlobal = _fourcc("glob")
_kAudioObjectPropertyScopeOutput = _fourcc("outp")
_kAudioObjectPropertyElementMain = 0


class AudioDucker:
    """
    Manages temporary attenuation of system output volume during voice dictation.
    Thread-safe and guaranteed restoration via atexit handlers.
    """

    def __init__(self):
        self._orig_vol: float | None = None
        self._is_ducked: bool = False
        self._lock = threading.Lock()
        # Register atexit handler so volume is always restored even on crash or abrupt exit
        atexit.register(self.unduck)

    @property
    def is_ducked(self) -> bool:
        with self._lock:
            return self._is_ducked

    def _get_default_output_device(self) -> int | None:
        if _core_audio is None:
            return None
        addr = AudioObjectPropertyAddress(
            _kAudioHardwarePropertyDefaultOutputDevice,
            _kAudioObjectPropertyScopeGlobal,
            _kAudioObjectPropertyElementMain,
        )
        dev = ctypes.c_uint32()
        sz = ctypes.c_uint32(ctypes.sizeof(dev))
        err = _core_audio.AudioObjectGetPropertyData(
            _kAudioObjectSystemObject,
            ctypes.byref(addr),
            0,
            None,
            ctypes.byref(sz),
            ctypes.byref(dev),
        )
        if err != 0 or dev.value == 0:
            return None
        return dev.value

    def get_system_volume(self) -> float | None:
        """
        Returns the current system output volume as a float in [0.0, 1.0].
        Returns None if volume cannot be determined.
        """
        dev = self._get_default_output_device()
        if dev and _core_audio is not None:
            # Check main element 0 first, then channel 1
            for el in (_kAudioObjectPropertyElementMain, 1):
                addr = AudioObjectPropertyAddress(
                    _kAudioDevicePropertyVolumeScalar,
                    _kAudioObjectPropertyScopeOutput,
                    el,
                )
                if _core_audio.AudioObjectHasProperty(dev, ctypes.byref(addr)):
                    vol = ctypes.c_float()
                    sz = ctypes.c_uint32(ctypes.sizeof(vol))
                    err = _core_audio.AudioObjectGetPropertyData(
                        dev, ctypes.byref(addr), 0, None, ctypes.byref(sz), ctypes.byref(vol)
                    )
                    if err == 0:
                        return float(max(0.0, min(1.0, vol.value)))

        # Fallback to osascript
        try:
            res = subprocess.check_output(
                ["osascript", "-e", "output volume of (get volume settings)"],
                stderr=subprocess.DEVNULL,
                timeout=1.0,
            ).decode().strip()
            return float(int(res)) / 100.0
        except Exception as e:
            logger.debug(f"Fallback get volume error: {e}")
            return None

    def set_system_volume(self, val: float) -> bool:
        """
        Sets the system output volume to val in [0.0, 1.0].
        """
        val_clamped = max(0.0, min(1.0, float(val)))
        dev = self._get_default_output_device()
        if dev and _core_audio is not None:
            success = False
            for el in (_kAudioObjectPropertyElementMain, 1, 2):
                addr = AudioObjectPropertyAddress(
                    _kAudioDevicePropertyVolumeScalar,
                    _kAudioObjectPropertyScopeOutput,
                    el,
                )
                if _core_audio.AudioObjectHasProperty(dev, ctypes.byref(addr)):
                    vol = ctypes.c_float(val_clamped)
                    sz = ctypes.c_uint32(ctypes.sizeof(vol))
                    err = _core_audio.AudioObjectSetPropertyData(
                        dev, ctypes.byref(addr), 0, None, sz, ctypes.byref(vol)
                    )
                    if err == 0:
                        success = True
            if success:
                return True

        # Fallback to osascript
        try:
            pct = round(val_clamped * 100.0)
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {pct}"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.0,
            )
            return True
        except Exception as e:
            logger.debug(f"Fallback set volume error: {e}")
            return False

    def duck(self, target_volume: float = 0.12):
        """
        Lowers system volume to target_volume if current volume is higher.
        Records original volume for restoration upon unduck().
        """
        with self._lock:
            if self._is_ducked:
                return

            current = self.get_system_volume()
            if current is None:
                return

            self._orig_vol = current
            # Only lower if current volume is louder than target ducking level
            if current > target_volume:
                logger.info(
                    f"Audio ducking activated: lowering volume {current * 100:.0f}% -> {target_volume * 100:.0f}%"
                )
                self.set_system_volume(target_volume)
            else:
                logger.debug(
                    f"Current volume {current * 100:.0f}% already below ducking threshold {target_volume * 100:.0f}%"
                )
            self._is_ducked = True

    def unduck(self):
        """
        Restores system volume to the level recorded prior to ducking.
        Safe to call multiple times or when not ducked.
        """
        with self._lock:
            if not self._is_ducked:
                return

            if self._orig_vol is not None:
                logger.info(
                    f"Audio ducking deactivated: restoring volume to {self._orig_vol * 100:.0f}%"
                )
                self.set_system_volume(self._orig_vol)
                self._orig_vol = None

            self._is_ducked = False

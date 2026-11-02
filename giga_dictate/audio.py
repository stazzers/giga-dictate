import logging
import threading
from typing import List, Optional, Tuple
import numpy as np
import sounddevice as sd

logger = logging.getLogger("giga_dictate.audio")

SAMPLE_RATE = 16000


def split_audio_into_chunks(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    max_sec: float = 20.0,
    min_split_sec: float = 12.0,
) -> List[np.ndarray]:
    """
    Splits long audio into chunks shorter than max_sec (GigaAM limit is 25s),
    finding the quietest point (silence) between min_split_sec and max_sec.
    """
    max_len = int(max_sec * sr)
    if len(audio) <= max_len:
        return [audio]

    chunks = []
    start = 0
    total = len(audio)
    min_split = int(min_split_sec * sr)

    while start < total:
        remaining = total - start
        if remaining <= max_len:
            chunks.append(audio[start:])
            break

        # Search window for split: from start + min_split to start + max_len
        search_start = start + min_split
        search_end = start + max_len
        search_audio = audio[search_start:search_end]

        # Calculate energy in 100ms frames
        frame_size = int(0.1 * sr)
        num_frames = len(search_audio) // frame_size
        if num_frames > 0:
            frames = search_audio[: num_frames * frame_size].reshape(num_frames, frame_size)
            energies = np.sum(frames**2, axis=1)
            min_idx = int(np.argmin(energies))
            split_offset = search_start + (min_idx * frame_size) + (frame_size // 2)
        else:
            split_offset = search_end

        chunks.append(audio[start:split_offset])
        start = split_offset

    return chunks


class AudioRecorder:
    """
    On-demand audio recorder capturing 16kHz float32 audio.
    Microphone is accessed ONLY while recording is active.
    When idle, no audio streams or devices are kept open.
    Thread-safe and deadlock-free using synchronized stream transitions and stream.abort().
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device = device
        self._stream: Optional[sd.InputStream] = None
        self._frames: List[np.ndarray] = []
        self._recording = False
        self._current_level = 0.0
        self._speech_speed = 1.0
        self._prev_rms = 0.0
        self._lock = threading.Lock()
        self._stream_mutex = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._recording

    def get_level(self) -> float:
        """Returns the current audio amplitude level in range [0.0, 1.0]."""
        return self._current_level if self._recording else 0.0

    def get_speech_speed(self) -> float:
        """Returns speech cadence/tempo multiplier in range [1.0, 3.0]."""
        return self._speech_speed if self._recording else 1.0

    def get_audio_state(self) -> Tuple[float, float]:
        """Returns (amplitude_level, speech_speed) in a single atomic call."""
        if not self._recording:
            return 0.0, 1.0
        return self._current_level, self._speech_speed

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio stream status: {status}")

        if not self._recording:
            return

        with self._lock:
            if self._recording:
                self._frames.append(indata.copy())

        try:
            rms = float(np.sqrt(np.mean(indata**2)))
            if rms < 0.002:
                target = 0.0
                instant_speed = 1.0
            else:
                target = min(1.0, max(0.0, (rms - 0.002) * 35.0))
                # Syllable delta and articulation / timbre proxy
                delta_e = abs(rms - self._prev_rms)
                self._prev_rms = rms
                zcr = float(np.mean(indata[:-1] * indata[1:] < 0)) if len(indata) > 1 else 0.0

                # Dynamic speed multiplier (1.0x for slow, up to 3.0x for fast dictation)
                instant_speed = 1.0 + min(2.2, delta_e * 55.0 + zcr * 10.0)

            # Responsive volume attack (snappier, less sluggish, instant syllable rise)
            if target > self._current_level:
                self._current_level = self._current_level * 0.10 + target * 0.90
            else:
                self._current_level = self._current_level * 0.50 + target * 0.50

            # Smooth speed transition (faster cadence tracking)
            self._speech_speed = self._speech_speed * 0.60 + instant_speed * 0.40
        except Exception:
            pass

    def _cleanup_unlocked(self):
        """Stops and closes the stream immediately without buffer-draining deadlocks."""
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                try:
                    self._stream.close()
                except Exception:
                    pass
            finally:
                self._stream = None

    def start(self) -> bool:
        """
        Activates the microphone and starts capturing audio frames.
        Guaranteed non-blocking with mutex timeout.
        """
        if not self._stream_mutex.acquire(timeout=1.5):
            logger.warning("Could not acquire stream mutex to start recording.")
            return False

        try:
            self._cleanup_unlocked()

            with self._lock:
                self._frames = []
                self._recording = True
                self._current_level = 0.0

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
                callback=self._callback,
                blocksize=512,  # ~32ms per buffer
            )
            self._stream.start()
            logger.info("Microphone activated: recording started.")
            return True
        except Exception as e:
            logger.error(f"Failed to open microphone: {e}")
            with self._lock:
                self._recording = False
            self._cleanup_unlocked()
            return False
        finally:
            self._stream_mutex.release()

    def stop(self) -> Tuple[np.ndarray, float]:
        """
        Stops recording and immediately schedules microphone release.
        Returns audio frames instantly without waiting for hardware teardown.
        """
        with self._lock:
            if not self._recording:
                return np.array([], dtype=np.float32), 0.0
            self._recording = False
            self._current_level = 0.0
            frames = self._frames
            self._frames = []

        audio = np.concatenate(frames, axis=0).flatten() if frames else np.array([], dtype=np.float32)
        duration = len(audio) / self.sample_rate

        def _async_close():
            if self._stream_mutex.acquire(timeout=2.0):
                try:
                    self._cleanup_unlocked()
                    logger.info("Microphone deactivated: stream closed completely.")
                finally:
                    self._stream_mutex.release()

        threading.Thread(target=_async_close, daemon=True).start()
        return audio, duration

    def set_device(self, device: Optional[int]):
        """Sets the preferred audio device (takes effect on next recording)."""
        self.device = device

    def close(self):
        """Immediately closes any active stream on application shutdown."""
        with self._lock:
            self._recording = False
        if self._stream_mutex.acquire(timeout=1.0):
            try:
                self._cleanup_unlocked()
            finally:
                self._stream_mutex.release()

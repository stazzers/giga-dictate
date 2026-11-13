import logging
import threading
import time

import numpy as np
import sounddevice as sd

logger = logging.getLogger("giga_dictate.audio")

SAMPLE_RATE = 16000


def split_audio_into_chunks(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    max_sec: float = 20.0,
    min_split_sec: float = 12.0,
) -> list[np.ndarray]:
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

        # Calculate energy in 100ms frames without extra array allocations
        frame_size = int(0.1 * sr)
        num_frames = len(search_audio) // frame_size
        if num_frames > 0:
            frames = search_audio[: num_frames * frame_size].reshape(
                num_frames, frame_size
            )
            energies = np.einsum("ij,ij->i", frames, frames)
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
    When idle, no audio streams or devices are kept open. Each recording opens a
    fresh CoreAudio stream: another capture session can invalidate a paused one.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, device: int | None = None):
        self.sample_rate = sample_rate
        self.device = device
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._sample_count = 0
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

    def get_audio_state(self) -> tuple[float, float]:
        """Returns (amplitude_level, speech_speed) in a single atomic call."""
        if not self._recording:
            return 0.0, 1.0
        return self._current_level, self._speech_speed

    def get_stream_snapshot(self, max_seconds: float | None = None) -> np.ndarray:
        """
        Thread-safe snapshot of recorded audio for live preview streaming.
        If max_seconds is None, returns all accumulated audio of the current recording session
        so full phrases can be transcribed. If max_seconds is a positive number, returns the tail slice.
        If max_seconds <= 0, returns an empty array.
        Does not interrupt active recording or cause audio buffer dropouts.
        """
        if max_seconds is not None and max_seconds <= 0:
            return np.array([], dtype=np.float32)

        with self._lock:
            if not self._recording or not self._frames:
                return np.array([], dtype=np.float32)

            n_frames = len(self._frames)
            if max_seconds is not None:
                max_samples = int(max_seconds * self.sample_rate)
                if max_samples <= 0:
                    return np.array([], dtype=np.float32)

                # Pre-compute needed frame boundaries from the end without O(N^2) buffer reallocation
                accumulated = 0
                start_idx = 0
                for idx in range(n_frames - 1, -1, -1):
                    accumulated = accumulated + len(self._frames[idx])
                    if accumulated >= max_samples:
                        start_idx = idx
                        break
                frames_slice = self._frames[start_idx:]
            else:
                max_samples = None
                frames_slice = list(self._frames)

        if not frames_slice:
            return np.array([], dtype=np.float32)

        # Instantaneous tail slice extraction: avoid full-array concatenation when single frame
        if len(frames_slice) == 1:
            audio = frames_slice[0].ravel()
        else:
            audio = np.concatenate(frames_slice, axis=0).ravel()

        if max_samples is not None and len(audio) > max_samples:
            audio = audio[-max_samples:]
        return audio.astype(np.float32, copy=False)

    def get_stream_snapshot_from(self, offset: int) -> np.ndarray:
        """Copy only uncommitted samples; call on the preview worker, not the UI.

        Audio frames remain immutable after capture. Copy their references under
        the lock and concatenate outside it, keeping the real-time callback free.
        """
        with self._lock:
            if not self._recording:
                return np.array([], dtype=np.float32)
            end = self._sample_count
            selected = []
            for frame in reversed(self._frames):
                start = end - len(frame)
                if end <= offset:
                    break
                selected.append(frame[max(0, offset - start):])
                end = start
        selected.reverse()
        if not selected:
            return np.array([], dtype=np.float32)
        if len(selected) == 1:
            return selected[0].ravel().astype(np.float32, copy=False)
        return np.concatenate(selected, axis=0).ravel().astype(np.float32, copy=False)

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio stream status: {status}")

        if not self._recording:
            return

        with self._lock:
            if not self._recording:
                return
            self._frames.append(indata.copy())
            self._sample_count += len(indata)

        try:
            flat = indata.ravel()
            n_samples = len(flat)
            if n_samples == 0:
                return

            # Vectorized RMS via BLAS scalar product (zero intermediate array allocations)
            rms = float(np.sqrt(np.dot(flat, flat) / n_samples))
            if rms < 0.002:
                target = 0.0
                instant_speed = 1.0
            else:
                target = min(1.0, max(0.0, (rms - 0.002) * 35.0))
                # Syllable delta and articulation / timbre proxy
                delta_e = abs(rms - self._prev_rms)
                self._prev_rms = rms

                # Fast zero-crossing rate via signbit transitions (no temporary array multiplications)
                signs = np.signbit(flat)
                zcr = (
                    float(np.count_nonzero(signs[:-1] != signs[1:]) / (n_samples - 1))
                    if n_samples > 1
                    else 0.0
                )

                # Dynamic speed multiplier (1.0x for slow, up to 3.0x for fast dictation)
                instant_speed = 1.0 + min(2.2, delta_e * 55.0 + zcr * 10.0)

            # Responsive volume attack (snappier, less sluggish, instant syllable rise)
            if target > self._current_level:
                self._current_level = self._current_level * 0.10 + target * 0.90
            else:
                self._current_level = self._current_level * 0.50 + target * 0.50

            # Smooth speed transition (faster cadence tracking)
            self._speech_speed = self._speech_speed * 0.60 + instant_speed * 0.40
        except Exception as e:
            logger.debug(f"Audio volume computation error ignored: {e}")

    def start(self) -> bool:
        """
        Activates the microphone and starts capturing audio frames.
        Thread-safe, non-blocking, and deadlock-free.
        Never resumes an old AudioUnit after a meeting or input route change.
        """
        with self._stream_mutex:
            with self._lock:
                if self._recording:
                    return False
            if not self._close_stream_locked():
                return False
            with self._lock:
                self._frames = []
                self._sample_count = 0
                self._recording = True
                self._current_level = 0.0
                self._prev_rms = 0.0
                self._speech_speed = 1.0

            # Open a fresh stream, including after another recorder used the mic.
            for attempt in range(2):
                try:
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
                    logger.warning(
                        f"Failed to open microphone (attempt {attempt + 1}/2): {e}"
                    )
                    if not self._close_stream_locked():
                        break
                    if attempt == 0:
                        time.sleep(0.05)

            with self._lock:
                self._recording = False
            return False

    def stop(self) -> tuple[np.ndarray, float]:
        """
        Stops recording and closes microphone IO before handing off audio.
        Returns audio frames instantly.
        """
        with self._stream_mutex:
            with self._lock:
                self._recording = False
                self._current_level = 0.0
                frames = self._frames
                self._frames = []
                self._sample_count = 0
            self._close_stream_locked()

        if frames:
            audio = np.concatenate(frames, axis=0).ravel()
        else:
            audio = np.array([], dtype=np.float32)
        duration = len(audio) / self.sample_rate

        return audio, duration

    def _close_stream_locked(self) -> bool:
        """Caller holds _stream_mutex, never _lock (callbacks may be finishing)."""
        stream = self._stream
        if stream is None:
            return True
        try:
            if not stream.closed and stream.active:
                stream.stop()
        except Exception:
            logger.debug("microphone_stop_failed; closing stream")
        try:
            if not stream.closed:
                stream.close()
        except Exception:
            # Keep the handle for a later cleanup attempt; never overlap streams.
            logger.warning("microphone_close_failed")
            return False
        self._stream = None
        logger.info("Microphone deactivated: stream closed.")
        return True

    def set_device(self, device: int | None):
        """Sets the preferred audio device (takes effect on next recording)."""
        with self._stream_mutex:
            if self.device != device:
                if self._close_stream_locked():
                    self.device = device

    def close(self):
        """Immediately closes any active stream on application shutdown."""
        with self._stream_mutex:
            with self._lock:
                self._recording = False
                self._current_level = 0.0
                self._frames = []
                self._sample_count = 0
            self._close_stream_locked()

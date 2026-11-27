import gc
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

# Set before importing the backend, including its indirect import by GigaAM.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gigaam
import numpy as np
import torch

from .audio import split_audio_into_chunks

logger = logging.getLogger("giga_dictate.engine")

_SPACES_RE = re.compile(r"\s+")

@lru_cache(maxsize=1)
def performance_cores():
    if sys.platform == "darwin":
        try:
            cores = int(subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "hw.perflevel0.physicalcpu"],
                stderr=subprocess.DEVNULL, timeout=.25,
            ))
            if cores > 0:
                return cores
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return os.cpu_count() or 4


def torch_threads(device):
    return min(performance_cores(), 4 if device == "mps" else 8)


@dataclass
class _PreviewCache:
    offset: int = 0
    parts: list[str] = field(default_factory=list)
    last_size: int = 0
    tail_text: str = ""


class GigaAMEngine:
    """Manages GigaAM speech recognition model and high-speed in-memory inference."""

    def __init__(self, model_name: str = "v3_e2e_ctc", device: str = "mps"):
        self.model_name = model_name
        self.requested_device = device
        self.device_str = device
        self.model = None
        self._lock = threading.Lock()
        self._loaded_event = threading.Event()
        self._is_loaded = False
        self._load_error: str | None = None
        self._preview_cache = _PreviewCache()
        self._last_preview_len: int = 0
        self._last_preview_text: str = ""

    def reset_preview_cache(self):
        # Replacing the object isolates an in-flight worker from a new session.
        self._preview_cache = _PreviewCache()
        self._last_preview_len = 0
        self._last_preview_text = ""

    @property
    def preview_offset(self) -> int:
        return self._preview_cache.offset

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def wait_loaded(self, timeout: float = 20.0) -> bool:
        """Blocks until model is loaded and ready, or timeout expires."""
        return self._loaded_event.wait(timeout)

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self, model_name: str | None = None, device: str | None = None) -> bool:
        """Loads or reloads the specified GigaAM model and warms up inference."""
        with self._lock:
            self.reset_preview_cache()
            if model_name:
                self.model_name = model_name
            if device:
                self.requested_device = device
            self.device_str = self.requested_device

            self._loaded_event.clear()
            self._is_loaded = False
            candidates = [self.requested_device]
            if self.requested_device == "mps":
                candidates = ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]
            if candidates[0] != self.requested_device:
                logger.warning("MPS device requested but not available. Falling back to CPU.")

            for device in candidates:
                if self.model is not None:
                    self.model = None
                    gc.collect()
                    self._clear_mps_cache()
                self.device_str = device
                start = time.monotonic()
                try:
                    self._set_threads()
                    self.model = gigaam.load_model(self.model_name, device=device, fp16_encoder=False)
                    self._warmup(strict=device == "mps")
                except Exception as error:
                    self._load_error = str(error)
                    logger.warning("asr_load_failed device=%s", device)
                    continue
                self._is_loaded = True
                self._load_error = None
                self._loaded_event.set()
                logger.info("Model '%s' loaded & warmed up on %s in %.2fs",
                            self.model_name, device, time.monotonic() - start)
                return True
            self.model = None
            self._loaded_event.set()
            return False

    @staticmethod
    def _clear_mps_cache():
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except RuntimeError:
            logger.debug("asr_mps_cache_release_failed")

    def _set_threads(self):
        try:
            count = torch_threads(self.device_str)
            torch.set_num_threads(count)
            logger.info("asr_thread_pool device=%s threads=%d", self.device_str, count)
        except (RuntimeError, ValueError):
            logger.debug("asr_thread_pool_unavailable")

    def _warmup(self, *, strict=False) -> None:
        """Run dummy forward passes to compile MPS shaders and warm up caches for both full and preview slices."""
        if self.model is None:
            return
        dummy_audio = None
        wav = None
        length = None
        encoded = None
        encoded_len = None
        try:
            for sec in (1.0, 2.5):
                dummy_audio = np.zeros(int(sec * 16000), dtype=np.float32)
                wav = torch.as_tensor(
                    dummy_audio, dtype=self.model._dtype, device=self.model._device
                ).unsqueeze(0)
                length = torch.full([1], wav.shape[-1], device=self.model._device)
                encoded, encoded_len = self.model.forward(wav, length)
                _ = self.model._decode(encoded, encoded_len, length, word_timestamps=False)
                del dummy_audio, wav, length, encoded, encoded_len
                dummy_audio = wav = length = encoded = encoded_len = None
        except Exception as e:
            if strict:
                raise
            logger.warning(f"Warmup pass warning: {e}")
        finally:
            del dummy_audio, wav, length, encoded, encoded_len
            if self.device_str == "mps":
                self._clear_mps_cache()

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribes 1D float32 audio numpy array.
        Handles long audio (>20s) by splitting into silence-aware chunks.
        """
        if self.model is None or not self._is_loaded:
            raise RuntimeError(f"Model is not loaded. Error: {self._load_error}")

        if len(audio) == 0:
            return ""

        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio)
        if audio.ndim != 1:
            audio = audio.ravel()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        if len(audio) <= int(20.0 * sample_rate):
            chunks = [audio]
        else:
            chunks = split_audio_into_chunks(audio, sr=sample_rate, max_sec=20.0)

        results: list[str] = []

        with self._lock:
            if self.model is None or not self._is_loaded:
                raise RuntimeError("Model is not loaded")
            for chunk in chunks:
                if len(chunk) < int(0.2 * sample_rate):
                    continue

                text = self._decode_with_fallback(chunk)
                if text:
                    results.append(text)

        full_text = " ".join(results)
        # Normalize repeated spaces
        full_text = _SPACES_RE.sub(" ", full_text).strip()
        return full_text

    @torch.inference_mode()
    def transcribe_preview(self, audio: np.ndarray, sample_rate: int = 16000,
                           *, offset: int | None = None, cancelled=None) -> str:
        """Recognize each committed chunk once, revising only the <=20s tail.

        offset=None accepts the legacy full-recording input. Streaming callers
        supply samples starting at preview_offset, avoiding full-history copies.
        Final transcription always performs its own independent pass.
        """
        if audio is None:
            return ""
        audio = np.asarray(audio, dtype=np.float32).ravel()
        if not np.isfinite(audio).all() or not self._lock.acquire(blocking=False):
            return ""
        try:
            if self.model is None or not self._is_loaded:
                return ""
            cache = self._preview_cache
            if offset is None:
                if audio.size < cache.offset + cache.last_size:
                    self.reset_preview_cache()
                    cache = self._preview_cache
                audio = audio[cache.offset:]
            elif offset != cache.offset:
                return ""  # Snapshot belongs to an obsolete cache/session.

            def stale():
                return cache is not self._preview_cache or (cancelled is not None and cancelled())

            max_len = int(20.0 * sample_rate)
            while audio.size > max_len:
                if stale():
                    return ""
                # Use exactly the same silence boundary as the final chunker.
                chunk = split_audio_into_chunks(audio[:max_len + 1], sr=sample_rate, max_sec=20.0)[0]
                text = self._decode_preview_chunk(chunk)
                if stale():
                    return ""
                cache.parts.append(text)
                cache.offset += chunk.size
                cache.last_size = 0
                cache.tail_text = ""
                audio = audio[chunk.size:]

            if stale():
                return ""
            if audio.size >= int(0.3 * sample_rate):
                unchanged = audio.size == cache.last_size and cache.last_size > 0
                quiet_delta = False
                if cache.last_size > 0 and audio.size > cache.last_size:
                    delta = audio[cache.last_size:]
                    quiet_delta = float(np.dot(delta, delta) / delta.size) < 9e-6
                if not unchanged and not quiet_delta:
                    text = self._decode_preview_chunk(audio)
                    if stale():
                        return ""
                    cache.tail_text = text
                cache.last_size = audio.size
            text = " ".join(part for part in [*cache.parts, cache.tail_text] if part)
            if stale():
                return ""
            self._last_preview_len = cache.offset + cache.last_size
            self._last_preview_text = text
            return text
        except Exception:
            logger.debug("preview_decode_failed")
            return ""
        finally:
            self._lock.release()

    def _decode_preview_chunk(self, audio):
        return self._decode_with_fallback(audio, preview=True)

    def _decode_with_fallback(self, audio, *, preview=False):
        try:
            return self._decode_chunk(audio, preview=preview)
        except RuntimeError as error:
            message = str(error).lower()
            if self.device_str != "mps" or ("mps" not in message and "metal" not in message):
                raise
        logger.warning("asr_mps_failed fallback=cpu")
        try:
            self.model.to("cpu")
        except Exception:
            self._is_loaded = False
            self._load_error = "Не удалось перенести GigaAM на CPU"
            raise
        self.device_str = "cpu"
        self._clear_mps_cache()
        self._set_threads()
        return self._decode_chunk(audio, preview=preview)

    def _decode_chunk(self, audio, *, preview):
        wav = torch.as_tensor(audio, dtype=self.model._dtype, device=self.model._device).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=self.model._device)
        encoded, encoded_len = self.model.forward(wav, length)
        decoded = self.model._decode(encoded, encoded_len, length, word_timestamps=False)
        if not decoded or decoded[0] is None:
            return ""
        first = decoded[0]
        raw = first[0] if isinstance(first, (list, tuple)) and first else first
        if not isinstance(raw, str) or (preview and raw.strip().startswith(("⁇", "\ufffd"))):
            return ""
        return _SPACES_RE.sub(" ", raw.replace("⁇", "").replace("\ufffd", "")).strip()

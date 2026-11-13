from dataclasses import dataclass, field

import gc
import logging
import os
import re
import threading
import time

import gigaam
import numpy as np
import torch

from .audio import split_audio_into_chunks

logger = logging.getLogger("giga_dictate.engine")

# Optimize PyTorch CPU thread pool to prevent thread contention and background CPU wakeups
try:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
except (RuntimeError, ValueError) as e:
    logger.debug(f"PyTorch thread count setting error ignored: {e}")


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
                self.device_str = device

            # Fallback to cpu if mps not available
            if self.device_str == "mps" and not torch.backends.mps.is_available():
                logger.warning(
                    "MPS device requested but not available. Falling back to CPU."
                )
                self.device_str = "cpu"

            logger.info(
                f"Loading GigaAM model '{self.model_name}' on '{self.device_str}'..."
            )
            self._loaded_event.clear()
            self._is_loaded = False

            # Explicitly release any previously loaded model to avoid doubling RAM/VRAM usage
            if self.model is not None:
                old_model = self.model
                self.model = None
                del old_model
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            t0 = time.time()
            try:
                # fp16_encoder=False is optimal for MPS float32 convs
                self.model = gigaam.load_model(
                    self.model_name,
                    device=self.device_str,
                    fp16_encoder=False,
                )
                self._warmup()
                self._is_loaded = True
                self._load_error = None
                self._loaded_event.set()
                logger.info(
                    f"Model '{self.model_name}' loaded & warmed up on {self.device_str} in {time.time() - t0:.2f}s"
                )
                return True
            except Exception as e:
                # If MPS initialization or warmup failed, fallback to CPU
                if self.device_str == "mps":
                    logger.warning(
                        f"Failed to load model on MPS ({e}). Falling back to CPU..."
                    )
                    try:
                        self.device_str = "cpu"
                        self.model = gigaam.load_model(
                            self.model_name,
                            device="cpu",
                            fp16_encoder=False,
                        )
                        self._warmup()
                        self._is_loaded = True
                        self._load_error = None
                        self._loaded_event.set()
                        logger.info(
                            f"Model '{self.model_name}' loaded & warmed up on CPU fallback in {time.time() - t0:.2f}s"
                        )
                        return True
                    except Exception as cpu_e:
                        logger.error(f"CPU fallback also failed: {cpu_e}")
                self._is_loaded = False
                self._load_error = str(e)
                self._loaded_event.set()  # unblock waiters with error
                logger.error(
                    f"Failed to load model {self.model_name}: {e}", exc_info=True
                )
                return False

    def _warmup(self) -> None:
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
            logger.warning(f"Warmup pass warning: {e}")
        finally:
            del dummy_audio, wav, length, encoded, encoded_len
            if self.device_str == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()

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

                wav = None
                length = None
                encoded = None
                encoded_len = None
                try:
                    wav = torch.as_tensor(
                        chunk,
                        dtype=self.model._dtype,
                        device=self.model._device,
                    ).unsqueeze(0)
                    length = torch.full(
                        [1], wav.shape[-1], device=self.model._device
                    )

                    encoded, encoded_len = self.model.forward(wav, length)
                    decoded = self.model._decode(
                        encoded, encoded_len, length, word_timestamps=False
                    )
                    if decoded and len(decoded) > 0 and decoded[0] is not None:
                        first = decoded[0]
                        raw = (
                            first[0]
                            if isinstance(first, (list, tuple)) and len(first) > 0
                            else first
                        )
                        if isinstance(raw, str):
                            text = raw.strip()
                            if text:
                                results.append(text)
                finally:
                    del wav, length, encoded, encoded_len

        full_text = " ".join(results)
        # Normalize repeated spaces
        full_text = re.sub(r"\s+", " ", full_text).strip()
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
        except Exception as error:
            logger.debug("preview_decode_failed")
            if self.device_str == "mps" and ("mps" in str(error).lower() or "metal" in str(error).lower()):
                logger.warning("preview_mps_failed fallback=cpu")
                try:
                    self.model.to("cpu")
                    self.model._device = torch.device("cpu")
                    self.device_str = "cpu"
                    self.reset_preview_cache()
                except Exception:
                    logger.warning("preview_cpu_fallback_failed")
            return ""
        finally:
            self._lock.release()

    def _decode_preview_chunk(self, audio):
        wav = torch.as_tensor(audio, dtype=self.model._dtype, device=self.model._device).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=self.model._device)
        encoded, encoded_len = self.model.forward(wav, length)
        decoded = self.model._decode(encoded, encoded_len, length, word_timestamps=False)
        if not decoded or decoded[0] is None:
            return ""
        first = decoded[0]
        raw = first[0] if isinstance(first, (list, tuple)) and first else first
        if not isinstance(raw, str) or raw.strip().startswith(("⁇", "\ufffd")):
            return ""
        return re.sub(r"\s+", " ", raw.replace("⁇", "").replace("\ufffd", "")).strip()

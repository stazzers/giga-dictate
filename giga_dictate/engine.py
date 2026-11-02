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
        self._last_preview_len: int = 0
        self._last_preview_text: str = ""

    def reset_preview_cache(self):
        """Resets the VAD energy and preview transcription cache for a new recording session."""
        self._last_preview_len = 0
        self._last_preview_text = ""

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
    def transcribe_preview(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Fast low-latency inference on the latest audio slice for Live HUD Preview.
        Runs single greedy forward pass without chunk splitting.
        Latency: ~20-35ms on Apple Silicon GPU (MPS).
        """
        if (
            self.model is None
            or not self._is_loaded
            or audio is None
        ):
            return ""

        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio)
        if audio.ndim != 1:
            audio = audio.ravel()
        if audio.size < int(0.3 * sample_rate) or not np.isfinite(audio).all():
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        # VAD Energy & Change Detection:
        # If input audio is shorter than previous snapshot, a new recording started -> reset cache
        if audio.size < self._last_preview_len:
            self._last_preview_len = 0
            self._last_preview_text = ""

        # If audio length has not changed, reuse cached transcription immediately
        if self._last_preview_len > 0 and audio.size == self._last_preview_len:
            return self._last_preview_text

        # If audio has accumulated and new delta slice is silence (< 0.003 RMS),
        # reuse cached transcription immediately to avoid redundant GPU forward passes during speech pauses.
        if self._last_preview_len > 0 and audio.size > self._last_preview_len:
            delta = audio[self._last_preview_len:]
            if delta.size > 0:
                delta_energy = float(np.dot(delta, delta) / delta.size)
                # Energy threshold 9e-6 corresponds to RMS ~ 0.003
                if delta_energy < 9e-6:
                    self._last_preview_len = audio.size
                    return self._last_preview_text

        # Acquire lock non-blockingly so preview never stalls final transcription
        if not self._lock.acquire(blocking=False):
            return ""

        wav = None
        length = None
        encoded = None
        encoded_len = None
        try:
            if audio.size <= int(20.0 * sample_rate):
                wav = torch.as_tensor(
                    audio,
                    dtype=self.model._dtype,
                    device=self.model._device,
                ).unsqueeze(0)
                length = torch.full([1], wav.shape[-1], device=self.model._device)

                encoded, encoded_len = self.model.forward(wav, length)
                decoded = self.model._decode(
                    encoded, encoded_len, length, word_timestamps=False
                )
                raw_text = ""
                if decoded and len(decoded) > 0 and decoded[0] is not None:
                    first = decoded[0]
                    if isinstance(first, (list, tuple)) and len(first) > 0:
                        raw_text = first[0]
                    elif isinstance(first, str):
                        raw_text = first

                if raw_text and isinstance(raw_text, str):
                    clean = raw_text.strip()
                    if clean and not clean.startswith("⁇") and not clean.startswith("\ufffd"):
                        clean = clean.replace("⁇", "").replace("\ufffd", "").strip()
                        if clean:
                            clean = re.sub(r"\s+", " ", clean)
                            self._last_preview_len = audio.size
                            self._last_preview_text = clean
                            return clean
                self._last_preview_len = audio.size
                return ""
            else:
                chunks = split_audio_into_chunks(audio, sr=sample_rate, max_sec=20.0)
                parts: list[str] = []
                for chunk in chunks:
                    if len(chunk) < int(0.2 * sample_rate):
                        continue
                    c_wav = torch.as_tensor(
                        chunk, dtype=self.model._dtype, device=self.model._device
                    ).unsqueeze(0)
                    c_len = torch.full([1], c_wav.shape[-1], device=self.model._device)
                    c_enc, c_enc_len = self.model.forward(c_wav, c_len)
                    c_dec = self.model._decode(
                        c_enc, c_enc_len, c_len, word_timestamps=False
                    )
                    del c_wav, c_len, c_enc, c_enc_len
                    if c_dec and len(c_dec) > 0 and c_dec[0] is not None:
                        first = c_dec[0]
                        c_raw = (
                            first[0]
                            if isinstance(first, (list, tuple)) and len(first) > 0
                            else first
                        )
                        if isinstance(c_raw, str):
                            c_clean = c_raw.replace("⁇", "").replace("\ufffd", "").strip()
                            if c_clean:
                                parts.append(c_clean)
                full = " ".join(parts).strip()
                if full:
                    full = re.sub(r"\s+", " ", full)
                self._last_preview_len = audio.size
                self._last_preview_text = full
                return full
        except Exception as e:
            logger.debug(f"Preview decode error: {e}")
            if self.device_str == "mps" and ("MPS" in str(e) or "metal" in str(e).lower()):
                logger.warning(f"MPS error during live preview ({e}). Falling back to CPU.")
                try:
                    if self.model is not None:
                        self.model.to("cpu")
                        self.model._device = torch.device("cpu")
                    self.device_str = "cpu"
                except Exception as fb_err:
                    logger.debug(f"CPU fallback error: {fb_err}")
            return ""
        finally:
            del wav, length, encoded, encoded_len
            self._lock.release()

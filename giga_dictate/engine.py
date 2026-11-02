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
        """Run a dummy forward pass to compile MPS shaders and warm up caches."""
        if self.model is None:
            return
        dummy_audio = None
        wav = None
        length = None
        encoded = None
        encoded_len = None
        try:
            dummy_audio = np.zeros(16000, dtype=np.float32)
            wav = torch.as_tensor(
                dummy_audio, dtype=self.model._dtype, device=self.model._device
            ).unsqueeze(0)
            length = torch.full([1], wav.shape[-1], device=self.model._device)
            encoded, encoded_len = self.model.forward(wav, length)
            _ = self.model._decode(encoded, encoded_len, length, word_timestamps=False)
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

        chunks = split_audio_into_chunks(audio, sr=sample_rate, max_sec=20.0)
        results: list[str] = []

        with self._lock:
            try:
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
                        if decoded and decoded[0][0]:
                            text = decoded[0][0].strip()
                            if text:
                                results.append(text)
                    finally:
                        del wav, length, encoded, encoded_len
            finally:
                if self.device_str == "mps" and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()

        full_text = " ".join(results)
        # Normalize repeated spaces
        full_text = re.sub(r"\s+", " ", full_text).strip()
        return full_text

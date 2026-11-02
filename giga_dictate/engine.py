import logging
import re
import threading
import time
from typing import List, Optional
import numpy as np
import torch
import gigaam

from .audio import split_audio_into_chunks

logger = logging.getLogger("giga_dictate.engine")


class GigaAMEngine:
    """Manages GigaAM speech recognition model and high-speed in-memory inference."""

    def __init__(self, model_name: str = "v3_e2e_ctc", device: str = "mps"):
        self.model_name = model_name
        self.device_str = device
        self.model = None
        self._lock = threading.Lock()
        self._loaded_event = threading.Event()
        self._is_loaded = False
        self._load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def wait_loaded(self, timeout: float = 20.0) -> bool:
        """Blocks until model is loaded and ready, or timeout expires."""
        return self._loaded_event.wait(timeout)

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load(self, model_name: Optional[str] = None, device: Optional[str] = None) -> bool:
        """Loads or reloads the specified GigaAM model and warms up inference."""
        with self._lock:
            if model_name:
                self.model_name = model_name
            if device:
                self.device_str = device

            # Fallback to cpu if mps not available
            if self.device_str == "mps" and not torch.backends.mps.is_available():
                logger.warning("MPS device requested but not available. Falling back to CPU.")
                self.device_str = "cpu"

            logger.info(f"Loading GigaAM model '{self.model_name}' on '{self.device_str}'...")
            self._loaded_event.clear()
            self._is_loaded = False
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
                logger.info(f"Model '{self.model_name}' loaded & warmed up in {time.time()-t0:.2f}s")
                return True
            except Exception as e:
                self._is_loaded = False
                self._load_error = str(e)
                self._loaded_event.set()  # unblock waiters with error
                logger.error(f"Failed to load model {self.model_name}: {e}", exc_info=True)
                return False

    def _warmup(self) -> None:
        """Run a dummy forward pass to compile MPS shaders and warm up caches."""
        if self.model is None:
            return
        try:
            dummy_audio = np.zeros(16000, dtype=np.float32)
            wav = torch.as_tensor(dummy_audio, dtype=self.model._dtype, device=self.model._device).unsqueeze(0)
            length = torch.full([1], wav.shape[-1], device=self.model._device)
            encoded, encoded_len = self.model.forward(wav, length)
            _ = self.model._decode(encoded, encoded_len, length, word_timestamps=False)
        except Exception as e:
            logger.warning(f"Warmup pass warning: {e}")

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
        results: List[str] = []

        with self._lock:
            for chunk in chunks:
                if len(chunk) < int(0.2 * sample_rate):
                    continue

                wav = torch.as_tensor(
                    chunk,
                    dtype=self.model._dtype,
                    device=self.model._device,
                ).unsqueeze(0)
                length = torch.full([1], wav.shape[-1], device=self.model._device)

                encoded, encoded_len = self.model.forward(wav, length)
                decoded = self.model._decode(encoded, encoded_len, length, word_timestamps=False)
                if decoded and decoded[0][0]:
                    text = decoded[0][0].strip()
                    if text:
                        results.append(text)

        full_text = " ".join(results)
        # Normalize repeated spaces
        full_text = re.sub(r"\s+", " ", full_text).strip()
        return full_text

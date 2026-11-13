"""Verify prepared assets locally. This module never downloads during capture."""
import hashlib
import json
from pathlib import Path

from .diarization import MODEL_ID, PREPROCESSING


def default_model_dirs():
    root = Path(__file__).resolve().parent.parent
    return [root / "meeting_models", root / "build" / "meeting_models",
            Path.home() / ".config" / "giga_dictate" / "meeting_models"]


def verify_models(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("model") != MODEL_ID or manifest.get("preprocessing") != PREPROCESSING:
        raise ValueError("Несовместимые модели встреч")
    for name in ("silero_vad.onnx", "campplus.onnx", "LICENSE-CAM++.txt", "LICENSE-Silero.txt"):
        expected = manifest["sha256"][name]
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError("Контрольная сумма модели встречи не совпадает")
    for name in ("LICENSE-CAM++.txt", "LICENSE-Silero.txt"):
        if not (directory / name).is_file():
            raise ValueError("Отсутствует лицензия модели")
    return directory


def find_models():
    for directory in default_model_dirs():
        if (directory / "manifest.json").exists():
            return verify_models(directory)
    raise ValueError("Модели встреч ещё не подготовлены. Выберите «Подготовить модели встреч» в меню.")


def prepare_models(destination=None):
    """Explicit pre-recording setup. Only pinned upstream bytes are executed.

    The checkpoint is loaded with weights_only=True. Export is compared with
    PyTorch before publishing a manifest, so interrupted setup is never ready.
    """
    import importlib.util
    from importlib.metadata import distribution
    import os
    import shutil
    import sys
    import tempfile
    import types
    import urllib.request

    if destination is None:
        try:
            return find_models()
        except (ValueError, OSError, KeyError):
            pass
    destination = Path(destination) if destination else default_model_dirs()[-1]
    if (destination / "manifest.json").exists():
        try:
            return verify_models(destination)
        except (ValueError, OSError, KeyError):
            pass  # Explicit setup can repair an interrupted/corrupted download.
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)

    def fetch(url, path, expected):
        with urllib.request.urlopen(url, timeout=60) as response, open(path, "wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Контрольная сумма загрузки не совпадает")

    commit = "065629c313eaf1a01c65c640c46d77e61e9607b4"
    base = f"https://raw.githubusercontent.com/modelscope/3D-Speaker/{commit}/"
    with tempfile.TemporaryDirectory(prefix="giga-meeting-prepare-") as temporary:
        work = Path(temporary)
        fetch(base + "speakerlab/models/campplus/layers.py", work / "layers.py",
              "b3d28ad94d14b30fcd20c96d1d83516d4e44de733959460b358e90d97476adb8")
        fetch(base + "speakerlab/models/campplus/DTDNN.py", work / "DTDNN.py",
              "52272359361ef9f03a8b4198f50139301ce712833e777396940a5b437177cd34")
        fetch(base + "LICENSE", work / "LICENSE-CAM++.txt",
              "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4")
        model_url = "https://modelscope.cn/api/v1/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/repo?Revision=v1.0.0&FilePath=campplus_cn_en_common.pt"
        fetch(model_url, work / "weights.pt",
              "92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199")
        source = (work / "DTDNN.py").read_text().replace("from speakerlab.models.campplus.layers", "from .layers")
        (work / "DTDNN.py").write_text(source)
        # PyTorch excludes the implicit ceil-mode tail padding. Make the same
        # denominator explicit for ONNX AveragePool, including windows <100.
        layers = (work / "layers.py").read_text().replace(
            "stride=seg_len, ceil_mode=True)", "stride=seg_len, ceil_mode=True, count_include_pad=False)", 1)
        (work / "layers.py").write_text(layers)
        package = types.ModuleType("_giga_campplus_export")
        package.__path__ = [str(work)]
        sys.modules[package.__name__] = package
        try:
            spec = importlib.util.spec_from_file_location(package.__name__ + ".DTDNN", work / "DTDNN.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            import torch
            import numpy as np
            import onnxruntime as ort
            model = module.CAMPPlus(feat_dim=80, embedding_size=192).eval()
            model.load_state_dict(torch.load(work / "weights.pt", map_location="cpu", weights_only=True))
            generator = torch.Generator().manual_seed(731)
            sample = torch.randn(1, 150, 80, generator=generator)
            with torch.no_grad():
                expected = model(sample).numpy()
                torch.onnx.export(model, sample, str(work / "campplus.onnx"),
                    input_names=["feature"], output_names=["embedding"],
                    dynamic_axes={"feature": {0: "batch", 1: "frames"}, "embedding": {0: "batch"}},
                    opset_version=17, dynamo=False)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(work / "campplus.onnx"), options, providers=["CPUExecutionProvider"])
            actual = session.run(None, {"feature": sample.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4)
            # Exercise both sides of CAM's 100-frame pooling boundary.
            for length in (99, 345):
                other = torch.randn(1, length, 80, generator=generator)
                with torch.no_grad():
                    expected = model(other).numpy()
                actual = session.run(None, {"feature": other.numpy()})[0]
                np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4)
        finally:
            for key in list(sys.modules):
                if key.startswith(package.__name__):
                    del sys.modules[key]
        dist = distribution("silero-vad")
        vad = dist.locate_file("silero_vad/data/silero_vad.onnx")
        if hashlib.sha256(vad.read_bytes()).hexdigest() != "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3":
            raise ValueError("Несовместимая версия Silero VAD")
        shutil.copyfile(vad, work / "silero_vad.onnx")
        license_path = next(path for path in dist.files if str(path).endswith("licenses/LICENSE"))
        shutil.copyfile(dist.locate_file(license_path), work / "LICENSE-Silero.txt")
        names = ("campplus.onnx", "silero_vad.onnx", "LICENSE-CAM++.txt", "LICENSE-Silero.txt")
        hashes = {name: hashlib.sha256((work / name).read_bytes()).hexdigest() for name in names}
        manifest = {"model": MODEL_ID, "preprocessing": PREPROCESSING,
                    "source_commit": commit, "sha256": hashes,
                    "calibration": "initial thresholds; Russian corpus evaluation required"}
        for name in names:
            shutil.copyfile(work / name, destination / name)
        temporary_manifest = destination / ".manifest.json"
        temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary_manifest, destination / "manifest.json")
    return verify_models(destination)

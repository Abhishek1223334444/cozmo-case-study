"""Local inference only. Download public weights separately; never upload captures."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DAMAGE_ID = "CIDAS/clipseg-rd64-refined"
REVISIONS = {
    "depth": "8078d68a9c75a972131914f6afd0c1723be0da7f",
    "damage": "999e0328d9e10b484360c477313983f9afdd7050",
}


def fetch(kind, directory=Path("models")):
    if kind == "matcher":
        import urllib.request

        folder = Path(directory) / kind
        folder.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/sift_lightglue.pth"
        urllib.request.urlretrieve(url, folder / "sift_lightglue.pth")
        digest = hashlib.sha256((folder / "sift_lightglue.pth").read_bytes()).hexdigest()
        if digest != "5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0":
            raise ValueError("Matcher weight checksum mismatch")
        (folder / "source.json").write_text(json.dumps({"url": url, "sha256": digest}, indent=2))
        return folder
    from huggingface_hub import snapshot_download

    model_id = DEPTH_ID if kind == "depth" else DAMAGE_ID
    revision = REVISIONS[kind]
    folder = Path(directory) / kind
    snapshot_download(
        model_id,
        revision=revision,
        local_dir=folder,
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
    )
    (folder / "source.json").write_text(
        json.dumps({"repository": model_id, "revision": revision}, indent=2)
    )
    return folder


def device_name():
    import torch

    return (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )


class MetricDepth:
    def __init__(self, directory=Path("models/depth"), device="auto"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device_name() if device == "auto" else device
        torch.set_num_threads(4)
        self.processor = AutoImageProcessor.from_pretrained(
            str(directory), local_files_only=True, use_fast=False
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            str(directory), local_files_only=True
        )
        self.model.to(self.device).eval()
        self.source = json.loads((Path(directory) / "source.json").read_text())

    def predict(self, rgb, cache_dir):
        import cv2
        import torch

        key = hashlib.sha256(
            rgb.tobytes() + json.dumps(self.source, sort_keys=True).encode()
        ).hexdigest()
        path = Path(cache_dir) / f"depth-{key}.npy"
        if path.exists():
            return np.load(path)
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            prediction = self.model(**inputs).predicted_depth
        depth = prediction[0].float().cpu().numpy()
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        depth = np.clip(depth, 0.15, 20).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, depth)
        return depth


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["depth", "damage", "matcher", "all"])
    p.add_argument("--directory", type=Path, default=Path("models"))
    a = p.parse_args()
    for kind in ["depth", "damage", "matcher"] if a.kind == "all" else [a.kind]:
        print(fetch(kind, a.directory))

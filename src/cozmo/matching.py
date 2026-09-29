"""Offline RootSIFT + LightGlue; match coordinates are cached per image content."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from lightglue import SIFT, LightGlue


class Matcher:
    def __init__(self, views, directory=Path("models/matcher"), cache=Path("out/cache")):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model = LightGlue(
            features=None,
            input_dim=128,
            add_scale_ori=True,
            flash=False,
            depth_confidence=0.9,
            width_confidence=0.95,
        ).eval()
        state = torch.load(
            Path(directory) / "sift_lightglue.pth", map_location="cpu", weights_only=True
        )
        for i in range(self.model.conf.n_layers):
            state = {
                k.replace(f"self_attn.{i}", f"transformers.{i}.self_attn").replace(
                    f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                ): v
                for k, v in state.items()
            }
        state.setdefault("confidence_thresholds", self.model.confidence_thresholds)
        self.model.load_state_dict(state, strict=True)
        self.source = json.loads((Path(directory) / "source.json").read_text())
        extractor = SIFT(max_num_keypoints=1024).eval()
        self.features, self.keys = [], []
        for v in views:
            rgb = torch.from_numpy(v.rgb.copy()).permute(2, 0, 1).float() / 255
            with torch.inference_mode():
                self.features.append(extractor.extract(rgb, resize=None))
            self.keys.append(hashlib.sha256(v.rgb.tobytes()).hexdigest()[:20])

    def pair(self, i, j):
        key = hashlib.sha256(
            (self.keys[i] + self.keys[j] + self.source["sha256"]).encode()
        ).hexdigest()
        file = self.cache / f"matches-{key}.npz"
        if file.exists():
            z = np.load(file)
            return z["a"], z["b"]
        if min(len(self.features[k]["keypoints"][0]) for k in (i, j)) < 8:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        with torch.inference_mode():
            out = self.model({"image0": self.features[i], "image1": self.features[j]})
        ids = out["matches"][0]
        a = self.features[i]["keypoints"][0][ids[:, 0]].cpu().numpy()
        b = self.features[j]["keypoints"][0][ids[:, 1]].cpu().numpy()
        np.savez_compressed(file, a=a, b=b)
        return a, b

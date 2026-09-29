# Model and software disclosure

All inference is local. No capture images were sent to model providers.

| Component | Source | Pinned identity |
|---|---|---|
| Metric depth | [Depth Anything V2 metric indoor small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf) | `8078d68a9c75a972131914f6afd0c1723be0da7f` |
| Damage candidate segmentation | [CLIPSeg rd64 refined](https://huggingface.co/CIDAS/clipseg-rd64-refined) | `999e0328d9e10b484360c477313983f9afdd7050` |
| Feature matching | [LightGlue](https://github.com/cvg/LightGlue), SIFT weights v0.1_arxiv | Code commit in `uv.lock`; weight SHA-256 `5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0` |
| Input format / capture app | [Stray Scanner](https://github.com/strayrobots/scanner/blob/main/docs/format.md) | Supplied raw ZIPs; archive hashes in downloader |

No fine-tuning was performed. SIFT is used as the detector; SuperPoint weights are
not used. Upstream model/software licenses remain applicable; model pages are the
authoritative licensing sources. Dependencies and exact versions are in `uv.lock`.

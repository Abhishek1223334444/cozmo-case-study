"""Standalone interactive viewer: no network, CDN, or server needed."""

import json
from pathlib import Path


def write_viewer(plan, path):
    template = (Path(__file__).parent / "viewer.html").read_text()
    payload = json.dumps(plan, allow_nan=False).replace("<", "\\u003c")
    Path(path).write_text(template.replace("__PLAN_DATA__", payload))

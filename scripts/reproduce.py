"""Reproduce supplied-sample artifacts; run from the repository root."""

import argparse
import subprocess
import sys


def call(*args):
    subprocess.run([sys.executable, "-m", "cozmo.cli", *map(str, args)], check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rgb", action="store_true", help="Also run RGB-only derived photos and video")
    p.add_argument(
        "--damage", action="store_true", help="Also run floor plans with damage assessment"
    )
    p.add_argument(
        "--property",
        action="store_true",
        help="Also run RGB-only photo/video inputs derived from the larger ceiling scan",
    )
    a = p.parse_args()
    call("benchmark", "--samples", "samples", "--out", "out/benchmark")
    if a.damage:
        call("run", "samples/c00a170fe1", "--damage", "-o", "out/lidar-single-full")
        call("run", "samples/1a8384c3f6", "--damage", "-o", "out/lidar-floor-full")
        call("run", "samples/c7d28f72c6", "--damage", "-o", "out/lidar-property-full")
    if a.rgb:
        call(
            "prepare-sample",
            "samples/c00a170fe1",
            "--plan",
            "out/benchmark/c00a170fe1/after/plan.json",
            "-o",
            "out/inputs/single",
        )
        call("run", "out/inputs/single/photos", "--tier", "photos", "-o", "out/photos-single")
        call(
            "run",
            "samples/c00a170fe1/rgb.mp4",
            "--tier",
            "video",
            "--rotation",
            90,
            "--max-frames",
            80,
            "-o",
            "out/video-single",
        )
        call(
            "run",
            "samples/c00a170fe1/rgb.mp4",
            "--tier",
            "video",
            "--rotation",
            90,
            "--max-frames",
            160,
            "-o",
            "out/video-single-dense",
        )
    if a.property:
        call(
            "prepare-sample",
            "samples/c7d28f72c6",
            "--plan",
            "out/benchmark/c7d28f72c6/after/plan.json",
            "-o",
            "out/inputs/property",
        )
        call("run", "out/inputs/property/photos", "--tier", "photos", "-o", "out/photos-property")
        call(
            "run",
            "samples/c7d28f72c6/rgb.mp4",
            "--tier",
            "video",
            "--rotation",
            90,
            "--max-frames",
            80,
            "-o",
            "out/video-property",
        )
    subprocess.run([sys.executable, "scripts/summarize_results.py"], check=True)

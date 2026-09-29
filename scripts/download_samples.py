"""Download and verify the exact interviewer-provided archives. No Google login required."""

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

FILES = {
    "single_room.zip": (
        "1kNNRQxw9KhkTrqGlprzWR1j51JIu1u71",
        "0805f742d378e4bda480fef6e5839304364807bb7b77bb459983003727e9699c",
    ),
    "single_scan_floor_only.zip": (
        "1Z3qOLpSe7hsJPeKCrYM36l2GXD71yeMd",
        "f822287268297d2adab49deff8e0ee5f50f05304b450d8d47d0b5eb476aa74d4",
    ),
    "single_scan_with_ceiling.zip": (
        "1TlqFfEL7mhY59_sXly9KvD5xSil-Qatm",
        "4bfbeb11ee21b114c46ad43cf0c9602d8ada827397f4e8b3c70dd827d0191379",
    ),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("samples"))
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, (ident, expected) in FILES.items():
        path = a.out / name
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            partial = path.with_suffix(".zip.partial")
            urllib.request.urlretrieve(
                f"https://drive.usercontent.google.com/download?id={ident}&export=download&confirm=t",
                partial,
            )
            if digest(partial) != expected:
                raise ValueError(
                    f"Checksum mismatch for {name}; partial file retained for inspection"
                )
            partial.replace(path)
        if digest(path) != expected:
            raise ValueError(f"Existing {path} has unexpected contents; not overwriting")
        with zipfile.ZipFile(path) as archive:
            roots = {m.filename.split("/")[0] for m in archive.infolist()}
            if len(roots) != 1:
                raise ValueError("Unexpected archive structure")
            folder = a.out / next(iter(roots))
            if not (folder / "odometry.csv").exists():
                for member in archive.infolist():
                    dest = (a.out / member.filename).resolve()
                    if (
                        not dest.is_relative_to(a.out.resolve())
                        or (member.external_attr >> 16) & 0o170000 == 0o120000
                    ):
                        raise ValueError("Unsafe archive member")
                archive.extractall(a.out)
        print(f"Verified {name}: {expected}")


if __name__ == "__main__":
    main()

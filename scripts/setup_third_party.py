"""Fetch the two upstream model runtimes and apply the recorded integration patch.

Existing files are kept if identical, otherwise installation stops. No weights
are downloaded. Upstream branches are explicit; patch drift is an error.
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ("CayleyZ/SmoothSpike", "spikingbert_rot_inf.py", "third_party/SmoothSpike/spikingbert_rot_inf.py"),
    ("CayleyZ/SmoothSpike", "hadamard_utils.py", "third_party/SmoothSpike/hadamard_utils.py"),
    ("CayleyZ/SmoothSpike", "utils.py", "third_party/SmoothSpike/utils.py"),
    ("CayleyZ/SmoothSpike", "convertor.py", "third_party/SmoothSpike/convertor.py"),
    ("BICLab/Spike-Driven-Transformer-V3", "SDT_V3/Classification/Model_Base/models.py",
     "third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py"),
]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    for command in ("git", "curl"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Install {command} before running this script")
    patch = ROOT / "third_party/patches/integration.patch"
    with tempfile.TemporaryDirectory(prefix=".bootstrap-", dir=ROOT) as folder:
        staging = Path(folder)
        for repository, source, destination in FILES:
            target = staging / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/{repository}/main/{source}"
            subprocess.run(
                ["curl", "--fail", "--location", "--silent", "--show-error",
                 "--connect-timeout", "10", "--max-time", "90", "--output", str(target), url],
                check=True,
            )
        for check in (True, False):
            command = ["git", "apply", "--directory", staging.name]
            if check:
                command.append("--check")
            subprocess.run([*command, str(patch)], cwd=ROOT, check=True)
        # Check all destinations before publishing any downloaded file.
        for _, _, destination in FILES:
            existing = ROOT / destination
            if existing.exists() and existing.read_bytes() != (staging / destination).read_bytes():
                raise RuntimeError(f"Existing runtime differs; keep a backup and resolve it explicitly: {existing}")
        for _, _, destination in FILES:
            existing = ROOT / destination
            existing.parent.mkdir(parents=True, exist_ok=True)
            if not existing.exists():
                os.replace(staging / destination, existing)
            print(f"Ready: {destination}")


if __name__ == "__main__":
    main()

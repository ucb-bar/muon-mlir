"""Build the original-size Spatter Gather Cyclotron checker out of tree."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cyclotron", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    (out / "src").mkdir(parents=True, exist_ok=True)
    (out / "src/main.rs").write_bytes((Path(__file__).parent / "spatter_full_gather_check.rs").read_bytes())
    (out / "Cargo.toml").write_text(
        '[package]\nname = "muon_mlir_spatter_full_gather_check"\n'
        'version = "0.1.0"\nedition = "2021"\n\n'
        '[dependencies]\ncyclotron = { path = '
        f'"{args.cyclotron.resolve()}"' + ' }\n')
    env = os.environ.copy()
    target_dir = args.target_dir.resolve() if args.target_dir else out / "target"
    env["CARGO_TARGET_DIR"] = str(target_dir)
    subprocess.run(["cargo", "build", "--release", "--manifest-path", str(out / "Cargo.toml")],
                   check=True, env=env)
    print(target_dir / "release/muon_mlir_spatter_full_gather_check")


if __name__ == "__main__":
    main()

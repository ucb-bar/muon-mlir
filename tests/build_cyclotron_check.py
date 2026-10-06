"""Build the OOT Cyclotron readback helper against an explicit Cyclotron checkout."""
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
    root = args.out.resolve()
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src/main.rs").write_bytes(
        (Path(__file__).parent / "cyclotron_check.rs").read_bytes())
    (root / "Cargo.toml").write_text(
        '[package]\nname = "muon_mlir_cyclotron_check"\n'
        'version = "0.1.0"\nedition = "2021"\n\n'
        '[dependencies]\ncyclotron = { path = '
        f'"{args.cyclotron.resolve()}"' + ' }\n')
    env = os.environ.copy()
    target_dir = args.target_dir.resolve() if args.target_dir else root / "target"
    env["CARGO_TARGET_DIR"] = str(target_dir)
    subprocess.run(["cargo", "build", "--release", "--manifest-path",
                    str(root / "Cargo.toml")], check=True, env=env)
    print(target_dir / "release/muon_mlir_cyclotron_check")


if __name__ == "__main__":
    main()

"""Inventory every local radiance-kernels source unit before claiming coverage.

The file and family lists come from the source tree, not a hand-picked test
roster. Generated inputs and compile configurations are recorded separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_targets(directory: Path) -> dict:
    """Read the evaluated default Makefile targets without building inputs."""
    recipe = ("codex-inventory-print:\n\t@printf "
              "'MU_SRCS=%s\\nMU_SRC_DEPS=%s\\nMU_VARIANTS=%s\\nRADIANCE_TARGETS=%s\\n' "
              "'$(MU_SRCS)' '$(MU_SRC_DEPS)' '$(MU_VARIANTS)' '$(RADIANCE_TARGETS)'")
    environment = os.environ.copy()
    environment.pop("MAKEFLAGS", None)
    result = subprocess.run(
        ["make", "--no-print-directory", "-s", "--eval", recipe,
         "codex-inventory-print"], cwd=directory, env=environment,
        capture_output=True, text=True)
    if result.returncode:
        return {"error": result.stderr.strip()[:1000]}
    values = dict(line.split("=", 1) for line in result.stdout.splitlines()
                  if "=" in line)
    names = {
        "entry_sources": "MU_SRCS", "support_sources": "MU_SRC_DEPS",
        "variants": "MU_VARIANTS", "radiance_elfs": "RADIANCE_TARGETS",
    }
    return {name: list(dict.fromkeys(values.get(variable, "").split()))
            for name, variable in names.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radiance-kernels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.radiance_kernels.resolve()
    files = subprocess.run(
        ["rg", "--files", "kernels", "-g", "*.cpp", "-g", "*.cc",
         "-g", "*.c", "-g", "*.S", "-g", "Makefile"],
        cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()
    families: dict[str, dict] = {}
    for relative in sorted(files):
        parts = Path(relative).parts
        if len(parts) < 3 or parts[0] != "kernels":
            continue
        family = parts[1]
        record = families.setdefault(family, {"name": family, "makefile": None,
                                              "source_units": []})
        path = root / relative
        if parts[-1] == "Makefile":
            record["makefile"] = {"path": relative, "sha256": sha256(path)}
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        host = bool(re.match(r"host([._-]|$)", path.stem))
        calls = {
            "mu_schedule": "mu_schedule(" in text,
            "mu_barrier": "mu_barrier(" in text,
            "mx_or_gemmini": bool(re.search(
                r"gemmini\.h|mxgemmini|mxgemm|gemmini_\w+|MX_\w+", text)),
            "inline_assembly": bool(re.search(r"\b(?:asm|__asm__)\b", text)),
        }
        record["source_units"].append({
            "path": relative, "sha256": sha256(path),
            "role": "host" if host else "device_or_support",
            "has_main_definition": bool(re.search(r"\b(?:int|void)\s+main\s*\(", text)),
            "language": path.suffix, "features": calls,
            "import_status": "unattempted",
        })
    families = {name: value for name, value in sorted(families.items())
                if value["makefile"] is not None}
    counts = Counter()
    for family in families.values():
        family["build_targets"] = make_targets(root / "kernels" / family["name"])
        if "error" in family["build_targets"]:
            counts["make_query_failures"] += 1
        else:
            counts["default_radiance_elfs"] += len(family["build_targets"]["radiance_elfs"])
            counts["default_variant_targets"] += len(family["build_targets"]["variants"])
        counts["source_units"] += len(family["source_units"])
        for unit in family["source_units"]:
            counts[unit["role"]] += 1
            if unit["has_main_definition"] and unit["role"] != "host":
                counts["direct_device_entry_candidates"] += 1
            if unit["features"]["mx_or_gemmini"]:
                counts["mx_referencing_units"] += 1
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                            capture_output=True, text=True, check=True).stdout
    doc = {"schema": "muon_mlir_kernel_inventory.v2",
           "source_root": str(root), "source_git_revision": revision,
           "source_git_clean": not bool(status.strip()),
           "family_count": len(families), "counts": dict(counts),
           "families": list(families.values())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"family_count": doc["family_count"], "counts": doc["counts"],
                      "out": str(args.out.resolve())}))


if __name__ == "__main__":
    main()

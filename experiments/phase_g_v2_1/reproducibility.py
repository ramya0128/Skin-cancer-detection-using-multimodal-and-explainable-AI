from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BEFORE_MANIFEST = (
    ROOT
    / "outputs"
    / "synthetic_phase_g_v2_1"
    / "v1_v2_protected_sha256_before.json"
)


def verify_protected_artifacts(
    root: Path | None = None,
) -> dict[str, Any]:
    root = Path(root or ROOT).resolve()
    manifest_path = root / BEFORE_MANIFEST.relative_to(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    observed = {}
    for relative, expected in manifest["sha256"].items():
        path = root / Path(relative)
        if not path.is_file():
            mismatches.append(relative)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        observed[relative] = actual
        if actual != expected:
            mismatches.append(relative)
    return {
        "unchanged": not mismatches,
        "file_count": len(observed),
        "mismatches": mismatches,
    }

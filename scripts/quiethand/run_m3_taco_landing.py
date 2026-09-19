#!/usr/bin/env python3
"""Land only the frozen QuietHand M3 TACO members from remote ZIP archives."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.taco_landing import (  # noqa: E402
    TacoLandingError,
    land_probe_members,
)


PROBE = ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_TACO_PROBE.json"
DATA_ROOT = ROOT / "external_data" / "taco_v1"
RECEIPT = (
    ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_TACO_LANDING_RECEIPT.json"
)


def main() -> int:
    try:
        receipt = land_probe_members(
            probe_path=PROBE,
            data_root=DATA_ROOT,
            receipt_path=RECEIPT,
        )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        TacoLandingError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"[hold] HOLD_DATA_ACCESS: {exc}", file=sys.stderr, flush=True)
        return 3
    print(
        f"[ready] {receipt['completed_file_count']} files, "
        f"{int(receipt['completed_uncompressed_bytes']) / 1024**3:.2f} GiB",
        flush=True,
    )
    print(f"[artifact] {RECEIPT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

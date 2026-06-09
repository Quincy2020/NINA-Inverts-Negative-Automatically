from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qnegative.core.dust_removal import linear_to_srgb_float, predict_dust_mask
from qnegative.core.models import DustRemovalParams


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated dust-mask inference worker.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.params.open("r", encoding="utf-8") as handle:
        params_payload = json.load(handle)
    params = DustRemovalParams(**params_payload)

    data = np.load(args.input)
    linear_rgb = np.ascontiguousarray(data["linear_rgb"].astype(np.float32, copy=False))

    def progress(value: int, text: str) -> None:
        print(
            json.dumps(
                {
                    "type": "progress",
                    "value": int(value),
                    "text": str(text),
                }
            ),
            flush=True,
        )

    progress(3, "Preparing image")
    srgb = linear_to_srgb_float(linear_rgb)
    mask, stats = predict_dust_mask(srgb, params, model_root=Path.cwd(), progress_callback=progress)
    np.savez_compressed(
        args.output,
        mask=mask.astype(np.uint8),
        stats=np.array(json.dumps(stats), dtype=object),
    )
    print(json.dumps({"type": "finished"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

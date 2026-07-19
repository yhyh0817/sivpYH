"""Range-download a small, source-diverse subset of the VidLLVIP archive."""

from __future__ import annotations

import time
from pathlib import Path

from remotezip import RemoteIOError, RemoteZip


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/raw/VidLLVIP_subset"
URL = (
    "https://hf-mirror.com/datasets/jianfeng0369/VidLLVIP/resolve/main/"
    "dataset/dataset.zip?download=true"
)
SOURCE_IDS = tuple(range(1, 15))


def main() -> None:
    selected = [
        f"dataset/{modality}/{source_id:02d}_0000_0005.mp4"
        for source_id in SOURCE_IDS
        for modality in ("vi", "ir")
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 9):
        try:
            with RemoteZip(URL) as archive:
                metadata = {item.filename: item for item in archive.infolist()}
                missing = [name for name in selected if name not in metadata]
                if missing:
                    raise FileNotFoundError(f"archive entries missing: {missing}")
                for index, name in enumerate(selected, start=1):
                    destination = OUTPUT / Path(name).relative_to("dataset")
                    expected = metadata[name].file_size
                    if destination.exists() and destination.stat().st_size == expected:
                        print(f"[{index}/{len(selected)}] cached {destination.name}")
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    temporary.write_bytes(archive.read(name))
                    if temporary.stat().st_size != expected:
                        raise IOError(f"incomplete archive member: {name}")
                    temporary.replace(destination)
                    print(f"[{index}/{len(selected)}] downloaded {destination.name}")
            return
        except RemoteIOError as error:
            if attempt == 8:
                raise
            print(f"remote session failed ({attempt}/8): {error}")
            time.sleep(3 * attempt)


if __name__ == "__main__":
    main()

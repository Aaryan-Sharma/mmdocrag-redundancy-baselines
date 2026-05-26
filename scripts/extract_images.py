"""
One-time extraction of images.zip from the MMDocRAG dataset.

Usage:
    python scripts/extract_images.py --zip-path /data/images.zip --out-dir /data

After extraction, pass --out-dir as --images-dir to run_pipeline.py.
Expected structure: <out-dir>/images/*.jpg (~14 826 files).
"""
import argparse
import zipfile
from pathlib import Path


EXPECTED_MIN = 14_000
EXPECTED_MAX = 15_000


def count_jpegs(directory: Path) -> int:
    # Exclude __MACOSX resource-fork artifacts that macOS zips include
    return sum(
        1 for p in directory.rglob("*.jpg")
        if "__MACOSX" not in p.parts
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    images_dir = args.out_dir / "images"

    # Idempotency check
    if images_dir.exists():
        n = count_jpegs(images_dir)
        if EXPECTED_MIN <= n <= EXPECTED_MAX:
            print(f"Already extracted: {n} JPEGs found in {images_dir}. Skipping.")
            return
        else:
            print(f"Found {n} JPEGs in {images_dir} — outside expected range "
                  f"[{EXPECTED_MIN}, {EXPECTED_MAX}]. Re-extracting.")

    if not args.zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {args.zip_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {args.zip_path} -> {args.out_dir} ...")

    with zipfile.ZipFile(args.zip_path, "r") as zf:
        zf.extractall(args.out_dir)

    n = count_jpegs(args.out_dir)
    print(f"Extraction complete: {n} JPEGs found.")

    if not (EXPECTED_MIN <= n <= EXPECTED_MAX):
        print(f"WARN: expected {EXPECTED_MIN}–{EXPECTED_MAX} JPEGs, got {n}. "
              "Check the zip integrity.")
    else:
        print("Count OK.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert a single PE or ELF binary file to a 128×128 grayscale PNG.

Usage:
    python scripts/convert_binary.py --input FILE [--output FILE.png] [--size INT]

Arguments:
    --input   PATH   Path to input binary file (.exe, .dll, ELF)  [required]
    --output  PATH   Path for output PNG file                      [default: <input>.png]
    --size    INT    Output image size (square)                     [default: 128]

Exit codes:
    0  success
    1  input file not found
    2  invalid binary format (not PE or ELF)
    3  conversion or save error

No ML dependencies. This script works without PyTorch or scikit-learn installed.
"""
import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PE/ELF binary to a grayscale PNG image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input',  type=str, required=True,
                        help='Path to input binary file (.exe, .dll, or ELF)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path for output PNG (default: <input_name>.png)')
    parser.add_argument('--size',   type=int, default=128,
                        help='Output image size in pixels (NxN square)')
    return parser.parse_args()


def main():
    args = parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix('.png')
    img_size    = args.size

    # ── 1. Verify input file exists ────────────────────────────────────────────
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not input_path.is_file():
        print(f"ERROR: Input path is not a file: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ── 2. Read raw bytes ──────────────────────────────────────────────────────
    print(f"Reading: {input_path}  ({input_path.stat().st_size:,} bytes)")
    file_bytes = input_path.read_bytes()

    # ── 3. Validate binary format ──────────────────────────────────────────────
    from modules.binary_to_image.utils import validate_binary_format, compute_sha256, get_file_metadata
    try:
        file_format = validate_binary_format(file_bytes)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Format:  {file_format}")

    # ── 4. Compute SHA-256 ─────────────────────────────────────────────────────
    sha256 = compute_sha256(file_bytes)
    print(f"SHA-256: {sha256}")

    # ── 5. Print full metadata ─────────────────────────────────────────────────
    meta = get_file_metadata(file_bytes, input_path.name, file_format)
    print(f"Size:    {meta['size_human']}")
    print(f"Time:    {meta['upload_time']}")

    # ── 6. Convert bytes → grayscale array ────────────────────────────────────
    from modules.binary_to_image.converter import BinaryConverter
    try:
        converter  = BinaryConverter(img_size=img_size)
        img_array  = converter.convert(file_bytes)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"ERROR during conversion: {e}", file=sys.stderr)
        sys.exit(3)

    # ── 7. Save PNG to disk ────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        converter.save(img_array, output_path)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"\nSaved {img_size}x{img_size} grayscale PNG to {output_path}")
    sys.exit(0)


if __name__ == '__main__':
    main()

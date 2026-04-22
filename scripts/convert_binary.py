"""
CLI utility: convert a single binary file to a 128x128 grayscale PNG.
No ML dependencies.

Usage:
    python scripts/convert_binary.py --input path/to/file.exe --output path/to/out.png

Exit codes:
    0 — success
    1 — file not found or format error
    2 — conversion error
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running this script directly
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MalTwin: Convert binary file to grayscale PNG"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Path to input PE or ELF binary file",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        type=Path,
        help="Path to output PNG file (will be created)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Output image size in pixels (default: 128)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Read bytes
    file_bytes = args.input.read_bytes()
    print(f"Read {len(file_bytes):,} bytes from {args.input.name}")

    # Validate format
    from modules.binary_to_image.utils import (
        validate_binary_format,
        compute_sha256,
        get_file_metadata,
    )
    try:
        file_format = validate_binary_format(file_bytes)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Detected format : {file_format}")

    # Compute metadata
    sha256 = compute_sha256(file_bytes)
    meta   = get_file_metadata(file_bytes, args.input.name, file_format)
    print(f"File size       : {meta['size_human']}")
    print(f"SHA-256         : {sha256}")

    # Convert
    from modules.binary_to_image.converter import BinaryConverter
    try:
        converter = BinaryConverter(img_size=args.size)
        img_array = converter.convert(file_bytes)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        converter.save(img_array, args.output)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Saved {args.size}x{args.size} grayscale PNG → {args.output}")


if __name__ == "__main__":
    main()

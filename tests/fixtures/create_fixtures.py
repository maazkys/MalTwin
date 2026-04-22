"""
Script to generate minimal binary fixture files for tests.
Run once: python tests/fixtures/create_fixtures.py

Creates:
    tests/fixtures/sample_pe.exe   — 1024-byte minimal PE binary
    tests/fixtures/sample.elf      — 1024-byte minimal ELF binary
"""
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent


def create_minimal_pe(output_path: Path) -> None:
    """Write a 1024-byte minimal PE binary (MZ magic + padding)."""
    header = b'MZ' + b'\x90' * 58
    body   = b'\x00' * (1024 - len(header))
    output_path.write_bytes(header + body)
    print(f"Created: {output_path} ({output_path.stat().st_size} bytes)")


def create_minimal_elf(output_path: Path) -> None:
    """Write a 1024-byte minimal ELF binary (ELF magic + padding)."""
    header = b'\x7fELF' + b'\x00' * 56
    body   = b'\x00' * (1024 - len(header))
    output_path.write_bytes(header + body)
    print(f"Created: {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    create_minimal_pe(FIXTURES_DIR / "sample_pe.exe")
    create_minimal_elf(FIXTURES_DIR / "sample.elf")
    print("Done.")

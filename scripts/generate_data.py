import os
import random
import argparse
import numpy as np
from pathlib import Path
import config  # This will now find DATA_DIR

def generate_synthetic_binary(file_path, file_type='elf', size_kb=128):
    """
    Generates a dummy binary with valid headers and structural patterns.
    """
    size_bytes = size_kb * 1024
    
    # 1. Generate Header
    if file_type == 'pe':
        # Simple MZ Header (DOS stub)
        header = bytearray(b'MZ' + b'\x90' * 62) 
    else:
        # Simple ELF Header
        header = bytearray(b'\x7fELF' + b'\x01\x01\x01\x00' * 7)

    # 2. Generate "Structural" Body
    body = bytearray()
    
    # Section 1: Low-entropy "Code" (repetitive instructions)
    code_size = int(size_bytes * 0.4)
    body.extend(random.choice([b'\x90', b'\x00', b'\xCC']) * code_size)
    
    # Section 2: Patterned "Data" (simulating strings or tables)
    data_size = int(size_bytes * 0.3)
    pattern = np.random.randint(0, 256, 16, dtype=np.uint8).tobytes()
    body.extend(pattern * (data_size // 16))
    
    # Section 3: High-entropy "Packed" content (random noise)
    remaining = size_bytes - len(header) - len(body)
    if remaining > 0:
        body.extend(os.urandom(remaining))

    # Save to disk
    with open(file_path, 'wb') as f:
        f.write(header + body)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic IIoT binaries for MalTwin testing")
    parser.add_argument("--families", nargs="+", default=["Mirai_Synthetic", "Gafgyt_Synthetic", "Benign_IoT"],
                        help="List of folder names (classes) to create")
    parser.add_argument("--count", type=int, default=50, help="Files per family")
    parser.add_argument("--size-min", type=int, default=64, help="Min file size in KB")
    parser.add_argument("--size-max", type=int, default=256, help="Max file size in KB")
    
    args = parser.parse_args()
    
    # Changed from config.MALTWIN_DATA_DIR to config.DATA_DIR
    base_dir = Path(config.DATA_DIR)
    
    print(f"🚀 Starting data generation in: {base_dir}")
    
    for family in args.families:
        family_path = base_dir / family
        family_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Generating {args.count} samples for family: {family}...")
        
        for i in range(args.count):
            # Use elf for malware, exe for "benign" to test variety
            file_ext = "elf" if "Benign" not in family else "exe"
            file_name = f"sample_{i:03d}.{file_ext}"
            file_path = family_path / file_name
            
            size = random.randint(args.size_min, args.size_max)
            generate_synthetic_binary(file_path, file_type=file_ext, size_kb=size)
            
    print("\n✅ Generation complete. You can now run your conversion script.")

if __name__ == "__main__":
    main()
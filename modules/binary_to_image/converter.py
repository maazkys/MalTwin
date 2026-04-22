"""
BinaryConverter: converts raw PE/ELF bytes to a 128x128 grayscale image.
Algorithm from Nataraj et al. (2011) — byte array reshaped to 2D, then resized.
No ML dependencies.
"""
import math
from pathlib import Path

import cv2
import numpy as np

import config


class BinaryConverter:
    """
    Convert raw binary file bytes into a standardised grayscale PNG image.

    Conversion algorithm:
        1. Read file bytes as flat uint8 numpy array via np.frombuffer.
        2. Determine 2D width:  width = max(1, int(math.sqrt(len(byte_array))))
        3. Determine rows:      rows  = len(byte_array) // width
        4. Trim array to exact (rows * width) length (discard tail bytes).
        5. Reshape to (rows, width) 2D array.
        6. Resize to (img_size, img_size) using cv2.INTER_LINEAR (bilinear).
        7. Cast result to uint8 and return.

    Constructor args:
        img_size (int): side length of output square image in pixels.
                        Must be > 0. Default: config.IMG_SIZE (128).

    Raises on construction:
        ValueError: "img_size must be a positive integer, got {img_size}"
            if img_size <= 0.
    """

    def __init__(self, img_size: int = config.IMG_SIZE) -> None:
        if img_size <= 0:
            raise ValueError(f"img_size must be a positive integer, got {img_size}")
        self.img_size = img_size

    def convert(self, file_bytes: bytes) -> np.ndarray:
        """
        Convert raw binary bytes to a grayscale image array.

        Args:
            file_bytes: raw bytes of a PE or ELF binary.
                        Caller is responsible for prior format validation.

        Returns:
            numpy.ndarray of shape (img_size, img_size), dtype=uint8, values 0–255.

        Raises:
            ValueError: "Binary file is empty or too small to convert (minimum 64 bytes)"
                if len(file_bytes) < 64.
        """
        if len(file_bytes) < 64:
            raise ValueError(
                "Binary file is empty or too small to convert (minimum 64 bytes)"
            )
        byte_array = np.frombuffer(file_bytes, dtype=np.uint8)
        n = len(byte_array)
        width = max(1, int(math.sqrt(n)))
        rows  = n // width
        trimmed   = byte_array[:rows * width]
        reshaped  = trimmed.reshape((rows, width))
        resized   = cv2.resize(
            reshaped,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized.astype(np.uint8)

    def to_png_bytes(self, img_array: np.ndarray) -> bytes:
        """
        Encode a grayscale numpy array to PNG bytes for in-memory use.

        Args:
            img_array: numpy array of shape (H, W), dtype uint8.

        Returns:
            PNG-encoded bytes. First 4 bytes will be b'\\x89PNG' (PNG magic).

        Raises:
            RuntimeError: "cv2.imencode failed to encode image as PNG"
                if cv2.imencode returns success=False.
        """
        success, encoded = cv2.imencode('.png', img_array)
        if not success:
            raise RuntimeError("cv2.imencode failed to encode image as PNG")
        return encoded.tobytes()

    def to_pil_image(self, img_array: np.ndarray):
        """
        Convert grayscale array to PIL Image for torchvision transform compatibility.

        Args:
            img_array: numpy array of shape (H, W), dtype uint8.

        Returns:
            PIL.Image.Image in mode 'L' (8-bit grayscale).
        """
        from PIL import Image
        return Image.fromarray(img_array, mode='L')

    def save(self, img_array: np.ndarray, output_path: Path) -> None:
        """
        Save grayscale array as PNG file to disk.

        Args:
            img_array:   numpy array of shape (H, W), dtype uint8.
            output_path: Path where PNG will be written.
                         Parent directory must already exist.

        Raises:
            RuntimeError: "Failed to save image to {output_path}"
                if cv2.imwrite returns False.
        """
        success = cv2.imwrite(str(output_path), img_array)
        if not success:
            raise RuntimeError(f"Failed to save image to {output_path}")

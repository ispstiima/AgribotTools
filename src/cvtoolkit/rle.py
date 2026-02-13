"""
Run-Length Encoding (RLE) utilities for mask conversion.

Provides encoding and decoding functions for converting between
binary masks and Label Studio's RLE format.
"""

import cv2
import numpy as np


class InputStream:
    """A simple input stream for reading bits from binary data."""

    def __init__(self, data: str):
        """Initialize with binary data string (0s and 1s)."""
        self.data = data
        self.i = 0

    def read(self, size: int) -> int:
        """Read specified number of bits and convert to integer."""
        out = self.data[self.i: self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data: bytes, num: int) -> int:
    """Access a specific bit from bytes array."""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bits2byte(arr_str: str, n: int = 8) -> list:
    """Convert bits string back to bytes list."""
    rle = []
    numbers = [arr_str[i: i + n] for i in range(0, len(arr_str), n)]
    for i in numbers:
        rle.append(int(i, 2))
    return rle


def bytes2bit(data: bytes) -> str:
    """Convert bytes data to bit string."""
    return "".join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def base_rle_encode(inarray: np.ndarray) -> tuple:
    """
    Run length encoding for input array.
    
    Returns:
        Tuple of (runlengths, startpositions, values)
    """
    ia = np.asarray(inarray)
    n = len(ia)

    if n == 0:
        return None, None, None
    else:
        y = ia[1:] != ia[:-1]
        i = np.append(np.where(y), n - 1)
        z = np.diff(np.append(-1, i))
        p = np.cumsum(np.append(0, z))[:-1]
        return z, p, ia[i]


def encode_rle(arr: np.ndarray, wordsize: int = 8, rle_sizes: list = None) -> list:
    """
    Encode a 1D array to run length encoding (RLE).
    
    Args:
        arr: Flattened numpy array from a 4D image (R, G, B, alpha)
        wordsize: Wordsize bits for decoding
        rle_sizes: List of integers for run lengths
    
    Returns:
        Run length encoded array
    """
    if rle_sizes is None:
        rle_sizes = [3, 4, 8, 16]
    
    num = len(arr)
    numbits = f"{num:032b}"

    wordsizebits = f"{wordsize - 1:05b}"
    rle_bits = "".join([f"{x - 1:04b}" for x in rle_sizes])
    base_str = numbits + wordsizebits + rle_bits
    out_str = ""
    
    for length_reeks, p, value in zip(*base_rle_encode(arr)):
        if length_reeks == 1:
            out_str += "0"
            out_str += "00"
            out_str += "000"
            out_str += f"{value:08b}"

        elif length_reeks > 1:
            if length_reeks <= 8:
                out_str += "1"
                out_str += "00"
                out_str += f"{length_reeks - 1:03b}"
                out_str += f"{value:08b}"

            elif 8 < length_reeks <= 16:
                out_str += "1"
                out_str += "01"
                out_str += f"{length_reeks - 1:04b}"
                out_str += f"{value:08b}"

            elif 16 < length_reeks <= 256:
                out_str += "1"
                out_str += "10"
                out_str += f"{length_reeks - 1:08b}"
                out_str += f"{value:08b}"

            else:
                length_temp = length_reeks
                while length_temp > 2 ** 16:
                    out_str += "1"
                    out_str += "11"
                    out_str += f"{2 ** 16 - 1:016b}"
                    out_str += f"{value:08b}"
                    length_temp -= 2 ** 16

                out_str += "1"
                out_str += "11"
                out_str += f"{length_temp - 1:016b}"
                out_str += f"{value:08b}"

    nzfill = 8 - len(base_str + out_str) % 8
    total_str = base_str + out_str
    total_str = total_str + nzfill * "0"

    rle = bits2byte(total_str)

    return rle


def decode_rle(rle: list) -> np.ndarray:
    """
    Decode an RLE-encoded list to a flattened numpy uint8 image.
    
    Args:
        rle: RLE-encoded list of integers
    
    Returns:
        Flattened numpy uint8 image [width, height, channel]
    """
    input_ = InputStream(bytes2bit(rle))
    num = input_.read(32)
    word_size = input_.read(5) + 1
    rle_sizes = [input_.read(4) + 1 for _ in range(4)]

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = input_.read(1)
        j = i + 1 + input_.read(rle_sizes[input_.read(2)])
        if x:
            val = input_.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = input_.read(word_size)
                out[i] = val
                i += 1
    return out


def mask_to_rle(mask: np.ndarray) -> list:
    """
    Convert mask to RLE format.
    
    Args:
        mask: uint8 or int numpy array mask with len(shape) == 2 (grayscale)
    
    Returns:
        List of integers in RLE format
    """
    assert len(mask.shape) == 2, "mask must be 2D np.array"
    assert mask.dtype == np.uint8 or mask.dtype == int, "mask must be uint8 or int"
    array = mask.ravel()
    array = np.repeat(array, 4)
    rle = encode_rle(array)
    return rle


def yolo_to_mask(contour: list, img_width: int, img_height: int) -> np.ndarray:
    """
    Convert a YOLO segmentation mask (polygon format) into a uint8 2D mask.
    
    Args:
        contour: List of normalized polygon points [x1, y1, x2, y2, ..., xn, yn]
        img_width: Original image width
        img_height: Original image height
    
    Returns:
        2D numpy array (uint8) representing the segmentation mask
    """
    polygon = np.array(contour, dtype=np.float32).reshape(-1, 2)
    polygon[:, 0] *= img_width
    polygon[:, 1] *= img_height
    polygon = polygon.astype(np.int32)

    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.fillPoly(img=mask, pts=[polygon], color=[255])

    return mask

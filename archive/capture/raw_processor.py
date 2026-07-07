"""
RAW Bayer processing: unpack, crop, channel split.

Handles OV13855 SBGGR10P (10-bit packed, BGGR pattern).
4224×3136 full-res → center crop → 1920×1080 → RGGB 4-plane.

Pure Python implementation — no numpy/opencv required for board deployment.
Uses Python's built-in `array` and `struct` for performance.
"""

import struct


# ── constants ────────────────────────────────────────────────
OV13855_BLACK_LEVEL = 64
OV13855_MAX_VAL = 1023  # 10-bit


def unpack_sbggr10p(raw_bytes, width, height, black_level=64):
    """Unpack V4L2 SBGGR10P → 10-bit Bayer numpy-style array.

    Format: 5 bytes = 4 pixels (p0[7:0], p1[7:0], p2[7:0], p3[7:0], {p3_low, p2_low, p1_low, p0_low})

    Returns: list of ints, length = width * height
    """
    total_pixels = width * height
    num_groups = total_pixels // 4
    expected_bytes = num_groups * 5

    if len(raw_bytes) < expected_bytes:
        raw_bytes = raw_bytes + b'\x00' * (expected_bytes - len(raw_bytes))

    # Use struct to unpack 5 bytes at a time
    # Pre-allocate result
    result = [0] * total_pixels

    # Process in chunks for speed
    for i in range(num_groups):
        base = i * 5
        b0 = raw_bytes[base]
        b1 = raw_bytes[base + 1]
        b2 = raw_bytes[base + 2]
        b3 = raw_bytes[base + 3]
        b4 = raw_bytes[base + 4]

        pi = i * 4
        result[pi]     = ((b0 << 2) | (b4 & 0x03))      # pixel 0
        result[pi + 1] = ((b1 << 2) | ((b4 >> 2) & 0x03))  # pixel 1
        result[pi + 2] = ((b2 << 2) | ((b4 >> 4) & 0x03))  # pixel 2
        result[pi + 3] = ((b3 << 2) | ((b4 >> 6) & 0x03))  # pixel 3

    # Subtract black level and clamp
    bl = black_level
    mx = 1023.0 - bl
    result = [min(max((v - bl) / mx, 0.0), 1.0) for v in result]

    return result


def bayer_crop(bayer_1d, src_w, src_h, dst_w, dst_h, offset_x=None, offset_y=None):
    """Crop a Bayer raster (1D float list) to dst_w × dst_h.

    Coordinates must be even-aligned to preserve Bayer pattern.
    If offset_x/offset_y are None, center-crop is used.
    """
    if offset_x is None:
        offset_x = (src_w - dst_w) // 2
    if offset_y is None:
        offset_y = (src_h - dst_h) // 2

    # Ensure even alignment
    offset_x = (offset_x // 2) * 2
    offset_y = (offset_y // 2) * 2

    result = [0.0] * (dst_w * dst_h)
    for row in range(dst_h):
        src_start = (offset_y + row) * src_w + offset_x
        dst_start = row * dst_w
        result[dst_start:dst_start + dst_w] = bayer_1d[src_start:src_start + dst_w]

    return result


def bayer_to_rggb_planes(bayer_1d, width, height, pattern='BGGR'):
    """Split Bayer raster → 4-channel RGGB planar.

    OV13855 BGGR:
        Row0: B  Gb B  Gb ...
        Row1: Gr R  Gr R  ...
        Row2: B  Gb B  Gb ...

    Returns: tuple of 4 lists (R, Gr, Gb, B), each length = (W/2) * (H/2)
    """
    hw = width // 2
    hh = height // 2
    plane_size = hw * hh

    R  = [0.0] * plane_size
    Gr = [0.0] * plane_size
    Gb = [0.0] * plane_size
    B  = [0.0] * plane_size

    # Determine which channel gets which position
    if pattern == 'BGGR':
        # Row 0: B(even col), Gb(odd col)
        # Row 1: Gr(even col), R(odd col)
        for row_h in range(hh):
            for col_h in range(hw):
                pi = row_h * hw + col_h
                src_row0 = row_h * 2
                src_row1 = row_h * 2 + 1
                src_col0 = col_h * 2
                src_col1 = col_h * 2 + 1

                B[pi]  = bayer_1d[src_row0 * width + src_col0]
                Gb[pi] = bayer_1d[src_row0 * width + src_col1]
                Gr[pi] = bayer_1d[src_row1 * width + src_col0]
                R[pi]  = bayer_1d[src_row1 * width + src_col1]
    elif pattern == 'RGGB':
        for row_h in range(hh):
            for col_h in range(hw):
                pi = row_h * hw + col_h
                src_row0 = row_h * 2
                src_row1 = row_h * 2 + 1
                src_col0 = col_h * 2
                src_col1 = col_h * 2 + 1

                R[pi]  = bayer_1d[src_row0 * width + src_col0]
                Gr[pi] = bayer_1d[src_row0 * width + src_col1]
                Gb[pi] = bayer_1d[src_row1 * width + src_col0]
                B[pi]  = bayer_1d[src_row1 * width + src_col1]
    else:
        raise ValueError("Unsupported Bayer pattern: %s" % pattern)

    return R, Gr, Gb, B


def write_raw_10bit(planes_rggb, filepath):
    """Write 4-plane RGGB float data to disk as 16-bit packed raw.
    Each value is float [0,1], scaled to 10-bit [0,1023].

    Format: short1.raw — concatenated R(10bit) + Gr(10bit) + Gb(10bit) + B(10bit)
    Each pixel stored as uint16, 10-bit value in lower bits.
    """
    import struct
    all_data = []
    for plane in planes_rggb:
        for v in plane:
            all_data.append(int(min(max(v, 0.0), 1.0) * 1023))
    with open(filepath, 'wb') as f:
        f.write(struct.pack('<%dH' % len(all_data), *all_data))


def write_raw_npy(planes_rggb, filepath):
    """Write 4-plane RGGB as .npy file (requires numpy).
    Faster than raw for training, compatible with IMX585 dataset format.
    """
    import numpy as np
    R, Gr, Gb, B = planes_rggb
    h = len(R)
    w_per_plane = 1  # just a marker; planes are 1D
    arr = np.array([R, Gr, Gb, B], dtype=np.float32)
    np.save(filepath, arr)


class TripletSaver:
    """Manages the ring-buffer logic for S-L-S triplet assembly.

    Usage:
        saver = TripletSaver(output_dir='/data/train')
        saver.feed(frame_bytes, exposure_type)  # 'S' or 'L'
        # Automatically saves when a complete triplet is formed
    """

    def __init__(self, output_dir, raw_width=4224, raw_height=3136,
                 crop_width=1920, crop_height=1080, bayer_pattern='BGGR',
                 save_format='raw'):
        self.output_dir = output_dir
        self.raw_w = raw_width
        self.raw_h = raw_height
        self.crop_w = crop_width
        self.crop_h = crop_height
        self.pattern = bayer_pattern
        self.save_format = save_format
        self.scene_idx = 0
        self.triplet_count = 0

        # Ring buffer for the sliding window
        self.ring = [None, None, None]  # [frame0, frame1, frame2]
        self.ring_types = [None, None, None]  # 'S' or 'L'
        self.ring_pos = 0

        import os
        os.makedirs(output_dir, exist_ok=True)

    def _process_frame(self, raw_bytes):
        """Full pipeline: unpack → crop → RGGB planes."""
        bayer = unpack_sbggr10p(raw_bytes, self.raw_w, self.raw_h,
                                black_level=OV13855_BLACK_LEVEL)
        bayer = bayer_crop(bayer, self.raw_w, self.raw_h,
                           self.crop_w, self.crop_h)
        planes = bayer_to_rggb_planes(bayer, self.crop_w, self.crop_h,
                                      pattern=self.pattern)
        return planes

    def feed(self, raw_bytes, exp_type):
        """Feed one frame (raw bytes) with exposure type 'S' or 'L'.

        When a new S frame completes a S-L-S triplet, saves to disk.
        """
        self.ring[self.ring_pos] = raw_bytes
        self.ring_types[self.ring_pos] = exp_type
        self.ring_pos = (self.ring_pos + 1) % 3

        # Check if we have a complete triplet: S, L, S
        # The triplet ends when ring has: ..., S, L, S (current is S)
        if exp_type == 'S' and self.ring_pos == 0:
            # Just wrapped around, check pattern
            if self.ring_types == ['S', 'L', 'S']:
                self._save_triplet()

        if exp_type == 'S' and self.ring_pos == 2:
            # Middle of ring, check if prev two were S, L
            if (self.ring_types[0] == 'S' and self.ring_types[1] == 'L'):
                # Current is S at pos 2 → triplet is [0]=S, [1]=L, [2]=S
                self._save_triplet()

    def _save_triplet(self):
        """Process and save the current triplet to disk."""
        scene_name = 'scene_%05d' % self.scene_idx
        scene_dir = os.path.join(self.output_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        # ring contains S1, L, S2 in order
        print('  Saving triplet %d → %s/' % (self.triplet_count, scene_name))

        s1_planes = self._process_frame(self.ring[0])
        l_planes  = self._process_frame(self.ring[1])
        s2_planes = self._process_frame(self.ring[2])

        if self.save_format == 'raw':
            write_raw_10bit(s1_planes, os.path.join(scene_dir, 'short1.raw'))
            write_raw_10bit(l_planes,  os.path.join(scene_dir, 'long.raw'))
            write_raw_10bit(s2_planes, os.path.join(scene_dir, 'short2.raw'))
        elif self.save_format == 'npy':
            write_raw_npy(s1_planes, os.path.join(scene_dir, 'short1.npy'))
            write_raw_npy(l_planes,  os.path.join(scene_dir, 'long.npy'))
            write_raw_npy(s2_planes, os.path.join(scene_dir, 'short2.npy'))

        self.scene_idx += 1
        self.triplet_count += 1
        print('  Total triplets: %d' % self.triplet_count)

    def flush(self):
        """Save any remaining triplet if buffer has data."""
        pass  # With sliding window there's no leftover


import os as _os

"""
V4L2 MMAP capture module for RK3588 ELF2.

Uses Python ctypes to call V4L2 ioctls directly — no subprocess overhead.
Handles multi-planar API (VIDIOC_*_MPLANE) required by RK3588 CIF driver.

Usage:
    cam = V4L2Capture('/dev/video0')
    cam.start()
    buf = cam.dequeue()      # returns bytes
    cam.stop()
"""

import os
import mmap
import ctypes
import fcntl
from ctypes import Structure, c_uint32, c_uint64, c_int32, c_void_p, sizeof

# ── V4L2 constants ───────────────────────────────────────────
_IOC_NRBITS   = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS  = 2
_IOC_NRSHIFT  = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT  = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_WRITE = 1
_IOC_READ  = 2

def _IOC(dir_, type_, nr, size):
    return ctypes.c_int32(
        (dir_  << _IOC_DIRSHIFT) |
        (ord(type_) << _IOC_TYPESHIFT) |
        (nr    << _IOC_NRSHIFT) |
        (size  << _IOC_SIZESHIFT)
    ).value

def _IOWR(type_, nr, size):
    return _IOC(_IOC_READ | _IOC_WRITE, type_, nr, size)

def _IOW(type_, nr, size):
    return _IOC(_IOC_WRITE, type_, nr, size)

def _IOR(type_, nr, size):
    return _IOC(_IOC_READ, type_, nr, size)

VIDEO_MAX_PLANES = 8

V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE = 9
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_NONE = 0

# ── V4L2 structs (multi-planar) ──────────────────────────────

class v4l2_plane(Structure):
    _fields_ = [
        ("bytesused",    c_uint32),
        ("length",       c_uint32),
        ("m_mem_offset", c_uint32),  # union m: mem_offset for MMAP
        ("padding1",     c_uint32),  # pad union to 64 bits
        ("data_offset",  c_uint32),
        ("reserved",     c_uint32 * 11),
    ]

class v4l2_buffer(Structure):
    _fields_ = [
        ("index",        c_uint32),
        ("type",         c_uint32),
        ("bytesused",    c_uint32),
        ("flags",        c_uint32),
        ("field",        c_uint32),
        ("timestamp",    c_uint64 * 2),
        ("timecode",     c_uint32 * 8),
        ("sequence",     c_uint32),
        ("memory",       c_uint32),
        ("m_planes",     v4l2_plane * VIDEO_MAX_PLANES),
        ("length",       c_uint32),
        ("reserved2",    c_uint32),
        ("request_fd",   c_int32),
    ]

class v4l2_requestbuffers(Structure):
    _fields_ = [
        ("count",        c_uint32),
        ("type",         c_uint32),
        ("memory",       c_uint32),
        ("capabilities", c_uint32),
        ("reserved",     c_uint32 * 2),
    ]

# Real ioctl numbers from linux/videodev2.h
VIDIOC_REQBUFS  = _IOWR('V', 8,  sizeof(v4l2_requestbuffers))
VIDIOC_QUERYBUF = _IOWR('V', 9,  sizeof(v4l2_buffer))
VIDIOC_QBUF     = _IOWR('V', 15, sizeof(v4l2_buffer))
VIDIOC_DQBUF    = _IOWR('V', 17, sizeof(v4l2_buffer))
VIDIOC_STREAMON = _IOW('V', 18, sizeof(c_int32))
VIDIOC_STREAMOFF = _IOW('V', 19, sizeof(c_int32))


class V4L2Capture:
    """Direct V4L2 MMAP capture for RK3588 (multi-planar API).

    Usage:
        cam = V4L2Capture('/dev/video0')
        cam.start()
        raw_bytes = cam.dequeue()
        cam.stop()
    """

    def __init__(self, device='/dev/video0', num_buffers=3):
        self.device = device
        self.num_buffers = num_buffers
        self.fd = -1
        self._buffers = []   # list of (mmap_obj, data_offset, buf_len)

    def open(self):
        self.fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK)
        if self.fd < 0:
            raise OSError("Cannot open %s" % self.device)

    def close(self):
        if self.fd >= 0:
            try:
                self._stream_off()
            except Exception:
                pass
            os.close(self.fd)
            self.fd = -1

    def _stream_on(self):
        buf_type = c_int32(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
        ret = fcntl.ioctl(self.fd, VIDIOC_STREAMON, buf_type)
        if ret != 0:
            raise OSError("VIDIOC_STREAMON failed: %d" % ret)

    def _stream_off(self):
        buf_type = c_int32(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
        fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, buf_type)

    def start(self):
        """Request buffers, mmap each, queue all, start streaming."""
        self.open()

        # 1) REQBUFS
        reqbuf = v4l2_requestbuffers()
        reqbuf.count = self.num_buffers
        reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
        reqbuf.memory = V4L2_MEMORY_MMAP

        ret = fcntl.ioctl(self.fd, VIDIOC_REQBUFS, reqbuf)
        if ret != 0:
            raise OSError("VIDIOC_REQBUFS failed: %d (device busy?)" % ret)
        self.num_buffers = reqbuf.count

        # 2) QUERYBUF + mmap + QBUF for each buffer
        self._buffers = []
        pagesize = mmap.PAGESIZE

        for i in range(self.num_buffers):
            buf = v4l2_buffer()
            buf.index = i
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
            buf.memory = V4L2_MEMORY_MMAP

            ret = fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, buf)
            if ret != 0:
                raise OSError("VIDIOC_QUERYBUF[%d] failed: %d" % (i, ret))

            plane0 = buf.m_planes[0]
            buf_len = plane0.length
            mem_offset = plane0.m_mem_offset  # byte offset into device mmap space

            # mmap offset must be page-aligned
            page_offset = (mem_offset // pagesize) * pagesize
            intra_offset = mem_offset - page_offset

            m = mmap.mmap(self.fd, buf_len + intra_offset,
                          mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                          offset=page_offset)
            self._buffers.append((m, intra_offset, buf_len))

            # 3) QBUF
            ret = fcntl.ioctl(self.fd, VIDIOC_QBUF, buf)
            if ret != 0:
                raise OSError("VIDIOC_QBUF[%d] failed: %d" % (i, ret))

        self._stream_on()

    def dequeue(self, timeout_ms=5000):
        """Block until a frame is ready, return raw bytes."""
        import select
        r, _, _ = select.select([self.fd], [], [], timeout_ms / 1000.0)
        if self.fd not in r:
            raise TimeoutError("dequeue timeout after %d ms" % timeout_ms)

        buf = v4l2_buffer()
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
        buf.memory = V4L2_MEMORY_MMAP

        ret = fcntl.ioctl(self.fd, VIDIOC_DQBUF, buf)
        if ret != 0:
            raise OSError("VIDIOC_DQBUF failed: %d" % ret)

        idx = buf.index
        m, intra_off, buf_len = self._buffers[idx]
        used = buf.m_planes[0].bytesused

        # Extract frame data
        data = m[intra_off:intra_off + used]

        # Re-queue
        ret = fcntl.ioctl(self.fd, VIDIOC_QBUF, buf)
        if ret != 0:
            raise OSError("VIDIOC_QBUF[%d] failed: %d" % (idx, ret))

        # Return a copy so the mmap buffer can be reused
        return bytes(data) if isinstance(data, memoryview) else data

    def stop(self):
        if self._buffers:
            self._stream_off()
            for m, _, _ in self._buffers:
                m.close()
            self._buffers.clear()
        self.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ── fallback for boards without Python V4L2 ──

    @staticmethod
    def dequeue_via_v4l2ctl(device='/dev/video0', count=1, output_path=None):
        """Fallback: use v4l2-ctl subprocess to capture frame(s).
        Slower but works on any Buildroot system.
        Returns list of byte strings (one per frame).
        """
        import subprocess
        args = ['v4l2-ctl', '-d', device, '--stream-mmap',
                '--stream-count', str(count)]
        if output_path:
            args += ['--stream-to', output_path]
            subprocess.run(args, check=True, capture_output=True)
            with open(output_path, 'rb') as f:
                return [f.read()]
        else:
            args += ['--stream-to', '-']
            result = subprocess.run(args, check=True, capture_output=True)
            # stdout contains all frames concatenated; split by frame size
            # For now just return raw output
            return [result.stdout]

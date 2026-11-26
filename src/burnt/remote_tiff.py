"""Windowed reads of uncompressed GeoTIFFs stored inside remote tar files (HTTP range requests).

FLOGA-v2 (Hugging Face `orion-ai-lab/FLOGA-GeoTIFFs`) ships 1.1 TB of tar shards for the 20 m grid. The GeoTIFFs
inside are uncompressed, so after indexing the tar members (scripts/hf_tar_index.py) any pixel window can be read
by fetching only its bytes. This avoids downloading the full dataset (see DECISIONS.md).
"""
from __future__ import annotations

import io
import time

import numpy as np
import requests
import tifffile

HF_BASE = "https://huggingface.co/datasets/orion-ai-lab/FLOGA-GeoTIFFs/resolve/main/"
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "ehb466-burnt-area (research; python-requests)"


def http_range(url, start, length, retries=6):
    """Return bytes [start, start+length) of `url`; retries with backoff."""
    for k in range(retries):
        try:
            r = _SESSION.get(url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=120)
            if r.status_code in (200, 206):
                data = r.content
                if r.status_code == 200:  # server ignored range
                    data = data[start:start + length]
                if len(data) == length:
                    return data
            time.sleep(2 ** k)
        except requests.RequestException:
            time.sleep(2 ** k)
    raise IOError(f"range read failed: {url} {start}+{length}")


class RangeFile(io.RawIOBase):
    """Read-only file object over a byte range [base, base+size) of a remote file, with a small block cache."""

    BLOCK = 1 << 16

    def __init__(self, url, base=0, size=None):
        self.url, self.base, self.size, self.pos = url, base, size, 0
        self._cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _block(self, i):
        if i not in self._cache:
            start = i * self.BLOCK
            n = min(self.BLOCK, self.size - start)
            self._cache[i] = http_range(self.url, self.base + start, n)
        return self._cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        if n > 4 * self.BLOCK:  # large read: fetch directly
            data = http_range(self.url, self.base + self.pos, n)
        else:
            out = []
            p, end = self.pos, self.pos + n
            while p < end:
                i = p // self.BLOCK
                b = self._block(i)
                s = p - i * self.BLOCK
                take = min(len(b) - s, end - p)
                out.append(b[s:s + take]); p += take
            data = b"".join(out)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d
        return len(d)


class RemoteTiff:
    """Uncompressed, strip- or tile-organised GeoTIFF at byte offset `base` of `url`."""

    def __init__(self, url, base, size):
        self.url, self.base, self.size = url, base, size
        fh = RangeFile(url, base, size)
        with tifffile.TiffFile(fh) as tf:
            p = tf.pages[0]
            if p.compression != 1:
                raise ValueError(f"compressed TIFF ({p.compression}) not supported")
            self.shape = p.shape
            self.dtype = np.dtype(p.dtype)
            self.planar = int(p.planarconfig)  # 1 contig, 2 separate
            self.samples = p.samplesperpixel
            self.height, self.width = p.imagelength, p.imagewidth
            self.is_tiled = p.is_tiled
            self.tile = (p.tilelength, p.tilewidth) if p.is_tiled else None
            self.rowsperstrip = p.rowsperstrip if not p.is_tiled else None
            self.offsets = np.asarray(p.dataoffsets, np.int64)
            self.bytecounts = np.asarray(p.databytecounts, np.int64)
            tags = {t.name: t.value for t in p.tags.values()}
            self.geo = {k: tags.get(k) for k in ("ModelPixelScaleTag", "ModelTiepointTag", "GeoKeyDirectoryTag")}
            self.nodata = tags.get("GDAL_NODATA")

    def _contiguous(self):
        """True if strip data are stored back to back (then any row range is one byte range)."""
        if len(self.offsets) < 2:
            return True
        return bool(np.all(self.offsets[1:] == self.offsets[:-1] + self.bytecounts[:-1]))

    def read_rows(self, r0, r1):
        """Rows [r0, r1) -> array (bands, rows, width)."""
        r0, r1 = max(0, r0), min(self.height, r1)
        it = self.dtype.itemsize
        if self.is_tiled:
            raise NotImplementedError("tiled layout")
        if self.planar == 1 and self._contiguous():
            row_bytes = self.width * self.samples * it
            start = int(self.offsets[0]) + r0 * row_bytes
            buf = http_range(self.url, self.base + start, (r1 - r0) * row_bytes)
            a = np.frombuffer(buf, self.dtype).reshape(r1 - r0, self.width, self.samples)
            return np.moveaxis(a, -1, 0).copy()
        if self.planar == 2:
            nstrips_band = len(self.offsets) // self.samples
            rps = self.rowsperstrip
            out = np.empty((self.samples, r1 - r0, self.width), self.dtype)
            s0, s1 = r0 // rps, (r1 - 1) // rps
            for b in range(self.samples):
                offs = self.offsets[b * nstrips_band + s0: b * nstrips_band + s1 + 1]
                cnts = self.bytecounts[b * nstrips_band + s0: b * nstrips_band + s1 + 1]
                if np.all(offs[1:] == offs[:-1] + cnts[:-1]):
                    buf = http_range(self.url, self.base + int(offs[0]), int(cnts.sum()))
                else:
                    buf = b"".join(http_range(self.url, self.base + int(o), int(c)) for o, c in zip(offs, cnts))
                a = np.frombuffer(buf, self.dtype).reshape(-1, self.width)
                out[b] = a[r0 - s0 * rps: r1 - s0 * rps]
            return out
        # generic contig, non-contiguous strips
        rps = self.rowsperstrip
        s0, s1 = r0 // rps, (r1 - 1) // rps
        buf = b"".join(http_range(self.url, self.base + int(self.offsets[s]), int(self.bytecounts[s]))
                       for s in range(s0, s1 + 1))
        a = np.frombuffer(buf, self.dtype).reshape(-1, self.width, self.samples)
        return np.moveaxis(a[r0 - s0 * rps: r1 - s0 * rps], -1, 0).copy()

    def read_all(self):
        return self.read_rows(0, self.height)

"""Windowed reads of uncompressed GeoTIFFs stored inside remote tar files (HTTP range requests).

FLOGA-v2 (Hugging Face `orion-ai-lab/FLOGA-GeoTIFFs`) ships 1.1 TB of tar shards for the 20 m grid. The GeoTIFFs
inside are uncompressed, so after indexing the tar members (scripts/hf_tar_index.py) any pixel window can be read
by fetching only its bytes. This avoids downloading the full dataset (see DECISIONS.md).
"""
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import tifffile

HF_BASE = "https://huggingface.co/datasets/orion-ai-lab/FLOGA-GeoTIFFs/resolve/main/"
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "ehb466-burnt-area (research; python-requests)"


_RESOLVED = {}  # hf url -> (cdn url, time resolved)


def resolve(url, force=False):
    """Resolve the Hugging Face redirect once and reuse the signed CDN URL (avoids HTTP 429 on /resolve/)."""
    if "huggingface.co" not in url:
        return url
    hit = _RESOLVED.get(url)
    if hit and not force and time.time() - hit[1] < 1800:
        return hit[0]
    for k in range(8):
        r = _SESSION.head(url, allow_redirects=False, timeout=60)
        if r.status_code in (301, 302, 307, 308):
            _RESOLVED[url] = (r.headers["location"], time.time())
            return _RESOLVED[url][0]
        if r.status_code == 200:
            _RESOLVED[url] = (url, time.time())
            return url
        time.sleep(min(60, 5 * 2 ** k))
    raise IOError(f"cannot resolve {url}: HTTP {r.status_code}")


def http_range(url, start, length, retries=8):
    """Return bytes [start, start+length) of `url`; retries with backoff, re-resolves expired CDN links."""
    for k in range(retries):
        try:
            r = _SESSION.get(resolve(url), headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=120)
            if r.status_code in (403, 410):
                resolve(url, force=True)
            if r.status_code == 429:
                time.sleep(min(60, 5 * 2 ** k))
                continue
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


_POOL = ThreadPoolExecutor(32)


def chunk_span(start, length, chunk=4 << 20):
    return [(start + o, min(chunk, length - o)) for o in range(0, length, chunk)]


def parallel_ranges(url, base, ranges):
    """Fetch [(offset, length), ...] relative to `base` concurrently; returns list of bytes in order."""
    if len(ranges) == 1:
        o, n = ranges[0]
        return [http_range(url, base + o, n)]
    return list(_POOL.map(lambda r: http_range(url, base + r[0], r[1]), ranges))


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

    def read_rows(self, r0, r1, cols=None):
        """Rows [r0, r1) -> array (bands, rows, width). Strip-organised files only.

        All strips touching the rows are fetched as one byte span (in parallel chunks) and sliced locally.
        If `cols=(c0, c1)` is given for a pixel-interleaved file, only those columns of each row are requested
        (one small range per row, fetched in parallel), which is much cheaper for narrow windows.
        """
        r0, r1 = max(0, r0), min(self.height, r1)
        if self.is_tiled:
            raise NotImplementedError("tiled layout")
        it = self.dtype.itemsize
        rps = self.rowsperstrip
        spb = len(self.offsets) // (self.samples if self.planar == 2 else 1)  # strips per band
        s0, s1 = r0 // rps, (r1 - 1) // rps
        bands = range(self.samples) if self.planar == 2 else [0]
        out = []
        for b in bands:
            offs = self.offsets[b * spb + s0: b * spb + s1 + 1]
            cnts = self.bytecounts[b * spb + s0: b * spb + s1 + 1]
            if cols is not None and self.planar == 1 and rps == 1:
                c0, c1 = max(0, cols[0]), min(self.width, cols[1])
                px = self.samples * it
                rng = [(int(o) + c0 * px, (c1 - c0) * px) for o in offs]
                parts = parallel_ranges(self.url, self.base, rng)
                a = np.frombuffer(b"".join(parts), self.dtype).reshape(len(rng), c1 - c0, self.samples)
                return np.moveaxis(a, -1, 0).copy()
            span0 = int(offs.min()); span1 = int((offs + cnts).max())
            if span1 - span0 <= 2 * int(cnts.sum()) + (1 << 20):
                buf = b"".join(parallel_ranges(self.url, self.base, chunk_span(span0, span1 - span0)))
                pieces = [buf[int(o) - span0: int(o) - span0 + int(c)] for o, c in zip(offs, cnts)]
            else:  # strips scattered in the file: fetch each strip
                pieces = parallel_ranges(self.url, self.base, [(int(o), int(c)) for o, c in zip(offs, cnts)])
            per = self.width * (self.samples if self.planar == 1 else 1)
            a = np.concatenate([np.frombuffer(x, self.dtype)[: (len(x) // it) // per * per] for x in pieces])
            a = a.reshape(-1, self.width, self.samples if self.planar == 1 else 1)
            out.append(a[r0 - s0 * rps: r1 - s0 * rps])
        a = np.concatenate(out, axis=-1) if self.planar == 2 else out[0]
        return np.moveaxis(a, -1, 0).copy()

    def read_window(self, r0, r1, c0, c1):
        """Pixel window with zero padding outside the image -> (bands, r1-r0, c1-c0)."""
        out = np.zeros((self.samples, r1 - r0, c1 - c0), self.dtype)
        rr0, rr1, cc0, cc1 = max(0, r0), min(self.height, r1), max(0, c0), min(self.width, c1)
        if rr0 >= rr1 or cc0 >= cc1:
            return out
        if self.planar == 1 and self.rowsperstrip == 1:
            a = self.read_rows(rr0, rr1, cols=(cc0, cc1))
        else:
            a = self.read_rows(rr0, rr1)[:, :, cc0:cc1]
        out[:, rr0 - r0: rr1 - r0, cc0 - c0: cc1 - c0] = a
        return out

    def read_all(self):
        return self.read_rows(0, self.height)

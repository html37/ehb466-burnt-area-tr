"""List the members of a remote tar on Hugging Face via HTTP range requests (no full download).

Usage: python scripts/hf_tar_index.py <shard e.g. sen2_20_mod_500/00000.tar> [max_members]
Writes data/raw/floga_v2/index/<shard>.csv with name, offset (data start), size.
"""
import csv, sys, tarfile, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from burnt.remote_tiff import http_range

REPO = "https://huggingface.co/datasets/orion-ai-lab/FLOGA-GeoTIFFs/resolve/main/"

def index_tar(shard, max_members=10**9):
    url = REPO + shard
    class _S:  # adapter: range GET via burnt.remote_tiff.http_range (CDN URL, retries)
        def get(self, url, headers, timeout):
            a, b = headers["Range"][6:].split("-")
            class R: pass
            r = R(); r.content = http_range(url, int(a), int(b) - int(a) + 1); return r
    s = _S()
    off, rows = 0, []
    while len(rows) < max_members:
        r = s.get(url, headers={"Range": f"bytes={off}-{off + 511}"}, timeout=60)
        hdr = r.content
        if len(hdr) < 512 or hdr == b"\0" * 512:
            break
        ti = tarfile.TarInfo.frombuf(hdr, "utf-8", "surrogateescape")
        name = ti.name
        data_off = off + 512
        if ti.type in (tarfile.GNUTYPE_LONGNAME, b"x", b"g"):  # long name / pax header
            r2 = s.get(url, headers={"Range": f"bytes={data_off}-{data_off + ti.size - 1}"}, timeout=60)
            if ti.type == tarfile.GNUTYPE_LONGNAME:
                longname = r2.content.rstrip(b"\0").decode()
            else:
                longname = None
                for line in r2.content.decode().splitlines():
                    if " path=" in line:
                        longname = line.split(" path=", 1)[1]
            off = data_off + ((ti.size + 511) // 512) * 512
            r = s.get(url, headers={"Range": f"bytes={off}-{off + 511}"}, timeout=60)
            ti = tarfile.TarInfo.frombuf(r.content, "utf-8", "surrogateescape")
            name = longname or ti.name
            data_off = off + 512
        rows.append((name, data_off, ti.size))
        off = data_off + ((ti.size + 511) // 512) * 512
    return rows

if __name__ == "__main__":
    shard = sys.argv[1]
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    rows = index_tar(shard, mx)
    out = pathlib.Path("data/raw/floga_v2/index") / (shard.replace("/", "__") + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "offset", "size"]); w.writerows(rows)
    for r in rows[:40]:
        print(r)
    print(len(rows), "members", sum(r[2] for r in rows) / 1e9, "GB")

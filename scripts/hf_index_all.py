"""Index every 20 m FLOGA-v2 tar shard on Hugging Face (resumable, parallel). Writes data/raw/floga_v2/index/*.csv."""
import json, pathlib, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import csv
from hf_tar_index import index_tar

API = "https://huggingface.co/api/datasets/orion-ai-lab/FLOGA-GeoTIFFs/tree/main/sen2_20_mod_500"
OUT = pathlib.Path("data/raw/floga_v2/index")

def job(shard):
    out = OUT / (shard.replace("/", "__") + ".csv")
    if out.exists():
        return shard, "cached"
    rows = index_tar(shard)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "offset", "size"]); w.writerows(rows)
    tmp.replace(out)
    return shard, len(rows)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    listing = json.loads(urllib.request.urlopen(API).read())
    shards = sorted(x["path"] for x in listing if x["path"].endswith(".tar"))
    (OUT / "shards.json").write_text(json.dumps(listing, indent=1))
    with ThreadPoolExecutor(14) as ex:
        for shard, n in ex.map(job, shards):
            print(shard, n, flush=True)

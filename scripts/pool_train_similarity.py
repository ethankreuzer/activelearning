"""Per-molecule Tanimoto similarity of each K-ladder arm's generated pool to the 20k training set.

For every molecule in generated_pool_round_0001.csv, over ECFP4 bits (Morgan r=2, 2048 bits,
computed here from SMILES -- no stored embeddings are needed), against ALL 20,000 molecules of
data/ampc_svgpscale_20000.csv (the surrogates' training set; the pool is not part of the reference):
  max_sim_train    = similarity to the nearest training molecule
  median_sim_train = median similarity to the 20,000 training molecules
Writes <arm>/pool_train_similarity_round_0001.npz (max_sim_train, median_sim_train, pool_rows = pool
row of each value; molecules RDKit cannot parse are excluded, as in scripts/pool_diversity.py).
Usage: python scripts/pool_train_similarity.py [--limit N] [arm_dir ...]   (default: all kladder_* arms)
"""
import argparse, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
from pool_diversity import fp  # same fingerprint as the within-pool analysis

ROOT = Path("outputs/ampc/k_ladder")
TRAIN_CSV = Path("data/ampc_svgpscale_20000.csv")

def similarity_to_reference(pool_fps, ref_fps, dev, chunk):
    R = torch.from_numpy(ref_fps).to(dev, torch.float16)
    rc = R.float().sum(1)
    n = pool_fps.shape[0]
    mx = torch.empty(n); md = torch.empty(n)
    for i in range(0, n, chunk):
        X = torch.from_numpy(pool_fps[i:i + chunk]).to(dev, torch.float16)
        inter = (X @ R.T).float()
        sim = inter / (X.float().sum(1)[:, None] + rc[None, :] - inter).clamp(min=1)
        mx[i:i + chunk] = sim.max(1).values.cpu()
        md[i:i + chunk] = sim.median(1).values.cpu()   # lower median for an even count (20,000)
    return mx.numpy(), md.numpy()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arms = [Path(x) for x in a.arms] or sorted(ROOT.glob("kladder_*_steps5000"))
    print("device", dev, "arms", len(arms), flush=True)
    with Pool(4) as pool:
        train = pd.read_csv(TRAIN_CSV)["SMILE"].tolist()
        tf = pool.map(fp, train, chunksize=2000)
        assert all(f is not None for f in tf), "unparseable training SMILES"
        ref = np.stack(tf)
        print("train molecules", len(train), flush=True)
        for d in arms:
            out = d / "pool_train_similarity_round_0001.npz"
            if a.limit is None and out.exists():
                print("skip", d.name); continue
            if not (d / "generated_pool_round_0001.csv").exists():
                print("NO POOL (generation hit its cap), skipping", d.name, flush=True); continue
            t = time.time()
            smi = pd.read_csv(d / "generated_pool_round_0001.csv")["SMILES"].tolist()[: a.limit]
            fps = pool.map(fp, smi, chunksize=2000)
            keep = np.array([f is not None for f in fps])
            if (~keep).any():
                print(f"  WARNING {d.name}: {(~keep).sum()} unparseable SMILES excluded", flush=True)
            mx, md = similarity_to_reference(np.stack([f for f in fps if f is not None]), ref, dev, a.chunk)
            if a.limit is None:
                np.savez(out, max_sim_train=mx, median_sim_train=md, pool_rows=np.flatnonzero(keep))
            print(f"{d.name} n={len(smi)} max med={np.median(mx):.3f} median med={np.median(md):.3f} {time.time()-t:.0f}s", flush=True)

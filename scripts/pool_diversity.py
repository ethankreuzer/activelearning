"""Per-molecule Tanimoto diversity of each K-ladder arm's generated pool.

For every molecule in generated_pool_round_0001.csv (100k) over ECFP4 bits (Morgan r=2,
2048 bits, as in the replay buffer), against all OTHER molecules in the same pool:
  max_sim    = similarity to its nearest neighbour
  median_sim = median similarity to the other 99,999 molecules
Writes <arm>/pool_diversity_round_0001.npz (max_sim, median_sim, pool_rows = pool row of each value;
molecules RDKit cannot parse are excluded).
Usage: python scripts/pool_diversity.py [--limit N] [arm_dir ...]   (default: all kladder_* arms)
"""
import argparse, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, torch
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path("outputs/ampc/k_ladder")

def fp(smi):
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), arr)
    return arr

def diversity(fps, dev, chunk):
    X = torch.from_numpy(fps).to(dev, torch.float16)
    n = X.shape[0]
    cnt = X.float().sum(1)
    mx = torch.empty(n); md = torch.empty(n)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        inter = (X[i:j] @ X.T).float()
        sim = inter / (cnt[i:j, None] + cnt[None, :] - inter).clamp(min=1)
        idx = torch.arange(i, j, device=dev)
        sim[idx - i, idx] = float("nan")
        mx[i:j] = torch.nan_to_num(sim, nan=-1.0).max(1).values.cpu()
        md[i:j] = sim.nanmedian(1).values.cpu()
    return mx.numpy(), md.numpy()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2048)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arms = [Path(x) for x in a.arms] or sorted(ROOT.glob("kladder_*_steps5000"))
    print("device", dev, "arms", len(arms), flush=True)
    with Pool(4) as pool:
        for d in arms:
            out = d / "pool_diversity_round_0001.npz"
            if a.limit is None and out.exists():
                print("skip", d.name); continue
            if not (d / "generated_pool_round_0001.csv").exists():
                print("NO POOL (generation hit its cap), skipping", d.name, flush=True); continue
            t = time.time()
            smi = pd.read_csv(d / "generated_pool_round_0001.csv")["SMILES"].tolist()[: a.limit]
            fps = pool.map(fp, smi, chunksize=2000)
            # A handful of accepted generations fail RDKit sanitisation (e.g. cannot kekulize);
            # they are left out of the diversity statistics and their pool rows recorded.
            keep = np.array([f is not None for f in fps])
            mx, md = diversity(np.stack([f for f in fps if f is not None]), dev, a.chunk)
            if (~keep).any():
                print(f"  WARNING {d.name}: {(~keep).sum()} unparseable SMILES excluded, rows {np.flatnonzero(~keep).tolist()}", flush=True)
            if a.limit is None:
                np.savez(out, max_sim=mx, median_sim=md, pool_rows=np.flatnonzero(keep))
            print(f"{d.name} n={len(smi)} max med={np.median(mx):.3f} median med={np.median(md):.3f} {time.time()-t:.0f}s", flush=True)

"""Fit per-KV-head OSCAR-style rotations (cov eigvecs @ Hadamard) from a Q/K/V dump."""
import argparse, glob, torch
from scipy.linalg import hadamard

AP = argparse.ArgumentParser()
AP.add_argument("--dump", required=True)
AP.add_argument("--layers", type=int, required=True)
AP.add_argument("--out-k", required=True)
AP.add_argument("--out-v", required=True)
AP.add_argument("--max-files", type=int, default=20)
A = AP.parse_args()

H = torch.tensor(hadamard(128), dtype=torch.float32) / (128 ** 0.5)

def fit(X):
    C = (X.T @ X) / X.shape[0]
    ev, evec = torch.linalg.eigh(C)
    return (evec[:, torch.argsort(ev, descending=True)] @ H).contiguous()

for sub, outp in (("k", A.out_k), ("v", A.out_v)):
    layers = {}
    for lid in range(A.layers):
        fs = sorted(glob.glob("%s/layer_%d/%s/*.pt" % (A.dump, lid, sub)))[:A.max_files]
        T = torch.cat([torch.load(f, map_location="cpu", weights_only=False).float() for f in fs], 0)
        heads = T.shape[1]
        R = torch.stack([fit(T[:, h, :]) for h in range(heads)], 0)   # [heads,128,128]
        layers[lid] = {"layer_id": lid, "rotation": R}
        if lid % 12 == 0:
            print("[fit] %s layer %d heads=%d tokens=%d" % (sub, lid, heads, T.shape[0]), flush=True)
    torch.save({"format_version": 2, "objective": "cov_eig_hadamard_perhead",
                "source_grouping": "per_kv_head", "layers": layers}, outp)
    print("[fit] saved", outp)

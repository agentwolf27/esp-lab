"""
Does a SMALL nonlinear model remove site identity where linear projection failed?

Linear INLP killed individual identity (cats -0.47, dogs -0.55) but barely dented
pig LAB identity (0.941 -> 0.828 at rank 32). The question is whether that is a
property of site nuisance, or just a limitation of linear projection.

Model: domain-adversarial MLP (DANN, Ganin & Lempitsky). Frozen 768-d embedding ->
encoder MLP -> two heads: the CONTEXT head is trained normally, the GROUP head sits
behind a gradient-reversal layer so the encoder is pushed to make group unpredictable.

~200k parameters, 5k vectors, CPU, minutes. No GPU. No foundation-model training.

Reported at every adversarial strength lambda, same discipline as before:
  context accuracy on the HELD-OUT group   -- must hold or improve
  group accuracy from the learned features -- must fall
  lambda=0 is the plain-MLP control (no adversary): if it matches, the adversary did nothing.
"""
from __future__ import annotations
import json, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import torch, torch.nn as nn
torch.set_num_threads(2)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import StratifiedKFold

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out_adversarial"); os.makedirs(OUT, exist_ok=True)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DANN(nn.Module):
    def __init__(self, d_in, n_ctx, n_grp, h=256, z=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(h, z), nn.ReLU())
        self.ctx = nn.Linear(z, n_ctx)
        self.grp = nn.Sequential(nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, n_grp))
    def forward(self, x, lam=0.0):
        zz = self.enc(x)
        return self.ctx(zz), self.grp(GRL.apply(zz, lam)), zz


def probe_group(Z, grp, min_n=5):
    ok = np.isin(grp, [g for g in np.unique(grp) if (grp == g).sum() >= min_n])
    Zi, gi = Z[ok], grp[ok]
    if len(np.unique(gi)) < 2: return float("nan")
    yp = np.empty(len(gi), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Zi, gi):
        m, s = Zi[tr].mean(0), Zi[tr].std(0) + 1e-8
        yp[te] = LogisticRegression(max_iter=2000).fit((Zi[tr]-m)/s, gi[tr]).predict((Zi[te]-m)/s)
    return float((yp == gi).mean())


def run(X, y, grp, lam, epochs=60, seed=0):
    """leave-one-group-out; returns (context acc on held-out, group acc of learned features)"""
    torch.manual_seed(seed); np.random.seed(seed)
    groups = np.unique(grp); yp = np.empty_like(y); Zall = np.zeros((len(y), 128), dtype=np.float32)
    for g in groups:
        te = grp == g
        if len(np.unique(y[~te])) < 2:
            yp[te] = y[~te][0]; continue
        Xtr, ytr, gtr = X[~te], y[~te], grp[~te]
        m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr_n = torch.tensor((Xtr - m) / s, dtype=torch.float32)
        Xte_n = torch.tensor((X[te] - m) / s, dtype=torch.float32)
        gmap = {v: i for i, v in enumerate(np.unique(gtr))}
        gt = torch.tensor([gmap[v] for v in gtr]); yt = torch.tensor(ytr, dtype=torch.long)
        # class weights for the imbalanced context label
        cw = torch.tensor([len(ytr)/(2*max((ytr==k).sum(),1)) for k in [0,1]], dtype=torch.float32)
        net = DANN(X.shape[1], 2, len(gmap))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lc, lg = nn.CrossEntropyLoss(weight=cw), nn.CrossEntropyLoss()
        n = len(yt); bs = min(128, n)
        for ep in range(epochs):
            perm = torch.randperm(n)
            ramp = lam * min(1.0, ep / max(epochs * 0.3, 1))   # warm up the adversary
            for i in range(0, n, bs):
                idx = perm[i:i+bs]
                opt.zero_grad()
                pc, pg, _ = net(Xtr_n[idx], ramp)
                loss = lc(pc, yt[idx]) + (lg(pg, gt[idx]) if ramp > 0 else 0.0)
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pc, _, _ = net(Xte_n, 0.0); yp[te] = pc.argmax(1).numpy()
            _, _, ztr = net(Xtr_n, 0.0); _, _, zte = net(Xte_n, 0.0)
        Zall[~te] = ztr.numpy(); Zall[te] = zte.numpy()
    return bacc(y, yp), probe_group(Zall, grp)


def main():
    import cross_species as cs
    rows = cs.load_all()
    E = np.load(os.path.join(D, "out_cross", "emb_wavlm.npy")).astype(np.float32)
    sp = np.array([r["species"] for r in rows]); y = np.array([r["pol"] for r in rows])
    ind = np.array([r["indiv"] for r in rows])
    Ep = np.load(os.path.join(D, "out_pigs", "emb_wavlm.npy")).astype(np.float32)
    mp = json.load(open(os.path.join(D, "out_pigs", "meta.json")))
    ds = {
        "pigs (6 LABS)":  (Ep[0], np.array([m["pol"] for m in mp]), np.array([m["team"] for m in mp])),
        "cats (20 indiv)": (E[9][sp=="cat"], y[sp=="cat"], ind[sp=="cat"]),
        "dogs (10 indiv)": (E[9][sp=="dog"], y[sp=="dog"], ind[sp=="dog"]),
    }
    res = {}
    for name, (X, yy, gg) in ds.items():
        print(f"\n=== {name} — n={len(yy)}, {len(np.unique(gg))} groups "
              f"(chance ctx 0.500, chance group {1/len(np.unique(gg)):.3f}) ===", flush=True)
        print(f"  {'lambda':>7} | {'context (held-out)':>18} {'group acc':>10}")
        rows_out = []
        for lam in [0.0, 0.1, 0.3, 1.0, 3.0]:
            c, gacc = run(X, yy, gg, lam)
            rows_out.append({"lambda": lam, "ctx": c, "grp": gacc})
            tag = "  <- plain MLP, no adversary" if lam == 0 else ""
            print(f"  {lam:>7.1f} | {c:>18.3f} {gacc:>10.3f}{tag}", flush=True)
        res[name] = rows_out
        base = rows_out[0]
        best = max(rows_out[1:], key=lambda r: r["ctx"] - 0.5 * max(r["grp"] - 1/len(np.unique(gg)), 0))
        print(f"   -> lambda={best['lambda']}: context {best['ctx']-base['ctx']:+.3f}, "
              f"group {best['grp']-base['grp']:+.3f} vs the plain MLP")
        json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/results.json")


if __name__ == "__main__":
    main()

"""Did we stop too early? Push INLP much deeper on pig LAB identity (768 dims available)."""
import json, os, numpy as np, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS","2")
import phase1_invariance as P
D=os.path.dirname(os.path.abspath(__file__))
Ep=np.load(os.path.join(D,"out_pigs","emb_wavlm.npy")).astype(np.float32)
mp=json.load(open(os.path.join(D,"out_pigs","meta.json")))
X=Ep[0]; y=np.array([m["pol"] for m in mp]); grp=np.array([m["team"] for m in mp])
print(f"pigs: n={len(y)}, dim={X.shape[1]}, {len(np.unique(grp))} labs (chance id 0.167)")
print(f"{'rank':>5} | {'ctx':>7} {'lab id':>7} | {'ctx rand':>8} {'id rand':>8}")
rows=[]
for k in [0,32,64,128,192,256,384]:
    ci=P.loGo_with_projection(X,y,grp,k,"inlp"); ii=P.ident_after_projection(X,grp,k,"inlp")
    cr=P.loGo_with_projection(X,y,grp,k,"rand") if k else ci
    ir=P.ident_after_projection(X,grp,k,"rand") if k else ii
    rows.append({"rank":k,"ctx":ci,"id":ii,"ctx_rand":cr,"id_rand":ir})
    print(f"{k:>5} | {ci:>7.3f} {ii:>7.3f} | {cr:>8.3f} {ir:>8.3f}",flush=True)
    json.dump(rows,open(os.path.join(D,"out_invariance","pigs_deep_rank.json"),"w"),indent=1)
b,l=rows[0],rows[-1]
print(f"\nlab identity {b['id']:.3f} -> {l['id']:.3f} ({l['id']-b['id']:+.3f}) at rank {l['rank']}")
print(f"context      {b['ctx']:.3f} -> {l['ctx']:.3f} ({l['ctx']-b['ctx']:+.3f})")
print("VERDICT:", "site identity IS removable, we stopped too early" if l['id']<0.45
      else ("partially removable, still well above chance" if l['id']<0.75 else "site identity resists linear removal"))

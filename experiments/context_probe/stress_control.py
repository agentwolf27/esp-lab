"""
Control for the channel-stress result: is identity more fragile than context,
or just harder (10-20 classes vs 2)?

Fair test: make identity BINARY too. Randomly split each species' animals into
two halves and predict which half a clip came from -- same class count, same
probe, same information type (who), 20 random splits averaged.

Also drop gain-12: we z-normalise every waveform before the encoder, so a gain
change is removed by construction. It flips nothing by design, not by finding.
"""
import json, os, numpy as np, warnings
warnings.filterwarnings("ignore")
import torch; torch.set_num_threads(2)
D=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(D,"out_stress")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
import cross_species as cs
rows=cs.load_all(); sp=np.array([r["species"] for r in rows]); y=np.array([r["pol"] for r in rows]); ind=np.array([r["indiv"] for r in rows])
E={n:np.load(os.path.join(OUT,f"emb_{n}.npy")).astype(np.float32) for n in ["clean","lowpass4k","reverb","noise+10dB"]}
fit=lambda X,t: LogisticRegression(max_iter=3000,C=1.0,class_weight="balanced").fit(X,t)
res={}; rng=np.random.default_rng(0)
print(f"{'perturbation':<12s} {'layer':>5} | {'ctx flip':>9} {'binID flip':>11} {'ratio':>6}")
for name in ["lowpass4k","reverb","noise+10dB"]:
    for li in [3,6,9,12]:
        cf,bf=[],[]
        for spc in ["cat","dog"]:
            m=sp==spc
            Xc,Xp=E["clean"][li][m],E[name][li][m]
            mu,sd=Xc.mean(0),Xc.std(0)+1e-8; Zc,Zp=(Xc-mu)/sd,(Xp-mu)/sd
            p=fit(Zc,y[m]); cf.append(float((p.predict(Zc)!=p.predict(Zp)).mean()))
            animals=np.unique(ind[m]); flips=[]
            for _ in range(20):                      # binary identity: which half of the animals
                half=set(rng.choice(animals,len(animals)//2,replace=False))
                g=np.array([a in half for a in ind[m]]).astype(int)
                if len(np.unique(g))<2: continue
                q=fit(Zc,g); flips.append(float((q.predict(Zc)!=q.predict(Zp)).mean()))
            bf.append(float(np.mean(flips)))
        c,b=float(np.mean(cf)),float(np.mean(bf))
        res[f"{name}_L{li}"]={"ctx_flip":c,"binary_identity_flip":b,"ratio":b/max(c,1e-6)}
        print(f"{name:<12s} {li:>5} | {c:>9.3f} {b:>11.3f} {b/max(c,1e-6):>6.2f}x")
json.dump(res,open(os.path.join(OUT,"control.json"),"w"),indent=1)
r9=[res[f"{n}_L9"]["ratio"] for n in ["lowpass4k","reverb","noise+10dB"]]
print(f"\nL9 ratios (binary-identity flips / context flips): {['%.2f'%v for v in r9]}")
print("VERDICT:", "identity STILL more fragile at matched difficulty" if min(r9)>1.2
      else ("no difference once difficulty is matched -- the earlier result was a class-count artifact" if max(r9)<1.2 else "mixed"))

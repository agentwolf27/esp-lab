"""Embed the same 5,031 pig calls with HuBERT, then recompute the pig->cat / dog->cat cells."""
import json, os, numpy as np, warnings
warnings.filterwarnings("ignore")
D=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(D,"out_pigs")
import pigs_run as pr
meta=json.load(open(os.path.join(OUT,"meta.json")))
cache=os.path.join(OUT,"emb_hubert.npy")
if not os.path.exists(cache):
    # reload the exact same clips in the same order
    import soundfile as sf
    from scipy.signal import resample_poly
    adir=os.path.join(pr.PD,"Soundwel Dataset - Audio and Spectrograms")
    wavs=[]
    for m in meta:
        x,sr=sf.read(os.path.join(adir,m["file"]),dtype="float32")
        if x.ndim>1: x=x.mean(1)
        if sr!=pr.SR:
            g=np.gcd(int(sr),pr.SR); x=resample_poly(x,pr.SR//g,sr//g).astype(np.float32)
        x=x[:int(pr.MAX_S*pr.SR)]
        if len(x)<1600: x=np.pad(x,(0,1600-len(x)))
        wavs.append(x)
    print(len(wavs),"clips; embedding with HuBERT...",flush=True)
    E=pr.embed(wavs,"facebook/hubert-base-ls960"); np.save(cache,E.astype(np.float16))
E=np.load(cache).astype(np.float32)
y=np.array([m["pol"] for m in meta]); team=np.array([m["team"] for m in meta])
import cross_species as cs
rows=cs.load_all(); Ecd=np.load(os.path.join(D,"out_cross","emb_hubert.npy")).astype(np.float32)
sp=np.array([r["species"] for r in rows]); ycd=np.array([r["pol"] for r in rows]); ind=np.array([r["indiv"] for r in rows])
c,d=sp=="cat",sp=="dog"; L=E.shape[0]
rng=np.random.default_rng(1)
def cell(Ea,ya,ga,Eb,yb,gb,nperm=200):
    Zs=[pr.zgroup(Ea[l],ga) for l in range(L)]; Zt=[pr.zgroup(Eb[l],gb) for l in range(L)]
    obs=np.mean([pr.bacc(yb,pr.fit(Zs[l],ya).predict(Zt[l])) for l in range(L)])
    null=[]
    for _ in range(nperm):
        yp=ya.copy()
        for g in np.unique(ga):
            m=ga==g; yp[m]=rng.permutation(ya[m])
        null.append(np.mean([pr.bacc(yb,pr.fit(Zs[l],yp).predict(Zt[l])) for l in range(L)]))
    return float(obs), float((np.sum(np.array(null)>=obs)+1)/(nperm+1))
z0c=np.zeros(c.sum()); z0d=np.zeros(d.sum())
res={}
for name,(Ea,ya,ga,Eb,yb,gb) in {
  "pig->cat":(E,y,team,Ecd[:,c],ycd[c],z0c),
  "dog->cat":(Ecd[:,d],ycd[d],z0d,Ecd[:,c],ycd[c],z0c),
  "cat->pig":(Ecd[:,c],ycd[c],z0c,E,y,team),
  "pig->dog":(E,y,team,Ecd[:,d],ycd[d],z0d),
}.items():
    obs,p=cell(Ea,ya,ga,Eb,yb,gb); res[name]={"acc":obs,"p":p}
    print(f"  HuBERT {name:<9s} {obs:.3f}  p={p:.3f}",flush=True)
lo,per=pr.loto(pr.zgroup(E[0],np.zeros(len(y))),y,team)
res["pig_within_loto_L0"]=lo
print(f"  HuBERT pig within (held-out lab, L0): {lo:.3f}")
json.dump(res,open(os.path.join(OUT,"hubert_replication.json"),"w"),indent=1); print("wrote hubert_replication.json")

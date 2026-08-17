"""Third encoder FAMILY: AVES-bio (bioacoustic SSL, not speech). Full 3x3 + placebo."""
import json, os, numpy as np, warnings, torch, torchaudio
warnings.filterwarnings("ignore")
D=os.path.dirname(os.path.abspath(__file__))
import pigs_run as pr, cross_species as cs
def aves():
    cfg=json.load(open(os.path.join(D,"out_enc","aves-base-bio.json")))
    m=torchaudio.models.wav2vec2_model(**cfg,aux_num_out=None)
    m.load_state_dict(torch.load(os.path.join(D,"out_enc","aves-base-bio.pt"),map_location="cpu")); return m.eval()
def embed(m,wavs):
    E=[]
    with torch.no_grad():
        for i,x in enumerate(wavs):
            f,_=m.extract_features(torch.from_numpy(x).float().unsqueeze(0))
            E.append(np.stack([h[0].mean(0).numpy() for h in f]))
            if (i+1)%500==0: print(f"    {i+1}/{len(wavs)}",flush=True)
    return np.stack(E,axis=1)
m=aves()
# cats+dogs
cd_cache=os.path.join(D,"out_cross","emb_aves.npy")
rows=cs.load_all()
if os.path.exists(cd_cache): Ecd=np.load(cd_cache).astype(np.float32)
else:
    print("embedding cats+dogs with AVES-bio...",flush=True); Ecd=embed(m,[r["wav"] for r in rows]); np.save(cd_cache,Ecd.astype(np.float16))
sp=np.array([r["species"] for r in rows]); ycd=np.array([r["pol"] for r in rows]); ind=np.array([r["indiv"] for r in rows])
# pigs (same clips/order as meta.json)
meta=json.load(open(os.path.join(D,"out_pigs","meta.json")))
p_cache=os.path.join(D,"out_pigs","emb_aves.npy")
if os.path.exists(p_cache): Ep=np.load(p_cache).astype(np.float32)
else:
    import soundfile as sf
    from scipy.signal import resample_poly
    adir=os.path.join(pr.PD,"Soundwel Dataset - Audio and Spectrograms"); wavs=[]
    for mm in meta:
        x,sr=sf.read(os.path.join(adir,mm["file"]),dtype="float32")
        if x.ndim>1: x=x.mean(1)
        if sr!=pr.SR: g=np.gcd(int(sr),pr.SR); x=resample_poly(x,pr.SR//g,sr//g).astype(np.float32)
        x=x[:int(pr.MAX_S*pr.SR)]
        if len(x)<1600: x=np.pad(x,(0,1600-len(x)))
        wavs.append(x)
    print("embedding pigs with AVES-bio...",flush=True); Ep=embed(m,wavs); np.save(p_cache,Ep.astype(np.float16))
yp=np.array([mm["pol"] for mm in meta]); team=np.array([mm["team"] for mm in meta])
c,d=sp=="cat",sp=="dog"; L=Ep.shape[0]; rng=np.random.default_rng(2)
species={"cat":(Ecd[:,c],ycd[c],np.zeros(c.sum())),"dog":(Ecd[:,d],ycd[d],np.zeros(d.sum())),"pig":(Ep,yp,team)}
def cell(a,b,nperm=200):
    Ea,ya,ga=species[a]; Eb,yb,gb=species[b]
    Zs=[pr.zgroup(Ea[l],ga) for l in range(L)]; Zt=[pr.zgroup(Eb[l],gb) for l in range(L)]
    obs=float(np.mean([pr.bacc(yb,pr.fit(Zs[l],ya).predict(Zt[l])) for l in range(L)]))
    null=[]
    for _ in range(nperm):
        yq=ya.copy()
        for g in np.unique(ga):
            mm=ga==g; yq[mm]=rng.permutation(ya[mm])
        null.append(np.mean([pr.bacc(yb,pr.fit(Zs[l],yq).predict(Zt[l])) for l in range(L)]))
    return obs,float((np.sum(np.array(null)>=obs)+1)/(nperm+1))
res={}
print("\n=== AVES-bio 3x3 (mean over layers, chance 0.5) ===")
for a in ["cat","dog","pig"]:
    for b in ["cat","dog","pig"]:
        if a==b: continue
        obs,p=cell(a,b); res[f"{a}->{b}"]={"acc":obs,"p":p}; print(f"  {a}->{b}  {obs:.3f}  p={p:.3f}",flush=True)
# placebo: dog sex -> cat affect
import pandas as pd
ann=pd.read_csv(os.path.join(D,"dogs","annotations.csv")); sexmap={r["filename"]:r["sex"] for _,r in ann.iterrows()}
sx=np.array([{"female":0,"male":1}.get(sexmap.get(os.path.basename(r["path"]),""),-1) for r in rows])
ok=(sx>=0)&d
accs=[pr.bacc(ycd[c],pr.fit(pr.zgroup(Ecd[l][ok],np.zeros(ok.sum())),sx[ok]).predict(pr.zgroup(Ecd[l][c],np.zeros(c.sum())))) for l in range(L)]
res["placebo_dogsex->cat"]=float(np.mean(accs)); print(f"  placebo dog SEX -> cat AFFECT {np.mean(accs):.3f}")
lo,_=pr.loto(pr.zgroup(Ep[0],np.zeros(len(yp))),yp,team); res["pig_within_L0"]=lo; print(f"  pig within (held-out lab, L0) {lo:.3f}")
json.dump(res,open(os.path.join(D,"out_pigs","aves_matrix.json"),"w"),indent=1); print("wrote out_pigs/aves_matrix.json")

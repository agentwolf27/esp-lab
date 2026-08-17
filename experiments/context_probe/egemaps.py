"""
eGeMAPS baseline (openSMILE, 88 functionals) on CatMeows -- the honest
paralinguistics baseline: it includes F0, jitter, shimmer, HNR, formants,
i.e. exactly the features the affect literature says carry arousal/valence.
If eGeMAPS lands near the encoders under leave-one-cat-out, the "encoders beat
handcrafted" claim shrinks; if it lands near MFCC-85 (0.399), it holds.
"""
import glob, os, json, warnings, numpy as np
warnings.filterwarnings("ignore")
import opensmile, soundfile as sf
from scipy.signal import resample_poly
D=os.path.dirname(os.path.abspath(__file__))
sm=opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
files=sorted(glob.glob(os.path.join(D,"catmeows","**","*.wav"),recursive=True))
X,y,cats=[],[],[]
for f in files:
    x,sr=sf.read(f,dtype="float32")
    if x.ndim>1: x=x.mean(1)
    if sr!=16000:
        g=np.gcd(int(sr),16000); x=resample_poly(x,16000//g,sr//g).astype(np.float32)
    if len(x)<800: continue
    feats=sm.process_signal(x,16000).values[0]
    p=os.path.basename(f)[:-4].split("_")
    X.append(feats); y.append(p[0]); cats.append(p[1])
X=np.nan_to_num(np.array(X,dtype=np.float32)); y=np.array(y); cats=np.array(cats)
print("eGeMAPS:",X.shape)
import context_probe as cp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
mk=lambda: LogisticRegression(max_iter=3000,C=0.5)
loco=bacc(y,cp.leave_one_cat_out(X,y,cats,mk)); rnd=bacc(y,cp.random_split_cv(X,y,mk))
# identity
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
ok=np.isin(cats,[c for c in np.unique(cats) if (cats==c).sum()>=5]); Xi,ci=X[ok],cats[ok]; yp=np.empty(ok.sum(),dtype=object)
for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xi,ci):
    sc=StandardScaler().fit(Xi[tr]); yp[te]=LogisticRegression(max_iter=3000).fit(sc.transform(Xi[tr]),ci[tr]).predict(sc.transform(Xi[te]))
ident=float((yp==ci).mean())
print(f"  context, held-out cat : {loco:.3f}   (MFCC-85 was 0.399; best encoder 0.571)")
print(f"  context, random split : {rnd:.3f}")
print(f"  cat identity          : {ident:.3f}")
r=json.load(open(os.path.join(D,"out_enc","results.json")))
r["egemaps"]={"layers":[{"layer":0,"dim":int(X.shape[1]),"ctx_loco":float(loco),"ctx_rand":float(rnd),"identity":ident}],
              "best":{"layer":0,"dim":int(X.shape[1]),"ctx_loco":float(loco),"ctx_rand":float(rnd),"identity":ident}}
json.dump(r,open(os.path.join(D,"out_enc","results.json"),"w"),indent=1); print("added to out_enc/results.json")

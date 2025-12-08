# pttppg_pipeline_v7.py
import os, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn import metrics

# ========== Models ==========
class CNNBiLSTMAE(nn.Module):
    def __init__(self, in_ch: int = 1, hidden: int = 256, bottleneck: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, padding=2), nn.ReLU(True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(True),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),
        )
        self.bi_lstm = nn.LSTM(64, hidden//2, num_layers=1, batch_first=True, bidirectional=True)
        self.bottleneck = nn.Linear(hidden, bottleneck)
        self.decoder_lstm = nn.LSTM(bottleneck, hidden//2, num_layers=1, batch_first=True, bidirectional=True)
        self.to_channels = nn.Conv1d(hidden, in_ch, kernel_size=1)
    def forward(self, x):
        B, C, T = x.shape
        z = self.encoder(x).permute(0,2,1)
        z,_ = self.bi_lstm(z)
        z = self.bottleneck(z)
        z,_ = self.decoder_lstm(z)
        z = z.permute(0,2,1)
        out = F.interpolate(z, size=T, mode='linear', align_corners=False)
        return self.to_channels(out)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=p), nn.ReLU(True),
            nn.Conv1d(out_ch, out_ch, k, padding=p), nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)

class UNet1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base);     self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock(base, base*2);    self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock(base*2, base*4);  self.pool3 = nn.MaxPool1d(2)
        self.enc4 = ConvBlock(base*4, base*8)
        self.up3 = nn.ConvTranspose1d(base*8, base*4, 2, 2); self.dec3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose1d(base*4, base*2, 2, 2); self.dec2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose1d(base*2, base, 2, 2);   self.dec1 = ConvBlock(base*2, base)
        self.out = nn.Conv1d(base, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1)); e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        d3 = self.up3(e4); d3 = torch.cat([d3, e3], 1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], 1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], 1); d1 = self.dec1(d1)
        return self.out(d1)

class MultiResSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(64,128,256), hops=(16,32,64), wins=(64,128,256)):
        super().__init__(); self.fft=fft_sizes; self.h=hops; self.w=wins
    def stft_mag(self, x, n, h, w): return torch.abs(torch.stft(x, n_fft=n, hop_length=h, win_length=w, return_complex=True))
    def forward(self, x, y):
        x=x.squeeze(1); y=y.squeeze(1); loss=0.0
        for n,h,w in zip(self.fft, self.h, self.w): loss += F.l1_loss(self.stft_mag(x,n,h,w), self.stft_mag(y,n,h,w))
        return loss/len(self.fft)

class HRBandEnergyReg(nn.Module):
    def __init__(self, fs: int, band=(0.6,3.5), n_fft=256, hop=64, win_length=256, w_in=0.05, w_out=0.05):
        super().__init__(); f=torch.linspace(0, fs/2, n_fft//2+1)
        self.register_buffer("mask_in",  ((f>=band[0]) & (f<=band[1])).float().view(1,-1,1))
        self.register_buffer("mask_out", ((f<band[0]) | (f>band[1])).float().view(1,-1,1))
        self.n=n_fft; self.h=hop; self.w=win_length; self.wi=w_in; self.wo=w_out
    def forward(self, y):
        y=y.squeeze(1)
        Y = torch.stft(y, n_fft=self.n, hop_length=self.h, win_length=self.w, return_complex=True)
        mag = torch.abs(Y)
        return self.wo*(mag*self.mask_out).mean() - self.wi*(mag*self.mask_in).mean()

# ========== Utils ==========

try:
    import pywt
except Exception:
    pywt = None

def zscore(x): m,s=np.mean(x),np.std(x)+1e-8; return (x-m)/s

def dwt_compress(x, wavelet="db4", level=2):
    if pywt is None: return x
    c=pywt.wavedec(x, wavelet, level=level); a=c[0]
    xp=np.linspace(0,1,len(a)); xq=np.linspace(0,1,len(x))
    return np.interp(xq,xp,a)

def butter_bandpass(sig, fs, lo=0.5, hi=8.0, order=3):
    try:
        from scipy.signal import butter, filtfilt
    except Exception: return sig
    b,a=butter(order,[lo/(fs/2),hi/(fs/2)],btype="band"); return filtfilt(b,a,sig)

def split_windows(N, fs, win, hop):
    w=int(win*fs); h=int(hop*fs)
    for s in range(0, max(1, N-w+1), h): yield s, s+w

def acc_rms(acc_win): return float(np.sqrt(np.mean(np.sum(acc_win**2, axis=1))))

# ========== Data Loading (CSV) ==========
CAND = {
    "time": ["time","Time","t"],
    "ecg": ["ecg","ECG"],
    "peaks": ["peaks","Rpeaks","r_peaks"],
    "pleth_4": ["pleth_4"], "pleth_5": ["pleth_5"], "pleth_6": ["pleth_6"],
    "a_x": ["a_x","AX","accX"], "a_y": ["a_y","AY","accY"], "a_z": ["a_z","AZ","accZ"],
    "g_x": ["g_x","GX","gyroX"], "g_y": ["g_y","GY","gyroY"], "g_z": ["g_z","GZ","gyroZ"],
}
def _pick(df, names):
    for n in names:
        if n in df.columns: return n
    return None

def load_physionet_csv(root: Path, fs: float):
    csv_dir = root / "csv"
    if not csv_dir.exists(): csv_dir = root / "files" / "pulse-transit-time-ppg" / "1.1.0" / "csv"
    paths = sorted([p for p in csv_dir.glob("s*_*.csv") if p.name!="subjects_info.csv"])
    sub2recs = {}
    for p in paths:
        df = pd.read_csv(p); m={k:_pick(df,v) for k,v in CAND.items()}; rec={}
        for k in CAND.keys():
            if m.get(k) is not None: rec[k]=df[m[k]].to_numpy(float)
        if "time" not in rec: rec["time"]=np.arange(len(df))/fs
        sid = p.stem.split("_")[0]; sub2recs.setdefault(sid, []).append(rec)
    return sub2recs

# ========== Datasets ==========
class DatasetAE(Dataset):
    def __init__(self, records, fs, win, hop, motion_thresh, setup):
        self.records=records; self.fs=fs; self.win=win; self.hop=hop; self.motion_thresh=motion_thresh; self.setup=setup
        self.index=[]
        for i,r in enumerate(records):
            N = len(r["pleth_5"]) if "pleth_5" in r else len(r["pleth_4"])
            for s,e in split_windows(N, fs, win, hop): self.index.append((i,s,e))
    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        i,s,e=self.index[idx]; r=self.records[i]
        ppg = (r["pleth_5"] if "pleth_5" in r else r["pleth_4"])[s:e].astype(np.float32)
        ax=r.get("a_x",np.zeros_like(ppg)); ay=r.get("a_y",np.zeros_like(ppg)); az=r.get("a_z",np.zeros_like(ppg))
        acc = np.column_stack([ax[s:e],ay[s:e],az[s:e]]).astype(np.float32)
        label = int(acc_rms(acc) > self.motion_thresh)
        x = zscore(dwt_compress(ppg))[None,:].astype(np.float32)
        y = zscore(ppg)[None,:].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(label, dtype=torch.long)

class DatasetDenoise(Dataset):
    def __init__(self, records, fs, win, hop, motion_thresh, setup):
        self.records=records; self.fs=fs; self.win=win; self.hop=hop; self.motion_thresh=motion_thresh; self.setup=setup
        self.index=[]
        for i,r in enumerate(records):
            N = len(r["pleth_5"]) if "pleth_5" in r else len(r["pleth_4"])
            for s,e in split_windows(N, fs, win, hop): self.index.append((i,s,e))
    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        i,s,e=self.index[idx]; r=self.records[i]
        p5 = (r["pleth_5"] if "pleth_5" in r else r["pleth_4"])[s:e].astype(np.float32)
        p4 = r.get("pleth_4", np.zeros_like(p5))[s:e].astype(np.float32)
        ax=r.get("a_x",np.zeros_like(p5))[s:e]; ay=r.get("a_y",np.zeros_like(p5))[s:e]; az=r.get("a_z",np.zeros_like(p5))[s:e]
        gx=r.get("g_x",np.zeros_like(p5))[s:e]; gy=r.get("g_y",np.zeros_like(p5))[s:e]; gz=r.get("g_z",np.zeros_like(p5))[s:e]
        x_list = [zscore(p5)[None,:], zscore(np.array([ax,ay,az])), zscore(np.array([gx,gy,gz])), zscore(p4)[None,:]]
        if self.setup==2:
            if "ecg" in r:     x_list.append(zscore(r["ecg"][s:e])[None,:])
            if "peaks" in r:   x_list.append(zscore(r["peaks"][s:e])[None,:])
            if "pleth_6" in r: x_list.append(zscore(r["pleth_6"][s:e])[None,:])
        x = np.vstack(x_list).astype(np.float32)
        y = zscore(butter_bandpass(p5, self.fs, 0.5, 8.0))[None,:].astype(np.float32)
        is_clean = int(acc_rms(np.column_stack([ax,ay,az])) <= self.motion_thresh)
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(is_clean, dtype=torch.float32)

# ========== Train & Eval ==========
def train_detector(train_ds, val_ds, fs, epochs=20, lr=1e-3):
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=CNNBiLSTMAE(in_ch=1).to(device); opt=torch.optim.Adam(model.parameters(), lr=lr)
    TL=DataLoader(train_ds,64,True); VL=DataLoader(val_ds,128,False); best={"val":1e9,"state":None}
    for ep in range(1,epochs+1):
        model.train(); tr=0.0
        for x,y,_ in TL:
            x,y=x.to(device),y.to(device); yhat=model(x); loss=F.mse_loss(yhat,y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tr+=loss.item()*x.size(0)
        tr/=len(TL.dataset); model.eval(); va=0.0
        with torch.no_grad():
            for x,y,_ in VL:
                x,y=x.to(device),y.to(device); yhat=model(x); va+=F.mse_loss(yhat,y).item()*x.size(0)
        va/=len(VL.dataset)
        if va<best["val"]: best={"val":va,"state":{k:v.cpu() for k,v in model.state_dict().items()}}
    model.load_state_dict(best["state"]); return model, best["val"]

def eval_detector(model, ds, fs, thr=None):
    device=next(model.parameters()).device
    L=DataLoader(ds,128,False); losses=[]; labels=[]
    with torch.no_grad():
        for x,y,lab in L:
            x,y=x.to(device),y.to(device); yhat=model(x)
            l=F.mse_loss(yhat,y,reduction="none").mean(dim=[1,2]).cpu().numpy()
            losses.append(l); labels.append(lab.numpy())
    losses=np.concatenate(losses); labels=np.concatenate(labels)
    thr=float(np.quantile(losses,0.95)) if thr is None else float(thr)
    pred=(losses>thr).astype(int)
    return dict(threshold=thr,
                pr_auc=float(metrics.average_precision_score(labels,losses)),
                roc_auc=float(metrics.roc_auc_score(labels,losses)),
                f1=float(metrics.f1_score(labels,pred)),
                bal_acc=float(metrics.balanced_accuracy_score(labels,pred)))

def train_denoiser(train_ds, val_ds, fs, epochs=20, lr=1e-3, lam_spec=0.5, lam_hr=0.1, lam_cons=0.2):
    device="cuda" if torch.cuda.is_available() else "cpu"
    in_ch = next(iter(DataLoader(train_ds,1)))[0].shape[1]
    model=UNet1D(in_ch=in_ch, out_ch=1, base=32).to(device); opt=torch.optim.Adam(model.parameters(), lr=lr)
    spec=MultiResSTFTLoss(); hr=HRBandEnergyReg(fs=fs, w_in=0.05, w_out=0.05)
    TL=DataLoader(train_ds,16,True); VL=DataLoader(val_ds,32,False); best={"val":1e9,"state":None,"hr_band":None}
    for ep in range(1,epochs+1):
        model.train(); tr=0.0
        for x,y,is_clean in TL:
            x,y,is_clean=x.to(device),y.to(device),is_clean.to(device)
            yhat=model(x)
            xa=x.clone(); xb=x.clone()
            xa[:,0:1,:]+=torch.randn_like(xa[:,0:1,:])*0.01
            xb[:,0:1,:]+=torch.randn_like(xb[:,0:1,:])*0.01
            ya=model(xa); yb=model(xb)
            l_rec = F.l1_loss(yhat[is_clean>0.5], y[is_clean>0.5]) if (is_clean>0.5).any() else torch.tensor(0.0,device=device)
            l_spec=spec(yhat,y); l_hr=hr(yhat); l_cons=F.l1_loss(ya,yb)
            loss = l_rec + lam_spec*l_spec + lam_hr*l_hr + lam_cons*l_cons
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tr+=loss.item()*x.size(0)
        tr/=len(TL.dataset); model.eval(); va=0.0; va_hr=0.0
        with torch.no_grad():
            for x,y,_ in VL:
                x,y=x.to(device),y.to(device); yhat=model(x)
                loss=F.l1_loss(yhat,y)+lam_spec*spec(yhat,y)+lam_hr*hr(yhat)
                va+=loss.item()*x.size(0); va_hr += hr(yhat).item()*x.size(0)
        va/=len(VL.dataset); va_hr/=len(VL.dataset)
        if va<best["val"]: best={"val":va,"state":{k:v.cpu() for k,v in model.state_dict().items()},"hr_band":va_hr}
    model.load_state_dict(best["state"]); return model, best

def eval_denoiser(model, ds, fs):
    device=next(model.parameters()).device
    L=DataLoader(ds,64,False); l1s=[]; snr_imp=[]; hr_losses=[]; hr=HRBandEnergyReg(fs=fs)
    with torch.no_grad():
        for x,y,_ in L:
            x,y=x.to(device),y.to(device); yhat=model(x)
            l1s.append(F.l1_loss(yhat,y).item())
            noisy=x[:,0:1,:]; err_b=(noisy-y).pow(2).mean().item(); err_a=(yhat-y).pow(2).mean().item()
            snr_imp.append(10*np.log10((err_b+1e-9)/(err_a+1e-9))); hr_losses.append(hr(yhat).item())
    return dict(l1=float(np.mean(l1s)) if l1s else float("nan"),
                snr_improvement_db=float(np.mean(snr_imp)) if snr_imp else float("nan"),
                hr_band_loss=float(np.mean(hr_losses)) if hr_losses else float("nan"))

# ========== Orchestration ==========
def subjectwise(records_by_subj):
    recs=[]; groups=[]
    for sid, lst in sorted(records_by_subj.items()):
        for r in lst: recs.append(r); groups.append(sid)
    return recs, groups

def split_indices(groups, mode="kfold", n_splits=5, train_size=0.8, random_state=42):
    idx=np.arange(len(groups))
    if mode=="holdout":
        gss=GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
        tr, va = next(gss.split(idx, groups=groups))
        return [(tr,va)]
    else:
        gkf=GroupKFold(n_splits=min(len(set(groups)), n_splits))
        return list(gkf.split(idx, groups=groups, groups=groups))

def run_one_setup(records_by_subj, fs, win, hop, motion_thresh, split_mode, n_splits, train_size,
                  epochs_ae, epochs_den, lr, thr=None, setup=1, outdir=Path("results"),
                  lam_spec=0.5, lam_hr=0.1, lam_cons=0.2):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    records, groups = subjectwise(records_by_subj)
    folds = split_indices(groups, mode=split_mode, n_splits=n_splits, train_size=train_size)
    det_folds=[]; den_folds=[]; det_hold=None; den_hold=None
    for k,(tr,va) in enumerate(folds, start=1):
        tr_recs=[records[i] for i in tr]; va_recs=[records[i] for i in va]
        tr_ae=DatasetAE(tr_recs, fs, win, hop, motion_thresh, setup); va_ae=DatasetAE(va_recs, fs, win, hop, motion_thresh, setup)
        ae, best_ae = train_detector(tr_ae, va_ae, fs, epochs_ae, lr)
        det = eval_detector(ae, va_ae, fs, thr); det.update({"val_loss":best_ae,"fold":k}); det_folds.append(det); det_hold=det

        tr_dn=DatasetDenoise(tr_recs, fs, win, hop, motion_thresh, setup); va_dn=DatasetDenoise(va_recs, fs, win, hop, motion_thresh, setup)
        dn, best_dn = train_denoiser(tr_dn, va_dn, fs, epochs_den, lr, lam_spec=lam_spec, lam_hr=lam_hr, lam_cons=lam_cons)
        den = eval_denoiser(dn, va_dn, fs); den.update({"val_loss":best_dn["val"],"hr_band_loss":best_dn["hr_band"],"fold":k}); den_folds.append(den); den_hold=den

    (outdir / "detector_results.json").write_text(json.dumps({"folds":det_folds,"holdout":det_hold}, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "denoiser_results.json").write_text(json.dumps({"folds":den_folds,"holdout":den_hold}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"detector":{"folds":det_folds,"holdout":det_hold}, "denoiser":{"folds":den_folds,"holdout":den_hold}}

def run_both_setups(data_root: str, fs=500.0, win=6.0, hop=1.0,
                    motion_thresh=0.8, split_mode="kfold", n_splits=5, train_size=0.8,
                    epochs_ae=20, epochs_denoise=20, lr=1e-3, detector_threshold=None,
                    outdir="results", lam_spec=0.5, lam_hr=0.1, lam_cons=0.2):
    root=Path(data_root)
    sub2recs=load_physionet_csv(root, fs=float(fs))
    out1=Path(outdir)/"setup1"
    out2=Path(outdir)/"setup2"
    res1=run_one_setup(sub2recs, fs, win, hop, motion_thresh, split_mode, n_splits, train_size,
                       epochs_ae, epochs_denoise, lr, detector_threshold, setup=1, outdir=out1,
                       lam_spec=lam_spec, lam_hr=lam_hr, lam_cons=lam_cons)
    res2=run_one_setup(sub2recs, fs, win, hop, motion_thresh, split_mode, n_splits, train_size,
                       epochs_ae, epochs_denoise, lr, detector_threshold, setup=2, outdir=out2,
                       lam_spec=lam_spec, lam_hr=lam_hr, lam_cons=lam_cons)
    comp = {
        "detector": {"setup1_holdout": res1["detector"]["holdout"], "setup2_holdout": res2["detector"]["holdout"]},
        "denoiser": {"setup1_holdout": res1["denoiser"]["holdout"], "setup2_holdout": res2["denoiser"]["holdout"]}
    }
    Path(outdir).mkdir(exist_ok=True)
    (Path(outdir)/"compare.json").write_text(json.dumps(comp, ensure_ascii=False, indent=2), encoding="utf-8")
    return res1, res2, comp

# ========== CLI ==========
def parse_cli():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--split_mode", type=str, default="kfold", choices=["kfold","holdout"])
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--train_size", type=float, default=0.8)
    ap.add_argument("--epochs_ae", type=int, default=20)
    ap.add_argument("--epochs_denoise", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--motion_thresh", type=float, default=0.8)
    ap.add_argument("--detector_threshold", type=float, default=None)
    ap.add_argument("--lam_spec", type=float, default=0.5)
    ap.add_argument("--lam_hr", type=float, default=0.1)
    ap.add_argument("--lam_cons", type=float, default=0.2)
    ap.add_argument("--outdir", type=str, default="results")
    return ap.parse_args()

if __name__=="__main__":
    args=parse_cli()
    run_both_setups(
        data_root=args.data_root, fs=args.fs, win=args.win, hop=args.hop,
        motion_thresh=args.motion_thresh, split_mode=args.split_mode, n_splits=args.n_splits, train_size=args.train_size,
        epochs_ae=args.epochs_ae, epochs_denoise=args.epochs_denoise, lr=args.lr, detector_threshold=args.detector_threshold,
        outdir=args.outdir, lam_spec=args.lam_spec, lam_hr=args.lam_hr, lam_cons=args.lam_cons
    )
    print("Done. See results/setup1, results/setup2 and results/compare.json")

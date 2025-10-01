# ======================================================================
# 0) Imports & setup
# ======================================================================
import os, re, math, random, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    brier_score_loss
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# external models
# !pip -q install tab-transformer-pytorch xgboost
from tab_transformer_pytorch import TabTransformer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# deterministic-ish
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATADIR = Path("/kaggle/input/kasese-hwc")
RAW_COMBINED = DATADIR / "kasese-hwc-data-2021-combined-2021-2022-partly-cleaned.csv"

OUTDIR = Path("/kaggle/working/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)


# %%
# ======================================================================
# 1) Load combined CSV & light cleaning
# ======================================================================
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"[^a-z0-9]+", "_", regex=True)
          .str.strip("_")
    )
    return df

def parse_date_any(x):
    """Robust mixed-format parser: day-first dates; also 'March 2021' -> 1st of month."""
    if pd.isna(x): return pd.NaT
    s = str(x).strip()
    # try common exact formats (day-first)
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try: return datetime.strptime(s, fmt)
        except Exception: pass
    # month-name + year
    for fmt in ("%B %Y", "%b %Y"):
        try:
            d = datetime.strptime(s, fmt)
            return d.replace(day=1)
        except Exception: pass
    # last resort
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def clean_text(s):
    if pd.isna(s): return np.nan
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s.title()

def fix_fields(df: pd.DataFrame) -> pd.DataFrame:
    # the combined file has these uppercase headers:
    # DATE, CA, DISTRICT, SUBCOUNTY, PARISH, VILLAGE, WILDLIFE, NATURE OF CONFLICT, ACTION TAKEN
    df = df.rename(columns={
        "date": "date",
        "ca": "ca",
        "district": "district",
        "subcounty": "subcounty",
        "parish": "parish",
        "village": "village",
        "wildlife": "species",
        "nature_of_conflict": "conflict_type",
        "action_taken": "response",
        # handle if the original mixed-case survived:
        "Date": "date",
        "CA": "ca",
        "DISTRICT": "district",
        "SUBCOUNTY": "subcounty",
        "PARISH": "parish",
        "VILLAGE": "village",
        "WILDLIFE": "species",
        "NATURE OF CONFLICT": "conflict_type",
        "ACTION TAKEN": "response",
    }).copy()
    # normalise simple texts
    for col in ["district","subcounty","parish","village","ca","species","conflict_type","response"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    return df

def normalise_conflict(raw):
    if pd.isna(raw): return np.nan
    s = re.sub(r"\s+", " ", str(raw)).strip().lower()
    s = s.replace("damages", "damage").replace("destruction", "damage")
    s = s.replace("threats", "threat").replace("threatening", "threat")
    s = s.replace("human threatening", "human threat")
    s = s.replace("predationof", "predation of ").replace("injuryof", "injury of ")
    s = s.replace("house damage", "property damage")
    s = s.replace("water tank damage", "property damage").replace("food stores damage","property damage")

    if "human death" in s or re.search(r"\bdeath\b", s): return "Human Death"
    if "human injury" in s or "injured" in s or "attack" in s: return "Human Injury"
    if "human threat" in s or "threat" in s: return "Human Threat"
    if "predation" in s or "livestock" in s or re.search(r"\b(goat|cow|pig|calf)\b", s): return "Livestock Predation"
    if "property damage" in s or "gate wall" in s: return "Property Damage"
    if "crop raiding" in s or "crop damage" in s: return "Crop Damage"
    if "assessment" in s: return "Assessment/Admin"
    return s.title()

raw = pd.read_csv(RAW_COMBINED, dtype=str)  # keep as string first
raw = clean_cols(raw)
raw = fix_fields(raw)
raw["date_parsed"] = raw["date"].apply(parse_date_any)

# keep core columns
keep = ["date_parsed","district","subcounty","parish","village","ca","species","conflict_type","response"]
df = raw[keep].copy()

# calendar features
df["month"] = df["date_parsed"].dt.to_period("M").dt.to_timestamp()
df["dow"]   = df["date_parsed"].dt.dayofweek
df["is_weekend"] = df["dow"].isin([5,6]).astype(int)

# normalised conflict & severity
df["conflict_std"] = df["conflict_type"].apply(normalise_conflict)
severity_map = {
    "Human Death": 3,
    "Human Injury": 2,
    "Livestock Predation": 1,
    "Crop Damage": 1,
    "Property Damage": 1,
    "Human Threat": 0,
    "Assessment/Admin": 0,
}
df["severity_level"] = df["conflict_std"].map(severity_map).fillna(1).astype(int)

# repeat incident within 30 days per parish
df = df.sort_values(["parish","date_parsed"])
df["next_incident"] = df.groupby("parish")["date_parsed"].shift(-1)
df["days_to_next"]  = (df["next_incident"] - df["date_parsed"]).dt.days
df["repeat_30d"]    = ((df["days_to_next"].notna()) & (df["days_to_next"] <= 30)).astype(int)
df = df.drop(columns=["next_incident"])

# save a clean copy for reproducibility
CLEAN_PATH = Path("/kaggle/working/data"); CLEAN_PATH.mkdir(parents=True, exist_ok=True)
CLEAN_FILE = CLEAN_PATH / "kasese_hwc_clean_2021_2022.csv"
df.to_csv(CLEAN_FILE, index=False)
print("Clean rows:", df.shape)

# %%
# ======================================================================
# 2) Train/valid/test split by time: 2021 train, 2022 H1 valid, 2022 H2 test
# ======================================================================
work = df.dropna(subset=["date_parsed"]).copy()
work["year"] = work["date_parsed"].dt.year
work["ym"]   = work["date_parsed"].dt.to_period("M").dt.to_timestamp()

train = work[work["year"] == 2021].copy()
valid = work[(work["year"] == 2022) & (work["ym"].dt.month <= 6)].copy()
test  = work[(work["year"] == 2022) & (work["ym"].dt.month >= 7)].copy()

print("Split sizes:", len(train), len(valid), len(test))

# %%
# ======================================================================
# 3) Baseline: multinomial LR (+ Platt calibration), metrics on valid/test
# ======================================================================
target   = "severity_level"
cat_cols = ["district","subcounty","parish","village","ca","species","conflict_std","response"]
num_cols = ["dow","is_weekend"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse=True), cat_cols),
    ("num", "passthrough", num_cols),
])

base_lr = LogisticRegression(max_iter=500, class_weight="balanced", multi_class="auto")
pipe = Pipeline([("pre", pre), ("clf", base_lr)])

X_tr, y_tr = train[cat_cols + num_cols], train[target]
X_va, y_va = valid[cat_cols + num_cols], valid[target]
X_te, y_te = test[cat_cols + num_cols],  test[target]

pipe.fit(X_tr, y_tr)

# uncalibrated valid
probs_va = pipe.predict_proba(X_va)
auc_va   = roc_auc_score(y_va, probs_va, multi_class="ovr", average="macro")
ap_va    = average_precision_score(pd.get_dummies(y_va), probs_va, average="macro")

# calibrate on valid
cal = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv="prefit")
cal.fit(X_va, y_va)
probs_va_cal = cal.predict_proba(X_va)

auc_va_cal = roc_auc_score(y_va, probs_va_cal, multi_class="ovr", average="macro")
ap_va_cal  = average_precision_score(pd.get_dummies(y_va), probs_va_cal, average="macro")

# brier (per-class mean)
classes_sorted = sorted(y_va.unique())
brier_uncal = np.mean([brier_score_loss((y_va==k).astype(int), probs_va[:,k]) for k in classes_sorted])
brier_cal   = np.mean([brier_score_loss((y_va==k).astype(int), probs_va_cal[:,k]) for k in classes_sorted])

print(f"[LR] Valid macro AUC {auc_va:.3f}  AP {ap_va:.3f}")
print(f"[LR] Calibrated Valid macro AUC {auc_va_cal:.3f}  AP {ap_va_cal:.3f}  "
      f"Brier uncal {brier_uncal:.4f} -> cal {brier_cal:.4f}")
print(classification_report(y_va, probs_va_cal.argmax(1), digits=3, zero_division=0))

# test via calibrated LR
probs_te_cal = cal.predict_proba(X_te)
auc_te = roc_auc_score(y_te, probs_te_cal, multi_class="ovr", average="macro")
ap_te  = average_precision_score(pd.get_dummies(y_te), probs_te_cal, average="macro")
print(f"[LR] Test macro AUC {auc_te:.3f}  AP {ap_te:.3f}")
print(classification_report(y_te, probs_te_cal.argmax(1), digits=3, zero_division=0))

# %%
# ======================================================================
# 4) TabTransformer single model (balanced loss) + validation metrics
# ======================================================================
# numeric month index (months since 2020-01)
base_y, base_m = 2020, 1
work["month_ix"] = ((work["month"].dt.year - base_y) * 12
                    + (work["month"].dt.month - base_m)).astype("float32")
num_cols_tt = ["dow","is_weekend","month_ix"]

train_df = work[work["year"] == 2021].copy()
valid_df = work[(work["year"] == 2022) & (work["ym"].dt.month <= 6)].copy()
test_df  = work[(work["year"] == 2022) & (work["ym"].dt.month >= 7)].copy()

# vocab from train only
vocab = {c: {v:i for i,v in enumerate(sorted(
    train_df[c].astype("string").fillna("NA").unique()
))} for c in cat_cols}

def enc_cats(frame):
    return np.stack([
        frame[c].astype("string").fillna("NA").map(vocab[c]).fillna(0).astype(int).values
        for c in cat_cols
    ], axis=1).astype("int64")

mu, sig = train_df[num_cols_tt].mean(), train_df[num_cols_tt].std().replace(0, 1.0)
def enc_nums(frame):
    return ((frame[num_cols_tt].astype(float).fillna(mu) - mu) / sig).values.astype("float32")
def enc_y(frame): return frame[target].astype(int).values

Xc_tr, Xn_tr, y_tr_tt = enc_cats(train_df), enc_nums(train_df), enc_y(train_df)
Xc_va, Xn_va, y_va_tt = enc_cats(valid_df), enc_nums(valid_df), enc_y(valid_df)
Xc_te, Xn_te, y_te_tt = enc_cats(test_df),  enc_nums(test_df),  enc_y(test_df)

class Tds(Dataset):
    def __init__(self, Xc, Xn, y): self.Xc, self.Xn, self.y = Xc, Xn, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.Xc[i]), torch.from_numpy(self.Xn[i]), torch.tensor(self.y[i], dtype=torch.long)

bs = 64
train_dl = DataLoader(Tds(Xc_tr, Xn_tr, y_tr_tt), bs, shuffle=True)
valid_dl = DataLoader(Tds(Xc_va, Xn_va, y_va_tt), bs)
test_dl  = DataLoader(Tds(Xc_te, Xn_te, y_te_tt), bs)

cat_cardinalities = [len(vocab[c]) for c in cat_cols]
n_num      = len(num_cols_tt)
n_classes  = int(max(y_tr_tt.max(), y_va_tt.max(), y_te_tt.max()) + 1)

model = TabTransformer(
    categories=tuple(cat_cardinalities), num_continuous=n_num,
    dim=64, depth=2, heads=4, attn_dropout=0.1, ff_dropout=0.1,
    mlp_hidden_mults=(2,1), mlp_act=nn.ReLU(),
    num_special_tokens=1, continuous_mean_std=None, dim_out=n_classes
).to(DEVICE)

# class-balanced CE
classes = np.arange(n_classes)
w_arr = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr_tt)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(w_arr.astype("float32"), device=DEVICE))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

def run_epoch(dl, train=True, return_logits=False):
    model.train(train)
    tot, n = 0.0, 0
    all_probs, all_y, all_logits = [], [], []
    for Xc, Xn, y in dl:
        Xc, Xn, y = Xc.to(DEVICE), Xn.to(DEVICE), y.to(DEVICE)
        if train: opt.zero_grad()
        logits = model(Xc, Xn)
        loss = criterion(logits, y)
        if train:
            loss.backward(); opt.step()
        tot += float(loss.item()) * y.size(0); n += y.size(0)
        all_logits.append(logits.detach().cpu())
        all_probs.append(torch.softmax(logits,1).detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
    probs = np.vstack(all_probs)
    ys    = np.concatenate(all_y)
    if return_logits:
        return tot/n, probs, ys, torch.cat(all_logits, dim=0).numpy()
    return tot/n, probs, ys

best, bad, patience = 1e9, 0, 8
for epoch in range(50):
    tr_loss, _, _ = run_epoch(train_dl, True)
    va_loss, va_probs, va_y = run_epoch(valid_dl, False)
    sched.step(va_loss)
    print(f"[TT] epoch {epoch+1:02d}  train {tr_loss:.4f}  valid {va_loss:.4f}")
    if va_loss < best - 1e-4:
        best, bad = va_loss, 0
        torch.save(model.state_dict(), "/kaggle/working/tabtr_best.pth")
    else:
        bad += 1
        if bad >= patience: break

model.load_state_dict(torch.load("/kaggle/working/tabtr_best.pth", map_location=DEVICE))

# valid/test metrics
_, va_probs, va_y = run_epoch(valid_dl, False)
_, te_probs, te_y = run_epoch(test_dl, False)

print("[TT] Valid AUC",
      roc_auc_score(va_y, va_probs, multi_class="ovr", average="macro"),
      "AP", average_precision_score(pd.get_dummies(pd.Series(va_y)), va_probs, average="macro"))
print(classification_report(va_y, va_probs.argmax(1), digits=3, zero_division=0))

print("[TT] Test  AUC",
      roc_auc_score(te_y, te_probs, multi_class="ovr", average="macro"),
      "AP", average_precision_score(pd.get_dummies(pd.Series(te_y)), te_probs, average="macro"))
print(classification_report(te_y, te_probs.argmax(1), digits=3, zero_division=0))


# %%
# ======================================================================
# 5) Small ensemble + temperature scaling on valid + conformal sets
# ======================================================================
def build_model():
    return TabTransformer(
        categories=tuple(cat_cardinalities), num_continuous=n_num,
        dim=64, depth=2, heads=4, attn_dropout=0.1, ff_dropout=0.1,
        mlp_hidden_mults=(2,1), mlp_act=nn.ReLU(),
        num_special_tokens=1, continuous_mean_std=None, dim_out=n_classes
    ).to(DEVICE)

def train_one(model, Xc_tr, Xn_tr, y_tr, Xc_va, Xn_va, y_va,
              class_weights, lr=1e-3, wd=1e-4, max_epochs=40, patience=6):
    w = torch.tensor(class_weights.astype("float32"), device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched= torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.5, 3)

    def run_np(Xc, Xn, y, train=True, bs=64, return_logits=False):
        model.train(train)
        tot, n = 0.0, 0
        P, Y, L = [], [], []
        for i in range(0, len(y), bs):
            xb_c = torch.from_numpy(Xc[i:i+bs]).to(DEVICE)
            xb_n = torch.from_numpy(Xn[i:i+bs]).to(DEVICE)
            yb   = torch.from_numpy(y[i:i+bs]).long().to(DEVICE)
            if train: opt.zero_grad()
            logits = model(xb_c, xb_n); loss = crit(logits, yb)
            if train: loss.backward(); opt.step()
            tot += float(loss.item())*yb.size(0); n += yb.size(0)
            P.append(torch.softmax(logits,1).detach().cpu().numpy())
            Y.append(yb.cpu().numpy()); L.append(logits.detach().cpu().numpy())
        probs = np.vstack(P); yy = np.concatenate(Y); ll = np.vstack(L)
        if return_logits: return tot/n, probs, yy, ll
        return tot/n, probs, yy

    best, bad = 1e9, 0
    for ep in range(max_epochs):
        tr_loss, _, _ = run_np(Xc_tr, Xn_tr, y_tr, True)
        va_loss, _, _ = run_np(Xc_va, Xn_va, y_va, False)
        sched.step(va_loss)
        print(f"  [ens] epoch {ep+1:02d}  train {tr_loss:.4f}  valid {va_loss:.4f}")
        if va_loss < best - 1e-4:
            best, bad = va_loss, 0
            torch.save(model.state_dict(), "/kaggle/working/tabtr_best_tmp.pth")
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(torch.load("/kaggle/working/tabtr_best_tmp.pth", map_location=DEVICE))
    return model

classes = np.arange(n_classes)
class_weights = compute_class_weight("balanced", classes=classes, y=y_tr_tt)

seeds = [11,22,33,44,55]
ens = []
for i, s in enumerate(seeds, 1):
    print(f"\n=== ensemble member {i}/{len(seeds)} (seed {s}) ===")
    torch.manual_seed(s); np.random.seed(s); random.seed(s)
    m = build_model()
    m = train_one(m, Xc_tr, Xn_tr, y_tr_tt, Xc_va, Xn_va, y_va_tt, class_weights)
    ens.append(m)

@torch.no_grad()
def ensemble_logits(models, Xc, Xn, bs=64):
    """Return stacked logits per model (for temperature scaling)."""
    logits_per_model = []
    for mdl in models:
        L = []
        for i in range(0, len(Xc), bs):
            xb_c = torch.from_numpy(Xc[i:i+bs]).to(DEVICE)
            xb_n = torch.from_numpy(Xn[i:i+bs]).to(DEVICE)
            L.append(mdl(xb_c, xb_n).cpu().numpy())
        logits_per_model.append(np.vstack(L))
    return np.stack(logits_per_model, axis=0)  # [M, N, C]

logits_va_M = ensemble_logits(ens, Xc_va, Xn_va)  # [M,N,C]
logits_te_M = ensemble_logits(ens, Xc_te, Xn_te)

# average logits, then softmax -> probabilities
def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

logits_va_mean = logits_va_M.mean(axis=0)
logits_te_mean = logits_te_M.mean(axis=0)

# ----- Temperature scaling on valid (optimize scalar T) -----
def fit_temperature(logits, y):
    T = torch.ones(1, requires_grad=True, device=DEVICE)
    y_t = torch.from_numpy(y).long().to(DEVICE)
    logit_t = torch.from_numpy(logits).float().to(DEVICE)
    optT = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def _loss():
        optT.zero_grad()
        scaled = logit_t / T.clamp_min(1e-3)
        loss = nn.CrossEntropyLoss()(scaled, y_t)
        loss.backward()
        return loss

    optT.step(_loss)
    return float(T.detach().cpu().item())

T_star = fit_temperature(logits_va_mean, y_va_tt)
print("Temperature T* =", T_star)

p_va_mean = softmax_np(logits_va_mean / T_star)
p_te_mean = softmax_np(logits_te_mean / T_star)

def entropy(p):
    p = np.clip(p, 1e-12, 1); return -(p*np.log(p)).sum(axis=1)

print("valid mean entropy", float(entropy(p_va_mean).mean()))
print("test  mean entropy", float(entropy(p_te_mean).mean()))

# ----- Split conformal sets on valid, evaluate on test -----
def split_conformal_threshold(probs_cal, y_cal, alpha=0.10):
    s = 1.0 - probs_cal[np.arange(len(y_cal)), y_cal]
    n = len(s)
    q = np.quantile(s, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")
    return float(q)

qhat = split_conformal_threshold(p_va_mean, y_va_tt, alpha=0.10)
S_te = p_te_mean >= (1.0 - qhat)
emp_cov = (S_te[np.arange(len(y_te_tt)), y_te_tt]).mean()
med_size = float(np.median(S_te.sum(axis=1)))
print(f"Conformal qhat={qhat:.3f}  test coverage={emp_cov:.3f}  median set size={med_size:.1f}")

# %%
# ======================================================================
# 6) Action normalisation (robust) + uplift (multi-arm T-learner)
# ======================================================================
# success label: no repeat within 30 days
df_u = df.copy()
df_u["y_success"] = 1 - df_u["repeat_30d"].astype(int)

# time split for uplift too
df_u["year"] = df_u["date_parsed"].dt.year
df_u["ym"]   = df_u["date_parsed"].dt.to_period("M").dt.to_timestamp()
train_u = df_u[df_u["year"] == 2021].copy()
valid_u = df_u[(df_u["year"] == 2022) & (df_u["ym"].dt.month <= 6)].copy()
test_u  = df_u[(df_u["year"] == 2022) & (df_u["ym"].dt.month >= 7)].copy()

def norm_text(s):
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return s

P_SCARE   = [r"\bscare ?shoot", r"\bscares? ?shoot", r"\bsscare shoot", r"\bscareshooting"]
P_LOCAL   = [r"local methods", r"scaring using local methods", r"\bdrum", r"\bvuvuz", r"camp fire"]
P_ASSESS  = [r"assessment", r"asses+e?ment", r"\bassessment carried out\b", r"pam assessment", r"assessment carried out by"]
P_MEDICAL = [r"taken to .*hospital", r"taken to hospital", r"taken to bwera hospital"]
P_CAPTURE = [r"\bcaptured\b", r"\bcuptured\b", r"\brescued\b", r"\btranslocated\b"]
P_SENS    = [r"sensiti[sz]ation", r"meeting with bee keepers"]
P_COMP    = [r"\bcompensa?tion\b", r"compans?ionate|compansionate|compansetion"]
P_POST    = [r"post ?mortem|postmoterm|body (not found|missed)"]

def map_action(x):
    t = norm_text(x)
    if any(re.search(p, t) for p in P_SCARE):   return "scare_shooting"
    if any(re.search(p, t) for p in P_LOCAL):   return "local_scaring"
    if any(re.search(p, t) for p in P_ASSESS):  return "assessment"
    if any(re.search(p, t) for p in P_MEDICAL): return "medical"
    if any(re.search(p, t) for p in P_CAPTURE): return "capture_move"
    if any(re.search(p, t) for p in P_SENS):    return "sensitisation"
    if any(re.search(p, t) for p in P_COMP):    return "compensation"
    if any(re.search(p, t) for p in P_POST):    return "postmortem"
    return "other"

for part in (train_u, valid_u, test_u):
    part["treat"] = part["response"].apply(map_action)

# features for uplift outcome models
cat_u = ["district","subcounty","parish","village","ca","species","conflict_std","treat"]
num_u = ["dow","is_weekend"]
feat_u = cat_u + num_u

def Xyt(frame):
    X = frame[feat_u].copy()
    y = frame["y_success"].astype(int).values
    t = frame["treat"].astype("category")
    return X, y, t

X_tr_u, y_tr_u, t_tr_u = Xyt(train_u)
X_va_u, y_va_u, t_va_u = Xyt(valid_u)
X_te_u, y_te_u, t_te_u = Xyt(test_u)

X_tr_ready = pd.get_dummies(X_tr_u, columns=cat_u, dummy_na=True)
X_va_ready = pd.get_dummies(X_va_u, columns=cat_u, dummy_na=True).reindex(columns=X_tr_ready.columns, fill_value=0)
X_te_ready = pd.get_dummies(X_te_u, columns=cat_u, dummy_na=True).reindex(columns=X_tr_ready.columns, fill_value=0)

t_names = list(t_tr_u.cat.categories); K = len(t_names)
print("treatments:", t_names)

def make_xgb(seed=42):
    return XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", n_jobs=-1, random_state=seed
    )

class ConstantProb:
    def __init__(self, p): self.p = float(p)
    def fit(self, X, y, sample_weight=None): return self
    def predict_proba(self, X):
        n = X.shape[0]; p1 = np.full(n, self.p, dtype=np.float32); p0 = 1. - p1
        return np.stack([p0, p1], 1)

def fit_bin(X, y, seed, w=None):
    y = np.asarray(y)
    if np.unique(y).size == 1:
        return ConstantProb(float(y.mean()))
    clf = make_xgb(seed=seed)
    clf.fit(X, y, sample_weight=w)
    return clf

# plain (unweighted) multi-arm T-learner
arm_models_plain = []
for k, arm in enumerate(t_names):
    is_arm = (t_tr_u.values == arm)
    mt = fit_bin(X_tr_ready[is_arm].values,  y_tr_u[is_arm],  100+k)
    mc = fit_bin(X_tr_ready[~is_arm].values, y_tr_u[~is_arm], 200+k)
    arm_models_plain.append((mt, mc))

def predict_uplift(arm_models, X_df):
    X = X_df.values if hasattr(X_df, "values") else X_df
    ups = []
    for mt, mc in arm_models:
        p_t = mt.predict_proba(X)[:,1]; p_c = mc.predict_proba(X)[:,1]
        ups.append(p_t - p_c)
    return np.vstack(ups).T  # [N, K]

uplift_va = predict_uplift(arm_models_plain, X_va_ready)
uplift_te = predict_uplift(arm_models_plain, X_te_ready)

# AUUC-above-random (Radcliffe-style)
def uplift_curve(y, uplift, treat_indicator):
    order = np.argsort(-uplift)
    y_ord = y[order]; t_ord = treat_indicator[order]
    cum_t = np.cumsum(t_ord); cum_c = np.cumsum(1 - t_ord)
    rate_t = np.cumsum(y_ord * t_ord) / np.maximum(cum_t, 1)
    rate_c = np.cumsum(y_ord * (1 - t_ord)) / np.maximum(cum_c, 1)
    gain = cum_t * (rate_t - rate_c)
    return gain

def auuc_above_random(y, uplift, t_ind):
    g = uplift_curve(y, uplift, t_ind)
    x = np.linspace(0, 1, len(g)); g_rand = g[-1] * x
    return float(np.trapz(g - g_rand, x))

# quick report for the most common arm in train
focal = pd.Series(t_tr_u.values).value_counts().index[0]
kf = t_names.index(focal)
print("Valid AUUC above random (plain) for", focal, ":",
      auuc_above_random(y_va_u, uplift_va[:, kf], (t_va_u.values==focal).astype(int)))

# Parish-level recommended action (plain T-learner)
best_idx = uplift_te.argmax(1)
policy_plain = test_u[["parish","subcounty","species","conflict_std"]].copy()
policy_plain["best_action"] = [t_names[i] for i in best_idx]
policy_plain["pred_gain_success"] = uplift_te.max(1)

rec_by_parish = (
    policy_plain.groupby(["parish","subcounty"])
      .agg(best_action=("best_action", lambda s: s.value_counts().idxmax()),
           mean_pred_gain=("pred_gain_success","mean"),
           n_cases=("best_action","size"))
      .reset_index()
      .sort_values(["n_cases","mean_pred_gain"], ascending=[False, False])
)
rec_by_parish.head()

# %%
# ======================================================================
# 7) Causal sanity: multinomial propensities + IPW T-learner + DR + overlap IPS
# ======================================================================
# features NOT including treatment for propensity
prop_cat = ["district","subcounty","parish","village","ca","species","conflict_std"]
prop_num = ["dow","is_weekend"]
X_prop_tr = pd.get_dummies(train_u[prop_cat + prop_num], columns=prop_cat, dummy_na=True)
X_prop_va = pd.get_dummies(valid_u[prop_cat + prop_num], columns=prop_cat, dummy_na=True).reindex(columns=X_prop_tr.columns, fill_value=0)
X_prop_te = pd.get_dummies(test_u [prop_cat + prop_num], columns=prop_cat, dummy_na=True).reindex(columns=X_prop_tr.columns, fill_value=0)

t_tr_codes = train_u["treat"].astype("category").cat.codes.values
t_va_codes = valid_u["treat"].astype("category").cat.codes.values
t_te_codes = test_u ["treat"].astype("category").cat.codes.values
t_names = list(train_u["treat"].astype("category").cat.categories)
K = len(t_names)

prop_clf = LogisticRegression(max_iter=400, multi_class="multinomial", solver="lbfgs")
prop_clf.fit(X_prop_tr, t_tr_codes)
prop_tr = np.clip(prop_clf.predict_proba(X_prop_tr), 1e-2, 1.0); prop_tr /= prop_tr.sum(1, keepdims=True)
prop_va = np.clip(prop_clf.predict_proba(X_prop_va), 1e-2, 1.0); prop_va /= prop_va.sum(1, keepdims=True)
prop_te = np.clip(prop_clf.predict_proba(X_prop_te), 1e-2, 1.0); prop_te /= prop_te.sum(1, keepdims=True)

# outcome features include one-hot of treat (as fitted earlier)
X_tr_out = pd.get_dummies(train_u[feat_u], columns=cat_u, dummy_na=True)
X_va_out = pd.get_dummies(valid_u[feat_u], columns=cat_u, dummy_na=True).reindex(columns=X_tr_out.columns, fill_value=0)
X_te_out = pd.get_dummies(test_u [feat_u], columns=cat_u, dummy_na=True).reindex(columns=X_tr_out.columns, fill_value=0)
y_tr = train_u["y_success"].astype(int).values
y_va = valid_u["y_success"].astype(int).values
y_te = test_u ["y_success"].astype(int).values

# IPW: fit per arm
arm_models_ipw = []
t_tr_vals = train_u["treat"].astype("category").values
for k, arm in enumerate(t_names):
    is_arm = (t_tr_vals == arm)
    X_t, y_t = X_tr_out[is_arm].values, y_tr[is_arm]
    w_t = 1.0 / prop_tr[is_arm, k]
    X_c, y_c = X_tr_out[~is_arm].values, y_tr[~is_arm]
    w_c = 1.0 / (1.0 - prop_tr[~is_arm, k])
    mt = fit_bin(X_t, y_t, 300+k, w=w_t)
    mc = fit_bin(X_c, y_c, 400+k, w=w_c)
    arm_models_ipw.append((mt, mc))

uplift_va_ipw = predict_uplift(arm_models_ipw, X_va_out)
uplift_te_ipw = predict_uplift(arm_models_ipw, X_te_out)

# per-arm AUUC-above-random on valid
scores = []
t_va_vals = valid_u["treat"].astype(str).values
for k, arm in enumerate(t_names):
    t_ind = (t_va_vals == arm).astype(int)
    scores.append((arm, auuc_above_random(y_va, uplift_va_ipw[:,k], t_ind)))
scores_df = pd.DataFrame(scores, columns=["arm","auuc_above_random"]).sort_values("auuc_above_random", ascending=False)
print(scores_df)

# policy under IPW uplift: choose argmax
pi_star_idx = uplift_te_ipw.argmax(axis=1)
# simple IPS on test
match = (t_te_codes == pi_star_idx).astype(int)
p_logged = prop_te[np.arange(len(prop_te)), t_te_codes]
ips = np.mean(match * y_te / np.clip(p_logged, 1e-3, 1.0))

# model-based value: E[success | choose arm] via outcome model mt
def outcome_value(arm_models, X_df, chosen_idx):
    X = X_df.values
    vals = []
    for i in range(len(X)):
        mt, mc = arm_models[chosen_idx[i]]
        vals.append(mt.predict_proba(X[i:i+1])[:,1][0])
    return float(np.mean(vals))

v_hat = outcome_value(arm_models_ipw, X_te_out, pi_star_idx)

# doubly robust estimate
def doubly_robust(arm_models, X_df, y, logged_idx, chosen_idx, prop):
    X = X_df.values; N = len(y); est = np.zeros(N, dtype=float)
    for i in range(N):
        mt, mc = arm_models[chosen_idx[i]]
        mu = mt.predict_proba(X[i:i+1])[:,1][0]
        if logged_idx[i] == chosen_idx[i]:
            w = 1.0 / np.clip(prop[i, logged_idx[i]], 1e-3, 1.0)
            est[i] = mu + w*(y[i]-mu)
        else:
            est[i] = mu
    return float(np.mean(est))

dr = doubly_robust(arm_models_ipw, X_te_out, y_te, t_te_codes, pi_star_idx, prop_te)

# overlap-weighted IPS (downweight extreme propensities)
ov_w = prop_te[np.arange(len(prop_te)), t_te_codes] * (1.0 - prop_te[np.arange(len(prop_te)), t_te_codes])
ips_ov = np.sum(ov_w * match * y_te / np.clip(p_logged, 1e-3, 1.0)) / np.sum(ov_w)

print(f"IPS={ips:.3f}  Overlap-IPS={ips_ov:.3f}  Model-value={v_hat:.3f}  DR={dr:.3f}")

# %%
# ======================================================================
# 8) Interpretability & diagnostics
# ======================================================================
# 8.1 Logistic: most influential features per class
ohe = pipe.named_steps["pre"].named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(cat_cols).tolist()
feature_names = cat_names + num_cols
coef = pipe.named_steps["clf"].coef_  # [C, D]
classes_lr = pipe.named_steps["clf"].classes_

rows = []
for ci, c in enumerate(classes_lr):
    w = coef[ci]
    idx = np.argsort(np.abs(w))[::-1][:25]
    for j in idx:
        rows.append({"class": int(c), "feature": feature_names[j], "weight": float(w[j])})
lr_top = pd.DataFrame(rows)
lr_top.to_csv(OUTDIR / "logistic_top_weights.csv", index=False)

# 8.2 TabTransformer permutation importance (one ensemble model, for speed)
def valid_loss(model, Xc, Xn, y, bs=128):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            xb_c = torch.from_numpy(Xc[i:i+bs]).to(DEVICE)
            xb_n = torch.from_numpy(Xn[i:i+bs]).to(DEVICE)
            yb   = torch.from_numpy(y[i:i+bs]).long().to(DEVICE)
            logits = model(xb_c, xb_n)
            loss = nn.functional.cross_entropy(logits, yb)
            tot += float(loss.item()) * yb.size(0); n += yb.size(0)
    return tot/n

m0 = ens[0]
base_loss = valid_loss(m0, Xc_va, Xn_va, y_va_tt)

def perm_importance_cat(model, Xc, Xn, y, col_ix, reps=5):
    losses = []
    for _ in range(reps):
        Xc_perm = Xc.copy()
        Xc_perm[:, col_ix] = np.random.permutation(Xc_perm[:, col_ix])
        losses.append(valid_loss(model, Xc_perm, Xn, y))
    return float(np.mean(losses) - base_loss)

def perm_importance_num(model, Xc, Xn, y, col_ix, reps=5):
    losses = []
    for _ in range(reps):
        Xn_perm = Xn.copy()
        Xn_perm[:, col_ix] = np.random.permutation(Xn_perm[:, col_ix])
        losses.append(valid_loss(model, Xc, Xn_perm, y))
    return float(np.mean(losses) - base_loss)

imp_rows = []
for i, c in enumerate(cat_cols):
    imp_rows.append({"feature_group": c, "perm_loss_increase": perm_importance_cat(m0, Xc_va, Xn_va, y_va_tt, i)})
for j, c in enumerate(num_cols_tt):
    imp_rows.append({"feature_group": c, "perm_loss_increase": perm_importance_num(m0, Xc_va, Xn_va, y_va_tt, j)})

tt_importance = pd.DataFrame(imp_rows).sort_values("perm_loss_increase", ascending=False)
tt_importance.to_csv(OUTDIR / "tabtransformer_perm_importance.csv", index=False)

# 8.3 Reliability diagram (top-class prob vs empirical accuracy) on valid
def top_p_and_true(probs, y):
    top = probs.max(1); pred = probs.argmax(1); correct = (pred == y).astype(int)
    return top, correct

# LR calibrated top p
p_lr = cal.predict_proba(X_va).max(axis=1)
y_corr_lr = (cal.predict(X_va) == y_va).astype(int)

# TT ensemble top p (after temperature scaling)
p_tt = p_va_mean.max(axis=1); y_corr_tt = (p_va_mean.argmax(1) == y_va_tt).astype(int)

def plot_reliability(p, corr, label, ax, n_bins=8):
    # uniform bins on confidence
    bins = np.linspace(0, 1, n_bins+1)
    mids = 0.5*(bins[1:] + bins[:-1])
    accs, confs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (p >= lo) & (p < hi)
        if idx.sum() == 0: 
            accs.append(np.nan); confs.append(mids[len(confs)]); 
        else:
            accs.append(corr[idx].mean()); confs.append(p[idx].mean())
    ax.plot(confs, accs, marker="o", label=label)

fig, ax = plt.subplots(figsize=(5,4))
plot_reliability(p_lr, y_corr_lr, "logistic calibrated", ax)
plot_reliability(p_tt, y_corr_tt, "tabtransformer ensemble", ax)
ax.plot([0,1],[0,1],"--",alpha=0.5)
ax.set_xlabel("predicted top-class probability")
ax.set_ylabel("empirical accuracy")
ax.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "reliability_valid.png", dpi=160)

# %%
# ======================================================================
# 9) Exports for report
# ======================================================================
# Policy table on 2022 H2 (from IPW uplift)
policy_df = test_u[["parish","subcounty","species","conflict_std","treat","y_success"]].copy()
policy_df["chosen_action"] = [t_names[i] for i in pi_star_idx]
policy_df["agree_logged"]  = (policy_df["chosen_action"] == test_u["treat"]).astype(int)
policy_df["pred_top_p"]    = p_te_mean.max(axis=1)
policy_df["entropy"]       = entropy(p_te_mean)

policy_df.to_csv(OUTDIR / "policy_kasese_2022H2.csv", index=False)
rec_by_parish.to_csv(OUTDIR / "uplift_recommendations_by_parish.csv", index=False)

print("\nSaved files:")
for f in OUTDIR.iterdir():
    print(" -", f)

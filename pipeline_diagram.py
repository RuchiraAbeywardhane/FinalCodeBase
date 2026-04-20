import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(18, 26))
ax.set_xlim(0, 18); ax.set_ylim(0, 26); ax.axis("off")
fig.patch.set_facecolor("#0f0f1a")

C = {"data":"#1e3a5f","feat":"#1a4731","proc":"#3b2a5e","nn":"#5e1a1a",
     "grl":"#7a3800","lda":"#1a4040","out":"#2d2d00","border":"#aaaacc",
     "text":"#f0f0f0","arrow":"#88aaff","accent":"#ffcc44","green":"#44ff88","red":"#ff4466"}

def box(ax,x,y,w,h,label,sublabel="",color="#1e3a5f",fontsize=11,accent=False):
    r=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",linewidth=1.5,
        edgecolor=C["accent"] if accent else C["border"],facecolor=color,zorder=3)
    ax.add_patch(r)
    cy=y+h/2+(0.15 if sublabel else 0)
    ax.text(x+w/2,cy,label,ha="center",va="center",color=C["text"],fontsize=fontsize,fontweight="bold",zorder=4)
    if sublabel:
        ax.text(x+w/2,y+h/2-0.22,sublabel,ha="center",va="center",color="#bbbbbb",fontsize=8.5,zorder=4)

def arrow(ax,x1,y1,x2,y2,color=None):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle="-|>",color=color or C["arrow"],lw=2.0,mutation_scale=18),zorder=5)

def sl(ax,x,y,t):
    ax.text(x,y,t,color="#888888",fontsize=9,fontstyle="italic",zorder=4)

# TITLE
ax.text(9,25.3,"DANN + LDA  -  Full Pipeline",ha="center",color=C["accent"],fontsize=16,fontweight="bold")
ax.text(9,24.85,"Domain Adversarial Neural Network for Cross-Subject EEG Emotion Recognition",ha="center",color="#aaaaaa",fontsize=10)

# 1 RAW DATA
sl(ax,0.3,24.45,"1  RAW DATA  (per trial, per subject)")
box(ax,0.5,23.5,4.8,0.7,"EEG  (4ch @ 256 Hz)","RAW_TP9 / AF7 / AF8 / TP10",C["data"])
box(ax,6.0,23.5,4.8,0.7,"Band Powers (20ch @ 10Hz)","Alpha/Beta/Theta/Delta/Gamma",C["data"])
box(ax,11.5,23.5,5.5,0.7,"BVP  (1ch @ 20 Hz)","Samsung Watch PPG",C["data"])

# 2 BASELINE
sl(ax,0.3,23.2,"2  BASELINE REDUCTION  +  TIME ALIGNMENT")
box(ax,3.0,22.3,12,0.7,
    "Per-subject z-score (EEG)  +  Truncate to shortest common duration",
    "baseline=BASELINE_MUSE_cleaned.json   |   dur=min(EEG, MUSE, BVP)",C["proc"])
arrow(ax,2.9,23.5,4.5,23.0); arrow(ax,8.4,23.5,9.0,23.0); arrow(ax,14.2,23.5,13.5,23.0)
arrow(ax,9,22.3,9,22.02)

# 3 FEATURE EXTRACTION
sl(ax,0.3,21.95,"3  FEATURE EXTRACTION   ->   262 features per window")
items=[
    ("EEG\n156 feat",  "band power, DE\nHjorth, wavelet\ncoherence, ZCR", 2.5),
    ("MUSE\n62 feat",  "mean/std per\nband-channel\na/b/t ratios",        2.5),
    ("BVP\n7 feat",    "stats + dominant\nHR frequency",                  2.5),
    ("HR\n5 feat",     "mean/std/min\nmax/range",                         2.3),
    ("PPI\n8 feat",    "RMSSD pNN50\nLF/HF HRV",                         2.3),
    ("FAA\n14 feat",   "frontal alpha\nasymmetry",                        2.3),
    ("Riem\n10 feat",  "log-cov upper\ntriangle SPD",                     2.3),
]
bx=0.4
for lbl,sub,w in items:
    box(ax,bx,20.7,w-0.1,1.1,lbl,sub,C["feat"],fontsize=9); bx+=w
arrow(ax,9,22.3,9,21.82)

# 4 EUCLIDEAN ALIGNMENT
sl(ax,0.3,20.45,"4  EUCLIDEAN ALIGNMENT  (Riemannian domain adaptation)")
box(ax,2.5,19.55,13,0.75,
    "Per-subject EA:  R = mean SPD cov   ->   aligned = R^(-1/2) @ window",
    "Re-centres each subject on Riemannian manifold  |  overwrites Riem columns",C["proc"])
arrow(ax,9,20.7,9,20.32)

# 5 PREPROCESSING
sl(ax,0.3,19.3,"5  LOSO PREPROCESSING  (fit on training subjects only - no leakage)")
box(ax,1.0,18.35,7.0,0.75,"VarianceThreshold","drop near-constant features  threshold=0.001",C["proc"])
box(ax,9.5,18.35,7.0,0.75,"Global z-score","mean/std of all train windows  clip[-10,+10]",C["proc"])
arrow(ax,9,19.55,4.5,19.12); arrow(ax,9,19.55,13.0,19.12)
arrow(ax,4.5,18.35,7.0,18.72); arrow(ax,13.0,18.35,11.0,18.72)
box(ax,7.0,17.55,4.0,0.65,"allF  (N_windows x D_selected)","",C["proc"],fontsize=9)
arrow(ax,9.0,18.35,9.0,18.2)

# 6 DANN
sl(ax,0.3,17.2,"6  DANN  -  Domain Adversarial Neural Network  (GPU)")
ax.add_patch(FancyBboxPatch((0.4,12.6),17.2,4.45,boxstyle="round,pad=0.1",
    linewidth=2,edgecolor=C["accent"],facecolor="#1a1a2e",zorder=2))
arrow(ax,9.0,17.55,9.0,17.07)

box(ax,5.5,15.9,7.0,0.95,"Feature Extractor  (MLP)",
    "Linear(D->256)->BN->ELU->Drop(0.4)  |  Linear(256->128)->BN->ELU->Drop(0.3)  |  Linear(128->64)->BN->ELU",
    C["nn"],fontsize=10,accent=True)
arrow(ax,9.0,17.0,9.0,16.87)

box(ax,6.2,15.1,5.6,0.65,"Bottleneck  (64-dim invariant features)","",C["nn"],fontsize=10,accent=True)
arrow(ax,9.0,15.9,9.0,15.77)

arrow(ax,7.5,15.1,4.2,14.5,color=C["green"])
arrow(ax,10.5,15.1,13.8,14.5,color=C["red"])

box(ax,1.0,13.55,5.8,0.82,"Emotion Head",
    "Linear(64->32)->ELU->Linear(32->4)\n-> CrossEntropyLoss  (maximise)","#1a4a1a",fontsize=10)
box(ax,11.2,13.55,5.8,0.82,"Subject Head + GRL",
    "GradReverse(lam)->Linear(64->64)->ELU\n->Linear(64->41)->CE  (minimise)","#4a1a1a",fontsize=10)

ax.text(9.0,14.72,"Gradient Reversal Layer  (lam annealed 0 -> 1)",
    ha="center",color=C["accent"],fontsize=9,fontstyle="italic")
box(ax,6.3,12.75,5.4,0.6,"lam = lam_max * (2/(1+exp(-10p)) - 1)   p=epoch/total",
    "ramps 0 -> 1.0  -  stabilises emotion training first",C["grl"],fontsize=8.5)
ax.text(9.0,13.3,"Total loss  =  L_emotion  +  lam * L_subject",
    ha="center",color="#ffcc44",fontsize=10,fontweight="bold")
ax.text(1.5,13.1,"wants to\nclassify emotion",color=C["green"],fontsize=8.5,ha="center")
ax.text(14.5,13.1,"wants to FOOL\nsubject classifier",color=C["red"],fontsize=8.5,ha="center")

# 7 EARLY STOPPING
sl(ax,0.3,12.45,"7  EARLY STOPPING  (patience=15 on validation emotion loss)")
box(ax,3.5,11.55,11.0,0.7,
    "Best checkpoint  ->  restore weights  ->  extract bottleneck features",
    "val split = last 2 training subjects  (no leakage to held-out subject)",C["proc"])
arrow(ax,9.0,12.6,9.0,12.27)

# 8 LDA
sl(ax,0.3,11.3,"8  LDA  on DANN bottleneck features")
box(ax,4.5,10.4,9.0,0.75,
    "LinearDiscriminantAnalysis  (solver=lsqr, shrinkage=auto)",
    "fit on 64-dim train bottleneck  |  predict on test bottleneck",C["lda"],accent=True)
arrow(ax,9.0,11.55,9.0,11.17)

# 9 TRIAL VOTE
sl(ax,0.3,10.15,"9  TRIAL-LEVEL AGGREGATION")
box(ax,4.0,9.25,10.0,0.75,
    "Exponential Weighted Vote  (e^0 -> e^1 ramp)",
    "later windows weighted up to 2.7x  -  emotion peaks toward video end",C["lda"])
arrow(ax,9.0,10.4,9.0,10.02)

# 10 OUTPUT
sl(ax,0.3,8.98,"10  OUTPUT")
box(ax,1.0,8.1,7.5,0.75,"Window-level predictions","dann_lda_loso_window.csv",C["out"])
box(ax,9.5,8.1,7.5,0.75,"Trial-level predictions","dann_lda_loso_trial.csv",C["out"])
arrow(ax,7.5,9.25,4.5,8.87); arrow(ax,10.5,9.25,13.0,8.87)

# LEGEND
leg=[(C["data"],"Raw data"),(C["feat"],"Feature extraction"),(C["proc"],"Preprocessing"),
     (C["nn"],"Neural network (GPU)"),(C["grl"],"GRL/adversarial"),(C["lda"],"LDA/aggregation"),(C["out"],"Output")]
lx=0.5
for col,lbl in leg:
    ax.add_patch(FancyBboxPatch((lx,7.35),0.35,0.28,boxstyle="round,pad=0.03",
        facecolor=col,edgecolor=C["border"],linewidth=1))
    ax.text(lx+0.45,7.49,lbl,color=C["text"],fontsize=8.5,va="center"); lx+=2.45

plt.tight_layout()
plt.savefig("/kaggle/working/dann_lda_pipeline.png",dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
print("Saved: /kaggle/working/dann_lda_pipeline.png")
plt.show()

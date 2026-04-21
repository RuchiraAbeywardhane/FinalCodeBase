
import re

with open('AddedRiemannFeatures.py', 'r', encoding='utf-8') as f:
    content = f.read()

# ── PART 1: MiniLCM config + class + helpers (inserted after constants block) ──
MINILCM_BLOCK = '''
# ═══════════════════════════════════════════════════════════════
# MINILCM CONFIG
# ═══════════════════════════════════════════════════════════════
USE_MINILCM     = True        # set False to skip the neural add-on
MINILCM_EPOCHS  = 30
MINILCM_LR      = 1e-3
MINILCM_BATCH   = 64
MINILCM_EMB_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════
# MINILCM  -- CNN + Transformer EEG encoder
# ═══════════════════════════════════════════════════════════════
class MiniLCM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, padding=7)
        enc = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                          batch_first=True, dropout=0.1)
        self.tr  = nn.TransformerEncoder(enc, num_layers=2)
        self.fc  = nn.Linear(64, MINILCM_EMB_DIM)
        self.cls = nn.Linear(MINILCM_EMB_DIM, num_classes)

    def embed(self, x):          # (B,4,T) -> (B,64)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2)
        x = self.tr(x)
        return self.fc(x.mean(dim=1))

    def forward(self, x):
        return self.cls(self.embed(x))


def train_minilcm(eeg_windows, labels, train_mask, val_mask):
    model = MiniLCM().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=MINILCM_LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MINILCM_EPOCHS)
    crit  = nn.CrossEntropyLoss()
    X_tr  = torch.tensor(eeg_windows[train_mask], dtype=torch.float32)
    y_tr  = torch.tensor(labels[train_mask],       dtype=torch.long)
    X_vl  = torch.tensor(eeg_windows[val_mask],    dtype=torch.float32)
    y_vl  = torch.tensor(labels[val_mask],          dtype=torch.long)
    dl_tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=MINILCM_BATCH, shuffle=True)
    best_val, best_state = -1, None
    for ep in range(1, MINILCM_EPOCHS + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        if ep % 5 == 0 or ep == MINILCM_EPOCHS:
            model.eval()
            with torch.no_grad():
                preds = torch.cat([model(c).argmax(1)
                    for c in torch.split(X_vl.to(DEVICE), MINILCM_BATCH)]).cpu()
            val_acc = (preds == y_vl).float().mean().item()
            if val_acc > best_val:
                best_val   = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    [MiniLCM] ep={ep:3d}  val_acc={val_acc:.4f}  best={best_val:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def extract_minilcm_embeddings(model, eeg_windows):
    model.eval()
    X = torch.tensor(eeg_windows, dtype=torch.float32)
    embs = []
    with torch.no_grad():
        for (xb,) in torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X), batch_size=MINILCM_BATCH):
            embs.append(model.embed(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)

'''

# Insert after the constants print line
ANCHOR1 = 'print(f"Raw feature count = {N_FEATURES_RAW}")'
assert ANCHOR1 in content, "ANCHOR1 not found"
content = content.replace(ANCHOR1, ANCHOR1 + MINILCM_BLOCK, 1)
print("Part 1 inserted: MiniLCM class/helpers")

# ── PART 2: collect raw EEG windows in windowing loop ──
# Add  all_eeg_w = []  alongside the other lists
OLD2 = "all_feat_w = []\nall_labels_w = []\nall_tkeys_w = []\nall_sids_w = []\nall_tidx_w = []"
NEW2 = "all_feat_w = []\nall_labels_w = []\nall_tkeys_w = []\nall_sids_w = []\nall_tidx_w = []\nall_eeg_w  = []           # raw EEG windows for MiniLCM"
assert OLD2 in content, "ANCHOR2 not found"
content = content.replace(OLD2, NEW2, 1)
print("Part 2a inserted: all_eeg_w list")

# Append  all_eeg_w.append(ew)  after  all_tidx_w.append(ti)  inside the loop
OLD3 = "        all_tidx_w.append(ti)\n\nallF_raw"
NEW3 = "        all_tidx_w.append(ti)\n        all_eeg_w.append(ew)       # store raw EEG window\n\nallF_raw"
assert OLD3 in content, "ANCHOR3 not found"
content = content.replace(OLD3, NEW3, 1)
print("Part 2b inserted: all_eeg_w.append(ew)")

# Stack into array after allTI
OLD4 = "allTI    = np.array(all_tidx_w)\n\nprint(f\"Training windows:"
NEW4 = "allTI    = np.array(all_tidx_w)\nallEEG_w = np.stack(all_eeg_w, axis=0).astype(np.float32)  # (N,4,EEG_WIN)\n\nprint(f\"Training windows:"
assert OLD4 in content, "ANCHOR4 not found"
content = content.replace(OLD4, NEW4, 1)
print("Part 2c inserted: allEEG_w stack")

# ── PART 3: train MiniLCM + concat embeddings (before per-fold MI) ──
MINILCM_TRAIN_BLOCK = '''
# ═══════════════════════════════════════════════════════════════
# MINILCM ADD-ON  (after preprocessing, before grid search)
# ═══════════════════════════════════════════════════════════════
if USE_MINILCM:
    print("\\n" + "="*60)
    print("MINILCM ADD-ON  --  training neural feature extractor")
    print("="*60)
    print(f"Device: {DEVICE}  epochs={MINILCM_EPOCHS}  lr={MINILCM_LR}  batch={MINILCM_BATCH}")

    # Per-channel z-score (fit on train fold only, no leakage)
    eeg_mu  = allEEG_w[tr_win_mask].mean(axis=(0, 2), keepdims=True)
    eeg_std = allEEG_w[tr_win_mask].std(axis=(0, 2),  keepdims=True)
    eeg_std = np.where(eeg_std < 1e-8, 1.0, eeg_std)
    allEEG_norm = ((allEEG_w - eeg_mu) / eeg_std).astype(np.float32)

    # Use fold (TEST_FOLD-1) as validation, rest as training for MiniLCM
    vl_mask_lcm = (win_pos == (TEST_FOLD - 1) % 4)
    tr_mask_lcm = tr_win_mask & ~vl_mask_lcm

    minilcm_model = train_minilcm(allEEG_norm, allY, tr_mask_lcm, vl_mask_lcm)

    print("Extracting MiniLCM embeddings ...")
    lcm_embs = extract_minilcm_embeddings(minilcm_model, allEEG_norm)   # (N, 64)
    print(f"Embeddings shape: {lcm_embs.shape}")

    # Z-score embeddings with train-fold stats
    emb_mu  = lcm_embs[tr_win_mask].mean(axis=0)
    emb_std = lcm_embs[tr_win_mask].std(axis=0)
    emb_std = np.where(emb_std < 1e-8, 1.0, emb_std)
    lcm_embs_n = np.clip((lcm_embs - emb_mu) / emb_std, -10, 10)

    # Concatenate to normalised feature matrix
    allF_n = np.concatenate([allF_n, lcm_embs_n], axis=1)
    N_FEATURES = allF_n.shape[1]
    print(f"Feature dims after MiniLCM concat: {N_FEATURES} "
          f"(handcrafted={N_FEATURES - MINILCM_EMB_DIM} + neural={MINILCM_EMB_DIM})")

'''

ANCHOR5 = 'print("\\nComputing per-fold MI ...")'
assert ANCHOR5 in content, "ANCHOR5 not found"
content = content.replace(ANCHOR5, MINILCM_TRAIN_BLOCK + ANCHOR5, 1)
print("Part 3 inserted: MiniLCM training block")

with open('AddedRiemannFeatures.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\nAll done! AddedRiemannFeatures.py updated successfully.")

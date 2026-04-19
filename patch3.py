TRAIN = r'e:\FInal Year Project\LDACode\train.py'
with open(TRAIN, 'r', encoding='utf-8') as f:
    c = f.read()

# P4: windowing concat — add f_faa + f_riem
c = c.replace(
    '        f_eeg  = extract_eeg_features(ew)\n'
    '        f_muse = extract_band_features(mw)\n'
    '        f_bvp  = extract_bvp_features(bw, sr=TRAIN_BVP_SR)\n'
    '        f_hr   = extract_hr_features(hr_win)\n'
    '        f_ppi  = extract_ppi_features(ppi_win)\n'
    '\n'
    '        feat = np.concatenate([f_eeg, f_muse, f_bvp, f_hr, f_ppi]).astype(np.float32)',

    '        f_eeg  = extract_eeg_features(ew)\n'
    '        f_muse = extract_band_features(mw)\n'
    '        f_bvp  = extract_bvp_features(bw, sr=TRAIN_BVP_SR)\n'
    '        f_hr   = extract_hr_features(hr_win)\n'
    '        f_ppi  = extract_ppi_features(ppi_win)\n'
    '        f_faa  = extract_faa_features(ew)          # 14 FAA features\n'
    '        f_riem = extract_riemannian_features(ew)   # 10 Riemannian (pre-EA)\n'
    '\n'
    '        feat = np.concatenate(\n'
    '            [f_eeg, f_muse, f_bvp, f_hr, f_ppi, f_faa, f_riem]\n'
    '        ).astype(np.float32)',
    1
)
assert 'f_faa' in c, 'P4 FAILED'
print('P4 ok')

# P5: EA block after windowing print statement
EA_BLOCK = (
    'print(f"Training windows: {len(allY)} in {time.time()-t1:.1f}s")\n'
    '\n'
    '# ===============================================================\n'
    '# EUCLIDEAN ALIGNMENT -- per-subject Riemannian re-centring\n'
    '# Updates Riemannian feature columns with EA-aligned covariances\n'
    '# ===============================================================\n'
    'print("Applying Euclidean Alignment (EA) per subject ...")\n'
    'RIEM_START = N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP + N_FEAT_HR + N_FEAT_PPI + N_FEAT_FAA\n'
    '\n'
    '_sid_to_ew   = defaultdict(list)   # sid -> [ew, ...]  window order\n'
    '_sid_to_gidx = defaultdict(list)   # sid -> [global_row_idx, ...]\n'
    '\n'
    'for ti, tr in enumerate(train_trials):\n'
    '    sid = tr["sid"]; eeg = tr["eeg"]; bvp = tr["bvp"]; band = tr["band"]\n'
    '    dur2 = min(eeg.shape[1]/EEG_SR, band.shape[1]/BAND_SR, len(bvp)/TRAIN_BVP_SR)\n'
    '    step2 = WINDOW_SEC * (1 - OVERLAP_FRAC)\n'
    '    for wi in range(int((dur2 - WINDOW_SEC) / step2) + 1):\n'
    '        e_s = int(wi * step2 * EEG_SR); e_e = e_s + EEG_WIN\n'
    '        if e_e > eeg.shape[1]: break\n'
    '        _sid_to_ew[sid].append(eeg[:, e_s:e_e])\n'
    '\n'
    'for gidx, sid in enumerate(allSID):\n'
    '    _sid_to_gidx[sid].append(gidx)\n'
    '\n'
    'for sid in sorted(_sid_to_ew.keys()):\n'
    '    ew_list = _sid_to_ew[sid]; gidxs = _sid_to_gidx[sid]\n'
    '    if len(ew_list) < 2: continue\n'
    '    aligned = euclidean_alignment(ew_list)\n'
    '    for k, gidx in enumerate(gidxs):\n'
    '        if k < len(aligned):\n'
    '            allF_raw[gidx, RIEM_START:RIEM_START+N_FEAT_RIEM] = \\\n'
    '                extract_riemannian_features(np.array(aligned[k], dtype=np.float32))\n'
    '\n'
    'print("EA done -- Riemannian features updated with subject-aligned covariances.")\n'
)

c = c.replace(
    'print(f"Training windows: {len(allY)} in {time.time()-t1:.1f}s")\n',
    EA_BLOCK, 1
)
assert 'EA done' in c, 'P5 FAILED'
print('P5 ok')

with open(TRAIN, 'w', encoding='utf-8') as f:
    f.write(c)
print('patch3 saved')

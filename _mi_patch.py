path = r"e:\FInal Year Project\LDACode\DANN_LDA.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

OLD1 = '# LDA on top of DANN\nDANN_LDA_SH = "auto"'
NEW1 = '# LDA on top of DANN\nDANN_LDA_SH = "auto"\nMI_TOP_K = 80   # keep top-K features ranked by MI; set to None to disable'
if "MI_TOP_K" not in src:
    assert OLD1 in src, "anchor 1 not found"
    src = src.replace(OLD1, NEW1, 1)
    print("MI_TOP_K constant added")
else:
    print("MI_TOP_K already present")

OLD2 = "    trF=np.clip((trF-mu)/sd,-10,10).astype(np.float32)\n    teF=np.clip((teF-mu)/sd,-10,10).astype(np.float32)\n\n    trY=allY[tr_m]"
NEW2 = "    trF=np.clip((trF-mu)/sd,-10,10).astype(np.float32)\n    teF=np.clip((teF-mu)/sd,-10,10).astype(np.float32)\n\n    # -- 1b. MI feature selection (fitted on train only) --\n    if MI_TOP_K is not None and MI_TOP_K < trF.shape[1]:\n        mi_scores = mutual_info_classif(trF, allY[tr_m], random_state=42)\n        mi_cols   = np.argsort(mi_scores)[::-1][:MI_TOP_K]\n        trF = trF[:, mi_cols]; teF = teF[:, mi_cols]\n\n    trY=allY[tr_m]"
if "1b. MI feature selection" not in src:
    assert OLD2 in src, "anchor 2 not found"
    src = src.replace(OLD2, NEW2, 1)
    print("MI selection block inserted")
else:
    print("MI block already present")

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("Patch complete.")

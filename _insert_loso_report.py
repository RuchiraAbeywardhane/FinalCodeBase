path = r'e:\FInal Year Project\LDACode\MaxEffort.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with win_acc print inside LOSO loop
insert_after = None
for i, line in enumerate(lines):
    if 'win_acc' in line and 'n_wins' in line and 'loso_sid' in line and 'print' in line:
        insert_after = i
        break

if insert_after is None:
    print("ERROR: target line not found")
else:
    # Check if already inserted
    next_line = lines[insert_after + 1] if insert_after + 1 < len(lines) else ''
    if 'per-subject class breakdown' in next_line:
        print("Already inserted, skipping.")
    else:
        snippet = [
            '    # per-subject class breakdown\n',
            '    _sub_rows = [r for r in loso_win_rows if r["subject"] == loso_sid]\n',
            '    if _sub_rows:\n',
            '        _yt = [r["true_idx"] for r in _sub_rows]\n',
            '        _yp = [r["pred_idx"] for r in _sub_rows]\n',
            '        print(classification_report(_yt, _yp,\n',
            '              target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)],\n',
            '              zero_division=0, digits=2))\n',
            '        print(confusion_matrix(_yt, _yp, labels=list(range(NUM_CLASSES))))\n',
            '        print()\n',
        ]
        lines = lines[:insert_after + 1] + snippet + lines[insert_after + 1:]
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"Inserted {len(snippet)} lines after line {insert_after + 1}.")

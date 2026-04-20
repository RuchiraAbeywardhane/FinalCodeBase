content = open(r'e:\FInal Year Project\LDACode\DANN_LDA.py', encoding='utf-8').read()

old_riem = '          extract_riemannian_features(ew)\n        ]).astype(np.float32)'
new_riem = '          extract_riemannian_features(ew),\n          np.array([wpos],dtype=np.float32)\n        ]).astype(np.float32)'

old_feat = '        feat=np.concatenate(['
new_feat = '        n_wins_in_trial=int((dur-WINDOW_SEC)/step)+1\n        wpos=float(wi)/max(n_wins_in_trial-1,1)\n        feat=np.concatenate(['

found_riem = old_riem in content
found_feat = old_feat in content
print(f'found_riem={found_riem}, found_feat={found_feat}')

if found_riem and found_feat:
    content = content.replace(old_riem, new_riem, 1)
    content = content.replace(old_feat, new_feat, 1)
    open(r'e:\FInal Year Project\LDACode\DANN_LDA.py', 'w', encoding='utf-8').write(content)
    print('SUCCESS')

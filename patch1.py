TRAIN = r'e:\FInal Year Project\LDACode\train.py'
with open(TRAIN, 'r', encoding='utf-8') as f:
    c = f.read()

# P1: pyriemann import
c = c.replace(
    'from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n\nwarnings',
    'from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n\n'
    'try:\n'
    '    from pyriemann.estimation import Covariances as _PyrCov\n'
    '    PYRIEMANN_OK = True\n'
    'except ImportError:\n'
    '    PYRIEMANN_OK = False\n'
    '    print("[info] pyriemann not found -- manual SPD fallback active")\n\n'
    'warnings', 1)
assert 'PYRIEMANN_OK' in c, 'P1 FAILED'
print('P1 ok')

# P2: feature size constants
c = c.replace(
    'N_FEAT_PPI  = 8\n'
    'N_FEATURES_RAW = N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP + N_FEAT_HR + N_FEAT_PPI',
    'N_FEAT_PPI  = 8\n'
    'N_FEAT_FAA  = 14   # Frontal Alpha Asymmetry + hemispheric asymmetries\n'
    'N_FEAT_RIEM = 10   # Riemannian tangent-space upper triangle of 4x4 log-cov\n'
    'N_FEATURES_RAW = (N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP +\n'
    '                  N_FEAT_HR  + N_FEAT_PPI  + N_FEAT_FAA + N_FEAT_RIEM)', 1)
assert 'N_FEAT_FAA' in c, 'P2 FAILED'
print('P2 ok')

with open(TRAIN, 'w', encoding='utf-8') as f:
    f.write(c)
print('patch1 saved')

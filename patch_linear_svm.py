with open('train.py', 'r', encoding='utf-8') as f:
    src = f.read()

# 1. Fix imports
src = src.replace(
    'from sklearn.svm import SVC, LinearSVC\nfrom sklearn.calibration import CalibratedClassifierCV',
    'from sklearn.svm import SVC\nfrom sklearn.calibration import CalibratedClassifierCV'
)
# Remove any duplicate SVC lines from previous edits
src = src.replace(
    'from sklearn.svm import SVC\nfrom sklearn.svm import SVC, LinearSVC\nfrom sklearn.calibration import CalibratedClassifierCV',
    'from sklearn.svm import SVC\nfrom sklearn.calibration import CalibratedClassifierCV'
)

# 2. Helper function — add after imports block (after CalibratedClassifierCV line)
helper = '''
# ---------------------------------------------------------------
# CLASSIFIER FACTORY
# LinearSVC wrapped in CalibratedClassifierCV gives predict_proba.
# Linear kernel avoids RBF curse-of-dimensionality in high-dim space.
# ---------------------------------------------------------------
def make_linear_svm(C=1.0):
    from sklearn.svm import LinearSVC
    base = LinearSVC(C=C, class_weight='balanced', max_iter=2000,
                     random_state=42, dual=True)
    return CalibratedClassifierCV(base, cv=3, method='isotonic')

'''

insert_after = 'from sklearn.calibration import CalibratedClassifierCV'
src = src.replace(insert_after, insert_after + helper, 1)

# 3. Grid search — replace C_GRID and classifier inside loop
src = src.replace(
    "C_GRID = [0.1, 1.0, 10.0, 100.0]",
    "C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]"
)
src = src.replace(
    """                clf = SVC(kernel='rbf', C=C, gamma='scale',
                          class_weight='balanced', probability=True,
                          random_state=42)""",
    "                clf = make_linear_svm(C=C)"
)

# 4. Final model
src = src.replace(
    """final_clf = SVC(kernel='rbf', C=FINAL_C, gamma='scale',
                class_weight='balanced', probability=True, random_state=42)""",
    "final_clf = make_linear_svm(C=FINAL_C)"
)

# 5. LOSO loop
src = src.replace(
    """        clf_l = SVC(kernel='rbf', C=FINAL_C, gamma='scale',
                    class_weight='balanced', probability=True, random_state=42)""",
    "        clf_l = make_linear_svm(C=FINAL_C)"
)

with open('train.py', 'w', encoding='utf-8') as f:
    f.write(src)

import ast
try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax ERROR: {e}")

checks = [
    'CalibratedClassifierCV',
    'make_linear_svm',
    'LinearSVC',
    'C_GRID = [0.001',
    'clf = make_linear_svm',
    'final_clf = make_linear_svm',
    'clf_l = make_linear_svm',
]
for c in checks:
    print(f"  {'OK' if c in src else 'MISSING'}: {c}")

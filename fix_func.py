src = open('train.py', encoding='utf-8').read()

func = """
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def make_linear_svm(C=1.0):
    base = LinearSVC(C=C, class_weight='balanced', max_iter=2000,
                     random_state=42, dual=True)
    return CalibratedClassifierCV(base, cv=3, method='isotonic')

"""

src = src.replace('from sklearn.svm import SVC\n', 'from sklearn.svm import SVC\n' + func, 1)
open('train.py', 'w', encoding='utf-8').write(src)

import ast
ast.parse(src)
print('Syntax OK')

checks = [
    'def make_linear_svm',
    'LinearSVC',
    'CalibratedClassifierCV',
    'clf = make_linear_svm(C=C)',
    'final_clf = make_linear_svm',
    'clf_l = make_linear_svm',
]
for c in checks:
    print('OK' if c in src else 'MISSING', c)

import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from train import collect_features

MODEL_PATH = Path('model_rf_full_aug.joblib')
DATASET = Path('DeepfakeVsReal/Dataset')

if not MODEL_PATH.exists():
    raise FileNotFoundError('model.joblib not found')

clf = joblib.load(MODEL_PATH)
print('Loaded model from', MODEL_PATH)

val_root = DATASET / 'Validation'
if val_root.exists():
    print('\nEvaluating on Validation set...')
    Xv, yv = [], []
    real_val = val_root / 'Real'
    fake_val = val_root / 'Fake'
    Xrv, yrv = collect_features(real_val, 0, max_images=None)
    Xfv, yfv = collect_features(fake_val, 1, max_images=None)
    if len(Xrv) + len(Xfv) > 0:
        Xv = np.array(Xrv + Xfv)
        yv = np.array(yrv + yfv)
        probs = clf.predict_proba(Xv)[:, 1]
        preds = (probs >= 0.5).astype(int)
        print('Val samples:', Xv.shape[0])
        print('Val accuracy:', accuracy_score(yv, preds))
        try:
            print('Val ROC AUC:', roc_auc_score(yv, probs))
        except Exception:
            pass
        print(classification_report(yv, preds))
    else:
        print('No validation images found')
else:
    print('No Validation folder found at', val_root)

# Test set
test_root = DATASET / 'Test'
if test_root.exists():
    print('\nEvaluating on Test set...')
    real_test = test_root / 'Real'
    fake_test = test_root / 'Fake'
    Xrt, yrt = collect_features(real_test, 0, max_images=None)
    Xft, yft = collect_features(fake_test, 1, max_images=None)
    if len(Xrt) + len(Xft) > 0:
        Xt = np.array(Xrt + Xft)
        yt = np.array(yrt + yft)
        probs = clf.predict_proba(Xt)[:, 1]
        preds = (probs >= 0.5).astype(int)
        print('Test samples:', Xt.shape[0])
        print('Test accuracy:', accuracy_score(yt, preds))
        try:
            print('Test ROC AUC:', roc_auc_score(yt, probs))
        except Exception:
            pass
        print(classification_report(yt, preds))
    else:
        print('No test images found')
else:
    print('No Test folder found at', test_root)

"""
Train the SignSpeak gesture classifier.

Default: trains on the bundled starter dataset (model/gestures.py).
If data/landmarks.csv exists (created by collect_data.py with real webcam
samples), it is merged in - real data is what lifts real-world accuracy.

Run:  python model/train_model.py
Out:  model/signspeak_model.pkl  (+ prints accuracy and a classification report)
"""
import csv
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gestures import generate_dataset, LABELS  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_DATA = os.path.join(HERE, "..", "data", "landmarks.csv")
OUT = os.path.join(HERE, "signspeak_model.pkl")


def load_real_data():
    if not os.path.exists(REAL_DATA):
        return [], []
    X, y = [], []
    with open(REAL_DATA, newline="") as f:
        for row in csv.reader(f):
            if len(row) != 43:
                continue
            y.append(row[0])
            X.append([float(v) for v in row[1:]])
    return X, y


def main():
    X, y = generate_dataset()
    Xr, yr = load_real_data()
    if Xr:
        print(f"Merging {len(Xr)} real samples from {REAL_DATA}")
        X, y = X + Xr, y + yr
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Samples: {len(X)}  Classes: {len(set(y))}  Test accuracy: {acc:.3f}")
    print(classification_report(y_test, pred))
    with open(OUT, "wb") as f:
        pickle.dump({"model": clf, "labels": LABELS}, f)
    print("Saved", OUT)


if __name__ == "__main__":
    main()

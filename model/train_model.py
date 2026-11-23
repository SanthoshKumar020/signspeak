"""
Train the SignSpeak gesture classifier.

If data/landmarks.csv exists (real samples - produced by
model/convert_asl_dataset.py from the ASL Now Fingerspelling dataset, or
by model/collect_data.py from a webcam), the model trains on the REAL
data only, and the label set is whatever classes the real data contains.
Otherwise it falls back to the bundled synthetic starter dataset
(model/gestures.py) so the app still runs out of the box.

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
    X, y = load_real_data()
    if X:
        labels = sorted(set(y))
        print(f"Training on {len(X)} REAL samples from {REAL_DATA}")
        print(f"Classes ({len(labels)}): {labels}")
    else:
        X, y = generate_dataset()
        labels = LABELS
        print("No real data found - training on the synthetic starter dataset")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Samples: {len(X)}  Classes: {len(set(y))}  Test accuracy: {acc:.3f}")
    print(classification_report(y_test, pred))
    with open(OUT, "wb") as f:
        pickle.dump({"model": clf, "labels": labels}, f)
    print("Saved", OUT)


if __name__ == "__main__":
    main()

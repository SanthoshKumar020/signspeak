"""
SignSpeak - Real-Time Sign Language Gesture Recognition
Flask web app: Home / Detect / Dashboard with login.

The browser extracts the 21 hand landmarks with MediaPipe Hands (JS) and
POSTs them to /predict; the trained scikit-learn model answers with the
sign + confidence. Results can be saved to the user's Dashboard history.
"""
import os
import pickle
import sqlite3
import sys
from functools import wraps

from flask import (Flask, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
from gestures import normalize  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
# On Vercel the project directory is read-only, so the SQLite DB lives in /tmp
# (demo-grade persistence). Locally it sits next to app.py.
DB = "/tmp/signspeak.db" if os.environ.get("VERCEL") else os.path.join(BASE, "signspeak.db")
MODEL_PATH = os.path.join(BASE, "model", "signspeak_model.pkl")

app = Flask(__name__,
            template_folder=os.path.join(BASE, "templates"),
            static_folder=os.path.join(BASE, "static"))
app.secret_key = os.environ.get("SIGNSPEAK_SECRET", "dev-secret-change-me")

with open(MODEL_PATH, "rb") as f:
    _bundle = pickle.load(f)
MODEL = _bundle["model"]
LABELS = _bundle["labels"]


# ---------------------------------------------------------------- database
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def _close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db():
    conn = sqlite3.connect(DB)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- auth
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warn")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 6:
            flash("Fill all fields; password must be at least 6 characters.", "error")
            return render_template("register.html")
        try:
            db().execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, generate_password_hash(password)))
            db().commit()
        except sqlite3.IntegrityError:
            flash("An account with this email already exists.", "error")
            return render_template("register.html")
        flash("Account created - please log in.", "ok")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect(url_for("detect"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ---------------------------------------------------------------- pages
@app.route("/")
def home():
    return render_template("home.html", labels=LABELS)


@app.route("/detect")
@login_required
def detect():
    return render_template("detect.html", labels=LABELS)


@app.route("/dashboard")
@login_required
def dashboard():
    rows = db().execute(
        "SELECT label, confidence, created_at FROM detections "
        "WHERE user_id = ? ORDER BY id DESC LIMIT 50",
        (session["user_id"],)).fetchall()
    stats = db().execute(
        "SELECT label, COUNT(*) AS n FROM detections WHERE user_id = ? "
        "GROUP BY label ORDER BY n DESC",
        (session["user_id"],)).fetchall()
    total = sum(r["n"] for r in stats)
    return render_template("dashboard.html", rows=rows, stats=stats, total=total)


# ---------------------------------------------------------------- api
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json(force=True, silent=True) or {}
    landmarks = data.get("landmarks")
    if not landmarks or len(landmarks) != 21:
        return jsonify(error="expected 21 landmarks"), 400
    try:
        pts = [(float(p["x"]), float(p["y"])) for p in landmarks]
    except (KeyError, TypeError, ValueError):
        return jsonify(error="landmarks must be {x, y} points"), 400
    feats = [normalize(pts)]
    label = str(MODEL.predict(feats)[0])
    confidence = float(max(MODEL.predict_proba(feats)[0]))
    return jsonify(label=label, confidence=round(confidence, 3))


@app.route("/save_result", methods=["POST"])
@login_required
def save_result():
    data = request.get_json(force=True, silent=True) or {}
    label = str(data.get("label", "")).strip()
    if label not in LABELS:
        return jsonify(error="unknown label"), 400
    confidence = float(data.get("confidence", 0))
    db().execute(
        "INSERT INTO detections (user_id, label, confidence) VALUES (?, ?, ?)",
        (session["user_id"], label, confidence))
    db().commit()
    return jsonify(ok=True)


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

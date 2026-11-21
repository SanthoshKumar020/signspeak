"""
SignSpeak gesture definitions + synthetic landmark dataset generator.

Each gesture is described by how curled each finger is and where the thumb
sits. A small procedural hand model turns that description into the 21
MediaPipe-style hand landmarks (x, y), with random jitter, rotation and
scale so the trained model tolerates natural variation.

The bundled model is trained on this starter dataset so the app runs out of
the box. For production accuracy, collect real samples with collect_data.py
and retrain - see README.md.
"""
import math
import random

# Finger base (MCP) positions in image coordinates (y grows downward),
# hand pointing up, right hand with the thumb on the left.
MCP = {
    "index":  (-0.30, -0.85),
    "middle": (-0.10, -0.92),
    "ring":   ( 0.10, -0.88),
    "pinky":  ( 0.28, -0.78),
}
SEGMENTS = {
    "index":  (0.34, 0.24, 0.17),
    "middle": (0.36, 0.25, 0.17),
    "ring":   (0.32, 0.23, 0.16),
    "pinky":  (0.26, 0.18, 0.13),
}
FAN = {"index": -10.0, "middle": -3.0, "ring": 4.0, "pinky": 13.0}  # degrees

# curl: 0 = fully extended, 1 = fully curled into the palm
# thumb modes: "up" (alongside index), "out" (spread away), "tucked" (across palm),
#              "touch" (tip bent toward the index finger, as in F / D)
GESTURES = {
    "A":     dict(curls=dict(index=1.0, middle=1.0, ring=1.0, pinky=1.0), thumb="up",     spread=0.8),
    "B":     dict(curls=dict(index=0.05, middle=0.05, ring=0.05, pinky=0.05), thumb="tucked", spread=0.55),
    "C":     dict(curls=dict(index=0.55, middle=0.55, ring=0.55, pinky=0.55), thumb="out",  spread=0.8),
    "D":     dict(curls=dict(index=0.15, middle=0.95, ring=0.95, pinky=0.95), thumb="touch", spread=0.8),
    "E":     dict(curls=dict(index=1.05, middle=1.05, ring=1.05, pinky=1.05), thumb="tucked", spread=0.7),
    "F":     dict(curls=dict(index=0.85, middle=0.05, ring=0.05, pinky=0.05), thumb="touch", spread=0.8),
    "I":     dict(curls=dict(index=1.0, middle=1.0, ring=1.0, pinky=0.0), thumb="tucked", spread=0.8),
    "L":     dict(curls=dict(index=0.0, middle=1.0, ring=1.0, pinky=1.0), thumb="out", spread=0.9),
    "Y":     dict(curls=dict(index=1.0, middle=1.0, ring=1.0, pinky=0.0), thumb="out", spread=1.1),
    "HELLO": dict(curls=dict(index=0.0, middle=0.0, ring=0.0, pinky=0.0), thumb="out", spread=1.3),
}

LABELS = list(GESTURES.keys())
FINGERS = ["index", "middle", "ring", "pinky"]
# MediaPipe landmark order: wrist, thumb x4, index x4, middle x4, ring x4, pinky x4
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def _finger_points(base, fan_deg, curl, segments):
    """Return the 3 joint points + tip for one finger."""
    ang = math.radians(-90.0 + fan_deg)  # pointing up
    joint_bends = [curl * 18.0, curl * 52.0, curl * 62.0]  # degrees per joint
    pts = []
    x, y = base
    for i, seg in enumerate(segments):
        ang += math.radians(joint_bends[i] if i > 0 else joint_bends[0] * 0.4)
        x += seg * math.cos(ang)
        y += seg * math.sin(ang)
        pts.append((x, y))
    return pts  # PIP, DIP, TIP (MCP is the base, added by caller)


def _thumb_points(mode, rng):
    """Thumb chain: CMC, MCP, IP, TIP for the given mode."""
    cmc = (-0.30, -0.32)
    if mode == "up":        # extended upward alongside the index finger (A)
        dirs = [(-0.25, -0.28), (-0.06, -0.30), (-0.02, -0.26)]
    elif mode == "out":     # spread away from the palm (HELLO / L / Y / C)
        dirs = [(-0.30, -0.16), (-0.28, -0.10), (-0.24, -0.06)]
    elif mode == "tucked":  # folded across the palm (B / E / I)
        dirs = [(-0.10, -0.24), (0.16, -0.16), (0.20, -0.08)]
    elif mode == "touch":   # tip bent in toward the index finger (F / D)
        dirs = [(-0.26, -0.22), (-0.10, -0.26), (0.10, -0.20)]
    else:
        raise ValueError(mode)
    pts = [cmc]
    x, y = cmc
    for dx, dy in dirs:
        jx = dx * rng.uniform(0.92, 1.08)
        jy = dy * rng.uniform(0.92, 1.08)
        x, y = x + jx, y + jy
        pts.append((x, y))
    return pts


def make_hand(gesture, rng):
    """Generate one jittered 21-landmark sample for a gesture label."""
    spec = GESTURES[gesture]
    spread = spec["spread"] * rng.uniform(0.92, 1.08)
    landmarks = [(0.0, 0.0)]  # wrist
    landmarks += _thumb_points(spec["thumb"], rng)
    for finger in FINGERS:
        bx, by = MCP[finger]
        base = (bx * spread, by)
        curl = max(0.0, min(1.1, spec["curls"][finger] + rng.gauss(0, 0.06)))
        fan = FAN[finger] * spread + rng.gauss(0, 2.0)
        landmarks.append(base)
        landmarks += _finger_points(base, fan, curl, SEGMENTS[finger])
    # global rotation + scale + per-point noise
    rot = math.radians(rng.uniform(-10, 10))
    scale = rng.uniform(0.85, 1.15)
    out = []
    for (x, y) in landmarks:
        xr = x * math.cos(rot) - y * math.sin(rot)
        yr = x * math.sin(rot) + y * math.cos(rot)
        out.append((xr * scale + rng.gauss(0, 0.012),
                    yr * scale + rng.gauss(0, 0.012)))
    return out


def normalize(landmarks):
    """Wrist-relative, scale-normalised flat feature vector (42 values).

    Used identically for training samples and live predictions in app.py.
    landmarks: sequence of 21 (x, y) pairs.
    """
    wx, wy = landmarks[0]
    pts = [(x - wx, y - wy) for (x, y) in landmarks]
    mx, my = pts[9]  # middle-finger MCP
    scale = math.hypot(mx, my) or 1.0
    feats = []
    for (x, y) in pts:
        feats += [x / scale, y / scale]
    return feats


def generate_dataset(samples_per_class=400, seed=42):
    """Return (X, y): feature rows and labels for every gesture."""
    rng = random.Random(seed)
    X, y = [], []
    for label in LABELS:
        for _ in range(samples_per_class):
            X.append(normalize(make_hand(label, rng)))
            y.append(label)
    return X, y

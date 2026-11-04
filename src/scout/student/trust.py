"""The student: a tiny neural network that learned, from Scout's own recorded episodes, which remembered objects to
trust.

It reads nothing but the memory table every brain receives (the plain-text table from `Episodic.table()`), so it has
exactly the information the language-model brains have. For each remembered object it outputs the probability that
the object is really there: that its category was in view of the head camera and its remembered position is within
1.5 m of a real instance, which is what the memory quiz rewards.

The network is a two-layer perceptron in plain numpy (no torch): 27 inputs, 8 hidden units, 1 output, 233 parameters. Its weights
are trained by `scripts/train_student.py` on the unchanged arm of earlier episodes (houses disjoint from the final
evaluation) and saved to `assets/brains/student_trust.npz`.

What it does with that probability:
  answer(table, question)   quiz: pick the most trusted remembered object of the asked category; say yes if its
                            probability is at least the threshold, otherwise say no
  resolve(table, question)  verification: the question is about the most trusted object of that category
  decide(table, state)      explore while openings remain; afterwards re-check the object it is least sure about
  phrase(...)               the rule brain's wording

Only `answer` has been evaluated (offline, on recorded memories: `scripts/quiz_offline.py`). The other three methods
have not been run live; they are here so the brain is complete and can be run with `--brains student`.
"""
from __future__ import annotations

import os
import re

import numpy as np

CATEGORIES = ["bed", "sofa", "television", "dining table", "fridge", "kitchen counter", "chair", "armchair",
              "floor lamp", "painting", "laptop", "toilet", "sink", "dresser", "bathtub", "house plant"]
FEATURE_NAMES = (["log_times_seen", "best_confidence", "span_s", "age_s", "log_misses", "retracted", "gone",
                  "confident", "log_same_label_tracks", "log_total_tracks", "first_seen_frac"]
                 + [f"is_{c.replace(' ', '_')}" for c in CATEGORIES])
N_FEATURES = len(FEATURE_NAMES)  # 27
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "assets", "brains", "student_trust.npz")
THRESHOLD = 0.4  # chosen by leave-one-house-out cross-validation on the training runs (see docs/results/student.md)


def parse_table(table: str) -> list[dict]:
    """The memory table back into rows: one per remembered object, with its misses and verdict lines folded in.

    Row: id, label, first, last, n, best, x, y, ids, confident, misses, retracted, gone."""
    rows, by_id = [], {}
    for line in table.splitlines()[1:]:
        p = [x.strip() for x in line.split("|")]
        if len(p) >= 10 and p[0].isdigit():
            r = dict(id=int(p[0]), label=p[1], first=float(p[2]), last=float(p[3]), n=int(p[4]), best=float(p[5]),
                     x=float(p[6]), y=float(p[7]), ids=[int(x) for x in re.findall(r"\d+", p[8])],
                     confident=p[9].startswith("confident"), misses=0, retracted=False, gone=False)
            rows.append(r)
            by_id[r["id"]] = r
        elif len(p) >= 2 and p[0].isdigit() and int(p[0]) in by_id:
            r = by_id[int(p[0])]
            if "NOT seen" in p[1]:
                r["misses"] += 1
            elif "misrecognition" in p[1]:
                r["retracted"] = True
            elif "found it gone" in p[1]:
                r["gone"] = True
    return rows


def features(rows: list[dict]) -> np.ndarray:
    """One feature vector per row, from the table alone. `now` is the latest sighting in the table."""
    if not rows:
        return np.zeros((0, N_FEATURES), np.float32)
    now = max(r["last"] for r in rows)
    first_all = min(r["first"] for r in rows)
    span_all = max(now - first_all, 1.0)
    same = {}
    for r in rows:
        same[r["label"]] = same.get(r["label"], 0) + 1
    X = np.zeros((len(rows), N_FEATURES), np.float32)
    for i, r in enumerate(rows):
        X[i, :11] = [np.log1p(r["n"]), r["best"], (r["last"] - r["first"]) / 60.0, (now - r["last"]) / 60.0,
                     np.log1p(r["misses"]), float(r["retracted"]), float(r["gone"]), float(r["confident"]),
                     np.log1p(same[r["label"]] - 1), np.log1p(len(rows)), (r["first"] - first_all) / span_all]
        if r["label"] in CATEGORIES:
            X[i, 11 + CATEGORIES.index(r["label"])] = 1.0
    return X


class TrustMLP:
    """27 -> 8 -> 1 perceptron with tanh hidden units and a sigmoid output. 233 parameters."""

    def __init__(self, hidden: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((N_FEATURES, hidden)) / np.sqrt(N_FEATURES)).astype(np.float64)
        self.b1 = np.zeros(hidden)
        self.W2 = (rng.standard_normal((hidden, 1)) / np.sqrt(hidden)).astype(np.float64)
        self.b2 = np.zeros(1)
        self.mu = np.zeros(N_FEATURES)
        self.sd = np.ones(N_FEATURES)

    def n_params(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def _forward(self, X):
        Z = (X - self.mu) / self.sd
        H = np.tanh(Z @ self.W1 + self.b1)
        logit = H @ self.W2 + self.b2
        return Z, H, 1.0 / (1.0 + np.exp(-logit))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return self._forward(np.asarray(X, np.float64))[2][:, 0]

    def fit(self, X, y, X_val=None, y_val=None, epochs: int = 400, lr: float = 3e-3, l2: float = 1e-3,
            batch: int = 64, seed: int = 0, pos_weight: float | None = None) -> dict:
        """Adam on the weighted cross-entropy. Keeps the epoch with the lowest validation loss, if a validation set is
        given. Returns the training curve."""
        X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        rng = np.random.default_rng(seed)
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        wpos = pos_weight if pos_weight is not None else float((len(y) - y.sum()) / max(y.sum(), 1))
        curve, best, best_state = [], np.inf, None
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                bi = idx[s:s + batch]
                Z, H, p = self._forward(X[bi])
                w = np.where(y[bi] > 0.5, wpos, 1.0)[:, None]
                g_logit = w * (p - y[bi][:, None]) / len(bi)
                gW2 = H.T @ g_logit + l2 * self.W2
                gb2 = g_logit.sum(0)
                gH = g_logit @ self.W2.T * (1 - H ** 2)
                gW1 = Z.T @ gH + l2 * self.W1
                gb1 = gH.sum(0)
                step += 1
                for i, (prm, g) in enumerate(zip(params, [gW1, gb1, gW2, gb2])):
                    m[i] = b1 * m[i] + (1 - b1) * g
                    v[i] = b2 * v[i] + (1 - b2) * g * g
                    prm -= lr * (m[i] / (1 - b1 ** step)) / (np.sqrt(v[i] / (1 - b2 ** step)) + eps)
            rec = dict(epoch=ep, train_loss=self.loss(X, y, wpos))
            if X_val is not None and len(X_val):
                rec["val_loss"] = self.loss(X_val, y_val, wpos)
                if rec["val_loss"] < best:
                    best, best_state = rec["val_loss"], [p.copy() for p in params]
            curve.append(rec)
        if best_state is not None:
            self.W1[:], self.b1[:], self.W2[:], self.b2[:] = best_state
        return dict(curve=curve, best_val_loss=None if best_state is None else float(best),
                    best_epoch=None if best_state is None else int(np.argmin([c["val_loss"] for c in curve])))

    def loss(self, X, y, wpos: float = 1.0) -> float:
        p = np.clip(self.predict(X), 1e-7, 1 - 1e-7)
        w = np.where(y > 0.5, wpos, 1.0)
        return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).mean())

    def save(self, path: str, **meta) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, mu=self.mu, sd=self.sd,
                 feature_names=np.array(FEATURE_NAMES), **{k: np.array(v) for k, v in meta.items()})

    @classmethod
    def load(cls, path: str = WEIGHTS) -> "TrustMLP":
        z = np.load(path, allow_pickle=False)
        if list(z["feature_names"]) != FEATURE_NAMES:
            raise ValueError("weights were trained with a different feature set")
        net = cls(hidden=z["W1"].shape[1])
        net.W1, net.b1, net.W2, net.b2, net.mu, net.sd = (z[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2", "mu", "sd"))
        return net


class StudentBrain:
    """A brain whose only learned part is the trust network. Everything else is the rule brain's behaviour with
    "most trusted" in place of "confident, then most recent"."""
    name = "student (trust network, 233 parameters)"

    def __init__(self, weights: str = WEIGHTS, threshold: float = THRESHOLD):
        self.net = TrustMLP.load(weights)
        self.threshold = threshold

    def trust(self, table: str) -> list[dict]:
        rows = parse_table(table)
        p = self.net.predict(features(rows))
        for r, pi in zip(rows, p):
            r["p"] = float(pi)
        return rows

    def _best(self, table, question):
        """The most trusted remembered object of the category the question names. Categories are matched whole, so
        a question about an armchair does not match a chair."""
        q = question.lower()
        named = [c for c in CATEGORIES if re.search(rf"\b{re.escape(c)}\b", q)]
        if named:
            cat = max(named, key=len)
            rows = [r for r in self.trust(table) if r["label"] == cat]
        else:
            rows = [r for r in self.trust(table) if r["label"] in q]
        rows.sort(key=lambda r: -r["p"])
        return rows[0] if rows else None

    def decide(self, table, state):
        m = re.search(r"unexplored openings reachable: (\d+)", state)
        if m and int(m.group(1)) > 0:
            return {"do": "explore"}
        # the agent ignores a check on the last object checked or on one already checked twice; skip those here
        last = re.search(r"last object checked: (\d+)", state)
        twice = re.search(r"objects already checked twice \(do not choose\): \[([\d, ]*)\]", state)
        banned = ({int(last.group(1))} if last else set()) | \
            ({int(x) for x in re.findall(r"\d+", twice.group(1))} if twice else set())
        rows = [r for r in self.trust(table) if not r["retracted"] and not r["gone"] and r["id"] not in banned]
        if not rows:
            return {"do": "explore"}
        r = min(rows, key=lambda r: abs(r["p"] - 0.5))
        return {"do": "check", "object_id": r["id"], "why": f"least sure about this one (trust {r['p']:.2f})"}

    def resolve(self, table, question):
        r = self._best(table, question)
        if r is None:
            return {"object_id": None, "say": "I never saw one."}
        return {"object_id": r["id"], "say": f"I last saw the {r['label']} at {r['last']:.0f} s. I'll go and look."}

    def phrase(self, question, remembered, finding):
        from ..brains import RuleBrain
        return RuleBrain().phrase(question, remembered, finding)

    def answer(self, table, question):
        r = self._best(table, question)
        if r is None or r["p"] < self.threshold:
            say = "I did not see one." if r is None else f"I do not trust that I saw one (trust {r['p']:.2f})."
            return {"answer": say, "cited_sightings": [], "observed": False, "object_id": None}
        return {"answer": f"I saw a {r['label']}, last at {r['last']:.0f} s (trust {r['p']:.2f}).",
                "cited_sightings": r["ids"][-1:], "observed": True, "object_id": r["id"]}

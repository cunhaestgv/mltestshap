"""Test generation: SHAP-sign partitions, entropy margins and decision trees.

A test case is (row, feature, expected sign): the input X_gen[row] must keep
feature `feature` contributing to the output in direction `expected` (+1/-1).
Test inputs are real samples. Only samples whose own SHAP sign matches their
partition are chosen, so the suite passes on the model it was generated from.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text


def entropy(npos, nneg):
    """Binary entropy (base 2) of SHAP signs; arrays allowed."""
    npos = np.asarray(npos, dtype=float)
    nneg = np.asarray(nneg, dtype=float)
    tot = np.maximum(npos + nneg, 1.0)
    h = np.zeros_like(tot)
    for c in (npos, nneg):
        p = c / tot
        with np.errstate(divide="ignore", invalid="ignore"):
            h -= np.where(p > 0, p * np.log2(p), 0.0)
    return h


def _majority(npos, nneg):
    return 1 if npos >= nneg else -1


def analyse_feature(v, s, theta, wide_frac):
    """Single-feature analysis (Algorithm 1 of the paper).

    v: feature values, s: SHAP signs (+1/-1). Scans from the minimum upwards
    and from the maximum downwards while the entropy of the samples beyond the
    threshold stays below theta. The values left between the two scans form
    the uncertainty margin.
    Returns a dict with type "single" (one partition), "narrow" (two
    partitions with a sharp boundary) or "wide" (blurred boundary).
    """
    u, inv = np.unique(v, return_inverse=True)
    pos = np.bincount(inv, weights=(s > 0), minlength=len(u))
    neg = np.bincount(inv, weights=(s < 0), minlength=len(u))
    P_le, N_le = np.cumsum(pos), np.cumsum(neg)
    P_ge, N_ge = np.cumsum(pos[::-1])[::-1], np.cumsum(neg[::-1])[::-1]
    H_all = float(entropy(P_le[-1], N_le[-1]))
    info = dict(H=H_all, min=float(u[0]), max=float(u[-1]), n_values=len(u))
    if H_all < theta:
        return dict(info, type="single", sign=_majority(P_le[-1], N_le[-1]),
                    lo=float(u[0]), hi=float(u[-1]), margin_frac=0.0)

    H_le, H_ge = entropy(P_le, N_le), entropy(P_ge, N_ge)
    low_ok = np.cumprod(H_le < theta).astype(bool)          # contiguous from the minimum
    high_ok = np.cumprod((H_ge < theta)[::-1]).astype(bool)[::-1]
    lo_k = int(np.flatnonzero(low_ok)[-1]) if low_ok.any() else -1
    hi_k = int(np.flatnonzero(high_ok)[0]) if high_ok.any() else len(u)

    if lo_k >= hi_k:
        # both scans meet: the boundary is sharp; split where purity is best
        n = P_le[-1] + N_le[-1]
        w = (P_le[:-1] + N_le[:-1]) / n
        split = w * H_le[:-1] + (1 - w) * H_ge[1:]
        c = int(np.argmin(split))
        lo_k, hi_k = c, c + 1
    if lo_k < 0 or hi_k >= len(u):
        return dict(info, type="wide", margin_frac=1.0, lo=None, hi=None)

    inside = (inv > lo_k) & (inv < hi_k)
    margin_frac = float(inside.mean())
    out = dict(info, lo=float(u[lo_k]), hi=float(u[hi_k]), margin_frac=margin_frac,
               sign_lo=_majority(P_le[lo_k], N_le[lo_k]),
               sign_hi=_majority(P_ge[hi_k], N_ge[hi_k]))
    out["type"] = "narrow" if margin_frac <= wide_frac else "wide"
    return out


def _closest(v, s, mask, target, sign, k):
    """Rows in mask with SHAP sign `sign`, closest to `target`."""
    idx = np.flatnonzero(mask & (s == sign))
    return idx[np.argsort(np.abs(v[idx] - target), kind="stable")[:k]]


def tree_partitions(X, s, f, theta, depth, min_leaf, k, names):
    """Multi-feature partitions of feature f's SHAP sign using a decision tree.

    Returns (tests, leaves): for every leaf whose entropy is below theta, the k
    samples nearest to the leaf's boundaries (the thresholds on its path).
    """
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth,
                                  min_samples_leaf=min_leaf, random_state=0)
    tree.fit(X, s > 0)
    t = tree.tree_
    leaf_of = tree.apply(X)
    path = tree.decision_path(X)
    span = np.maximum(X.max(0) - X.min(0), 1e-9)
    tests, leaves = [], []
    for leaf in np.unique(leaf_of):
        rows = np.flatnonzero(leaf_of == leaf)
        npos, nneg = (s[rows] > 0).sum(), (s[rows] < 0).sum()
        h = float(entropy(npos, nneg))
        if h >= theta:
            continue
        sign = _majority(npos, nneg)
        nodes = [n for n in path[rows[0]].indices if n != leaf]
        rule = []
        dist = np.full(len(rows), np.inf)
        for n in nodes:
            feat, thr = t.feature[n], t.threshold[n]
            left = X[rows[0], feat] <= thr
            rule.append("%s %s %.4g" % (names[feat], "<=" if left else ">", thr))
            dist = np.minimum(dist, np.abs(X[rows, feat] - thr) / span[feat])
        ok = s[rows] == sign
        chosen = rows[ok][np.argsort(dist[ok], kind="stable")[:k]]
        tests += [(int(r), f, sign, "tree", leaf) for r in chosen]
        leaves.append(dict(feature=names[f], leaf=int(leaf), sign=sign, entropy=h,
                           samples=len(rows), rule=" and ".join(rule) or "(root)"))
    return tests, leaves, export_text(tree, feature_names=list(names), max_depth=depth)


def generate(X, S, names, theta=0.1, wide_frac=0.1, k=3, depth=4, min_leaf_frac=0.01):
    """Generate the test suite from flattened inputs X and their SHAP values S."""
    signs = np.sign(S)
    min_leaf = max(5, int(min_leaf_frac * len(X)))
    tests, features, leaves, trees = [], [], [], {}
    for f, name in enumerate(names):
        ok = signs[:, f] != 0
        if ok.sum() < 2 * k:
            continue
        v, s = X[:, f], signs[:, f]
        a = analyse_feature(v[ok], s[ok], theta, wide_frac)
        features.append(dict(a, feature=name))
        if a["type"] == "single":
            for target in (a["lo"], a["hi"]):
                tests += [(int(r), f, a["sign"], "single", 0)
                          for r in _closest(v, s, ok, target, a["sign"], k)]
        elif a["type"] == "narrow":
            tests += [(int(r), f, a["sign_lo"], "boundary", 0)
                      for r in _closest(v, s, ok & (v <= a["lo"]), a["lo"], a["sign_lo"], k)]
            tests += [(int(r), f, a["sign_hi"], "boundary", 1)
                      for r in _closest(v, s, ok & (v >= a["hi"]), a["hi"], a["sign_hi"], k)]
        else:
            t, lv, text = tree_partitions(X[ok], s[ok], f, theta, depth, min_leaf, k, names)
            rows = np.flatnonzero(ok)
            tests += [(int(rows[r]), f, sg, kind, leaf) for r, _, sg, kind, leaf in t]
            leaves += lv
            trees[name] = text
    tests = np.array(tests, dtype=object).reshape(-1, 5)
    return dict(rows=tests[:, 0].astype(int), feats=tests[:, 1].astype(int),
                expected=tests[:, 2].astype(int), kind=tests[:, 3].astype(str),
                part=tests[:, 4].astype(int), features=features, leaves=leaves, trees=trees)


def random_suite(S, n, rng):
    """n random (input, feature) pairs; expected = SHAP sign on the original model."""
    rows = rng.integers(0, S.shape[0], n * 3)
    feats = rng.integers(0, S.shape[1], n * 3)
    sg = np.sign(S[rows, feats]).astype(int)
    keep = np.flatnonzero(sg != 0)[:n]
    return dict(rows=rows[keep], feats=feats[keep], expected=sg[keep])


def bva_suite(X, S, n):
    """Classic boundary-value analysis without partitions: real samples at and
    next to each feature's minimum and maximum (min, min+, max-, max, ...),
    spread evenly over the features until the suite has n tests."""
    n_feat = X.shape[1]
    per = int(np.ceil(n / n_feat))
    rows, feats, exp = [], [], []
    for f in range(n_feat):
        order_lo = np.argsort(X[:, f], kind="stable")
        order_hi = order_lo[::-1]
        picked = []
        for a, b in zip(order_lo, order_hi):
            for r in (a, b):
                if np.sign(S[r, f]) != 0 and r not in picked:
                    picked.append(r)
            if len(picked) >= per:
                break
        for r in picked[:per]:
            rows.append(r); feats.append(f); exp.append(int(np.sign(S[r, f])))
    order = np.argsort(np.arange(len(rows)) % per, kind="stable")[:n]   # balance over features
    rows, feats, exp = np.array(rows)[order], np.array(feats)[order], np.array(exp)[order]
    return dict(rows=rows, feats=feats, expected=exp)

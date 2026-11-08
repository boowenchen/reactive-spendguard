"""Train, quantize and export Spendguard's MLP.

    .venv/Scripts/python.exe train.py        (Windows)
    .venv/bin/python train.py                (POSIX)

Outputs (all deterministic for a given seed):
    model/weights.json        quantized weights, threshold, feature constants
    src/SpendguardModel.sol   generated Solidity library (do not edit by hand)
    test/vectors.json         parity vectors for the Foundry tests
    model/REPORT.md           held-out metrics and the single-feature baseline

The data is synthetic (see generate.py); the report measures the mechanism on
that data only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import generate as gen
import spendguard_ref as ref

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
Q1, Q2 = 20, 14                 # weight scales (powers of two)
HIDDEN_CANDIDATES = [8, 12, 16]
FREEZE_BUDGET = 0.01            # max share of normal sequences that may be frozen (validation)
LATENCY_S = 10                  # assumed seconds from flagged spend to freeze landing
TRAIN_SEED = 7
CONSTANTS = ["S", "FLOW_ONE", "GAP_MAX", "H_BURST", "H_FLOW", "H_FRESH", "TRUST_AGE", "ACC_MAX", "SIZE_CLAMP",
             "EMA_SHIFT"]
EARLY_HALF = 3.0                # attack-position weight half-life (spends)
POS_WEIGHT = 0.5                # total positive weight relative to total negative weight
EPOCHS = 60
HN_ROUNDS = 3                   # hard-negative rounds
HN_WEIGHT = 4.0                 # extra weight per round on each sequence's top normal spend


# ------------------------------------------------------------------ data
def featurize(seqs):
    rows = []
    for sid, (kind, s) in enumerate(seqs):
        xs = ref.run_learn_all(s)
        start = next((j for j, x in enumerate(s) if x["label"]), len(s))
        for i, (x, sp) in enumerate(zip(xs, s)):
            rows.append((sid, i, sp["label"], x, max(i - start, 0)))
    sid = np.array([r[0] for r in rows])
    pos = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows], dtype=np.float64)
    X = np.array([r[3] for r in rows], dtype=np.int64)
    k = np.array([r[4] for r in rows])
    return X, y, sid, pos, k


# ------------------------------------------------------------------ float MLP
def train_mlp(Xs, y, k, sid, hidden, seed, epochs=EPOCHS, batch=4096, lr=3e-3, l2=1e-4):
    """Weighted logistic regression through one ReLU layer, plain numpy + Adam.

    Round 0 trains on all spends. Each later round up-weights the normal spends
    that score highest *within their own sequence's top few*, because a single
    high score anywhere in a normal sequence is what causes a false freeze.
    """
    rng = np.random.default_rng(seed)
    n, d = Xs.shape
    W1 = rng.normal(0, math.sqrt(2 / d), (hidden, d))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, math.sqrt(1 / hidden), hidden)
    b2 = np.zeros(1)
    params = [W1, b1, W2, b2]
    # Early attacker spends matter most: weight halves every EARLY_HALF spends into the attack.
    early = np.where(y == 1, 2.0 ** (-k / EARLY_HALF), 0.0)
    neg_w = np.where(y == 0, 1.0, 0.0)
    for rnd in range(HN_ROUNDS + 1):
        if rnd > 0:
            z = forward(Xs, params)
            hard = np.zeros(n, dtype=bool)
            negz = np.where(y == 0, z, -np.inf)
            # top normal spend of every sequence (the one that would freeze it)
            best = np.full(sid.max() + 1, -np.inf)
            np.maximum.at(best, sid, negz)
            hard = (y == 0) & (negz >= best[sid]) & (negz >= np.quantile(z[y == 0], 0.9))
            neg_w = np.where(hard, neg_w * HN_WEIGHT, neg_w)
        pos_w = neg_w.sum() / max(early.sum(), 1e-9) * POS_WEIGHT
        sw = np.where(y == 1, early * pos_w, neg_w)
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        eps = epochs if rnd == 0 else epochs // 2
        for ep in range(eps):
            order = rng.permutation(n)
            cur_lr = lr * (0.5 * (1 + math.cos(math.pi * ep / eps)))
            for i in range(0, n, batch):
                idx = order[i:i + batch]
                xb, yb, wb = Xs[idx], y[idx], sw[idx]
                z1 = xb @ W1.T + b1
                h = np.maximum(z1, 0)
                z2 = h @ W2 + b2[0]
                p = 1 / (1 + np.exp(-np.clip(z2, -40, 40)))
                g = (p - yb) * wb / wb.sum()
                gW2 = h.T @ g + l2 * W2
                gb2 = np.array([g.sum()])
                gh = np.outer(g, W2) * (z1 > 0)
                gW1 = gh.T @ xb + l2 * W1
                gb1 = gh.sum(0)
                t += 1
                for j, (p_, g_) in enumerate(zip(params, [gW1, gb1, gW2, gb2])):
                    m[j] = 0.9 * m[j] + 0.1 * g_
                    v[j] = 0.999 * v[j] + 0.001 * g_ * g_
                    p_ -= cur_lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
    return W1, b1, W2, b2[0]


def forward(Xs, params):
    W1, b1, W2, b2 = params
    return np.maximum(Xs @ W1.T + b1, 0) @ W2 + b2[0]


def quantize(W1, b1, W2, b2, mu, sd):
    """Fold standardization into layer 1, then round to integers."""
    W1f = W1 / sd                       # acts on raw integer features
    b1f = b1 - (W1 * mu / sd).sum(1)
    w1 = np.round(W1f * 2 ** Q1).astype(np.int64)
    bb1 = np.round(b1f * 2 ** Q1).astype(np.int64)
    w2 = np.round(W2 * 2 ** Q2).astype(np.int64)
    bb2 = int(round(b2 * 2 ** (Q2 + 8)))
    return w1, bb1, w2, bb2


def int_scores(X, w1, b1, w2, b2):
    """Vectorised integer inference (int64); cross-checked against ref.QModel."""
    h = X @ w1.T + b1
    assert np.abs(h).max() < 2 ** 52
    h = np.where(h > 0, h >> (Q1 - 8), 0)
    acc = h @ w2 + b2
    assert np.abs(acc).max() < 2 ** 52
    return acc >> Q2


# ------------------------------------------------------------------ evaluation
def first_flags(flag, sid, pos, n_seq):
    first = np.full(n_seq, -1)
    idx = np.nonzero(flag)[0]
    for i in idx[::-1]:
        first[sid[i]] = pos[i]
    return first


def seq_threshold(score, sid, normal_seq_mask, budget):
    """Smallest threshold whose normal-sequence freeze rate is <= budget."""
    mx = np.full(len(normal_seq_mask), -np.inf)
    np.maximum.at(mx, sid, score)
    m = np.sort(mx[normal_seq_mask])[::-1]
    k = int(math.floor(budget * len(m)))      # sequences allowed to freeze
    return int(math.floor(m[k])) + 1           # strictly above the (k+1)-th largest max


def outcome(seqs, sel, first):
    """Sequence-level results for the selected sequence indices."""
    res = {"normal": {"n": 0, "frozen": 0}}
    for k in gen.ATTACKS:
        res[k] = {"n": 0, "detected": 0, "prefix_false_freeze": 0, "leaked_spends": [],
                  "lost_frac": [], "unguarded_lost_frac": []}
    for i in sel:
        kind, s = seqs[i]
        f = first[i]
        if kind == "normal":
            res["normal"]["n"] += 1
            res["normal"]["frozen"] += int(f >= 0)
            continue
        r = res[kind]
        r["n"] += 1
        start = next(j for j, x in enumerate(s) if x["label"])
        b_start = s[start]["balance"]
        total = sum(x["amount"] for x in s[start:])
        r["unguarded_lost_frac"].append(total / b_start)
        if 0 <= f < start:
            r["prefix_false_freeze"] += 1
            r["lost_frac"].append(0.0)
            continue
        if f < 0:
            r["leaked_spends"].append(len(s) - start)
            r["lost_frac"].append(total / b_start)
            continue
        r["detected"] += 1
        stop = s[f]["ts"] + LATENCY_S
        leaked = [x for x in s[start:] if x["ts"] <= stop]
        r["leaked_spends"].append(len(leaked))
        r["lost_frac"].append(sum(x["amount"] for x in leaked) / b_start)
    return res


def spend_metrics(flag, y, mask):
    f, t = flag[mask], y[mask] == 1
    tp, fp = int((f & t).sum()), int((f & ~t).sum())
    fn, tn = int((~f & t).sum()), int((~f & ~t).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1))


def summarize(res):
    attacks = [res[k] for k in gen.ATTACKS]
    n_att = sum(r["n"] for r in attacks)
    det = sum(r["detected"] for r in attacks)
    lost = [v for r in attacks for v in r["lost_frac"]]
    ung = [v for r in attacks for v in r["unguarded_lost_frac"]]
    return dict(
        normal_freeze_rate=res["normal"]["frozen"] / max(res["normal"]["n"], 1),
        detection_rate=det / max(n_att, 1),
        prefix_false_freeze=sum(r["prefix_false_freeze"] for r in attacks) / max(n_att, 1),
        mean_lost_frac=float(np.mean(lost)),
        mean_unguarded_lost_frac=float(np.mean(ung)),
        by_kind={k: dict(n=res[k]["n"], detected=res[k]["detected"],
                         rate=res[k]["detected"] / max(res[k]["n"], 1),
                         median_leaked=float(np.median(res[k]["leaked_spends"])) if res[k]["leaked_spends"] else 0.0,
                         mean_lost_frac=float(np.mean(res[k]["lost_frac"])),
                         mean_unguarded_lost_frac=float(np.mean(res[k]["unguarded_lost_frac"])))
                 for k in gen.ATTACKS},
    )


def evaluate(score, threshold, seqs, y, sid, pos, sel_seq, sel_rows):
    flag = score >= threshold
    first = first_flags(flag, sid, pos, len(seqs))
    return summarize(outcome(seqs, sel_seq, first)), spend_metrics(flag, y, sel_rows)


# ------------------------------------------------------------------ demo plan
DEMO = {
    "token_decimals": 18,
    "funding": 1000,
    "cap": 100,
    # (amount tokens, gap seconds before this spend, recipient label)
    "normal": [(4, 0, "A"), (6, 180, "B"), (3.5, 200, "A"), (5, 150, "C"), (7, 220, "B"), (4.5, 170, "A")],
    "attack": [(96, 150, "X"), (98, 6, "X"), (95, 6, "X"), (99, 6, "X"), (97, 6, "X")],
}
ADDR = {"A": "0x000000000000000000000000000000000000a11a", "B": "0x000000000000000000000000000000000000b22b",
        "C": "0x000000000000000000000000000000000000c33c", "X": "0x000000000000000000000000000000000000dead"}
T0 = 1_800_000_000


def demo_spends(gap_scale=1.0, burst_gap=None):
    bal = gen.tok(DEMO["funding"])
    ts = T0
    out = []
    for phase in ("normal", "attack"):
        for j, (amt, gap, to) in enumerate(DEMO[phase]):
            g = gap
            if phase == "normal" or j == 0:
                g = int(round(gap * gap_scale))
            elif burst_gap is not None:
                g = burst_gap
            ts += g
            a = gen.tok(amt)
            out.append(dict(amount=a, balance=bal, ts=ts, to=to, label=int(phase == "attack")))
            bal -= a
    return out


def main():
    seqs, split = gen.build()
    X, y, sid, pos, kpos = featurize(seqs)
    row_split = split[sid]
    tr, va, te = (row_split == s for s in ("train", "val", "test"))
    seq_kind = np.array([k for k, _ in seqs])
    normal_seq = seq_kind == "normal"
    val_seq = np.nonzero(split == "val")[0]
    test_seq = np.nonzero(split == "test")[0]

    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-9
    Xs = (X - mu) / sd

    # ---- architecture selection on validation
    cands = []
    for hidden in HIDDEN_CANDIDATES:
        W1, b1, W2, b2 = train_mlp(Xs[tr], y[tr], kpos[tr], sid[tr], hidden, TRAIN_SEED)
        q = quantize(W1, b1, W2, b2, mu, sd)
        sc = int_scores(X, *q)
        vmask = np.isin(sid, val_seq)
        thr = seq_threshold(np.where(vmask, sc, -10 ** 9), sid, normal_seq & (split == "val"), FREEZE_BUDGET)
        vsum, _ = evaluate(sc, thr, seqs, y, sid, pos, val_seq, va)
        cands.append((vsum["detection_rate"], -hidden, hidden, q, sc, thr, vsum, (W1, b1, W2, b2)))
        print(f"hidden={hidden:2d} val detection={vsum['detection_rate']:.3f} "
              f"normal freeze={vsum['normal_freeze_rate']:.3f} thr={thr}")
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
    _, _, hidden, (w1, b1q, w2, b2q), score, thr, vsum, floatw = cands[0]

    # ---- integer parity: vectorised vs reference implementation
    qm = ref.QModel(w1.tolist(), b1q.tolist(), w2.tolist(), int(b2q), Q1, Q2, thr)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(X), 2000, replace=False):
        assert qm.score(X[i].tolist()) == int(score[i])
    W1, b1, W2, b2 = floatw
    fl = (np.maximum(Xs @ W1.T + b1, 0) @ W2 + b2) * 256
    err = np.abs(fl - score)
    quant_err = (float(err.mean()), float(np.percentile(err, 99)), float(err.max()))

    # ---- baseline: best single-feature threshold rule (chosen on validation)
    base = []
    vnorm = normal_seq & (split == "val")
    for j, name in enumerate(ref.FEATURES):
        for sign in (1, -1):
            f = sign * X[:, j].astype(np.float64)
            t = seq_threshold(np.where(np.isin(sid, val_seq), f, -1e18), sid, vnorm, FREEZE_BUDGET)
            vs, _ = evaluate(f, t, seqs, y, sid, pos, val_seq, va)
            base.append((vs["detection_rate"], name, sign, t, f))
    base.sort(key=lambda b: b[0], reverse=True)
    b_det, b_name, b_sign, b_thr, b_f = base[0]

    nn_test, nn_spend = evaluate(score, thr, seqs, y, sid, pos, test_seq, te)
    bl_test, bl_spend = evaluate(b_f, b_thr, seqs, y, sid, pos, test_seq, te)
    # threshold-free per-spend comparison: AUC on test rows
    def auc(s, lab):
        order = np.argsort(s, kind="mergesort")
        r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
        # average ranks for ties
        _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
        npos = lab.sum(); nneg = len(lab) - npos
        return float((r[lab == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    nn_auc = auc(score[te].astype(float), y[te])
    bl_auc = auc(b_f[te], y[te])

    # ---- the same comparison at other false-freeze budgets (thresholds re-chosen on validation)
    vmask_rows = np.isin(sid, val_seq)
    curve = []
    for budget in (0.005, 0.01, 0.02, 0.05):
        t_nn = seq_threshold(np.where(vmask_rows, score, -10 ** 9), sid, vnorm, budget)
        best = None
        for j, name in enumerate(ref.FEATURES):
            for sign in (1, -1):
                f = sign * X[:, j].astype(np.float64)
                t = seq_threshold(np.where(vmask_rows, f, -1e18), sid, vnorm, budget)
                vs, _ = evaluate(f, t, seqs, y, sid, pos, val_seq, va)
                if best is None or vs["detection_rate"] > best[0]:
                    best = (vs["detection_rate"], name, sign, t, f)
        a, _ = evaluate(score, t_nn, seqs, y, sid, pos, test_seq, te)
        b, _ = evaluate(best[4], best[3], seqs, y, sid, pos, test_seq, te)
        curve.append(dict(budget=budget, nn=a, bl=b, rule=f"{best[1]} {'>=' if best[2] > 0 else '<='} "
                                                         f"{best[3] if best[2] > 0 else -best[3]:g}"))

    # ---- demo plan + robustness to live timing
    demo = demo_spends()
    demo_res, demo_first, _ = ref.run_onchain(qm, demo)
    grid = []
    for gs in (0.5, 0.75, 1.0, 1.5, 2.0):
        for bg in (2, 4, 6, 10, 15):
            r, f, _ = ref.run_onchain(qm, demo_spends(gs, bg))
            grid.append(dict(normal_gap_scale=gs, burst_gap_s=bg, first_flag=f,
                             max_normal_score=max(x["score"] for x in r[:len(DEMO["normal"])])))

    # ---- export weights.json
    quant = dict(w1=w1.tolist(), b1=b1q.tolist(), w2=w2.tolist(), b2=int(b2q), q1=Q1, q2=Q2)
    qhash = hashlib.sha256(json.dumps(quant, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    consts = {k: getattr(ref, k) for k in CONSTANTS}
    weights = dict(
        note="Synthetic-data demonstration model. Generated by model/train.py; do not edit.",
        architecture=dict(inputs=ref.N_FEATURES, hidden=hidden, activation="relu", outputs=1),
        features=ref.FEATURES, constants=consts, threshold=thr, score_units="logit * 256",
        quantized=quant, quantized_sha256=qhash,
        standardization=dict(mean=mu.tolist(), std=sd.tolist()),
        data_seed=gen.SEED, train_seed=TRAIN_SEED, freeze_budget=FREEZE_BUDGET,
    )
    (MODEL_DIR / "weights.json").write_text(json.dumps(weights, indent=1) + "\n", encoding="utf-8")
    write_solidity(w1, b1q, w2, b2q, thr, hidden, qhash)

    # ---- vectors for Foundry
    vec_rows = list(rng.choice(np.nonzero(te)[0], 240, replace=False))
    model_vecs = [dict(x=X[i].tolist(), y=int(score[i])) for i in vec_rows]
    lg = ref.log2fx
    edge = [[0] * ref.N_FEATURES,
            [lg(65537), 0, ref.ACC_MAX, ref.ACC_MAX, ref.ACC_MAX, 0, ref.ACC_MAX, ref.ACC_MAX, ref.SIZE_CLAMP],
            [0, lg(ref.GAP_MAX + 1), ref.S, 0, 0, lg(2 ** 40), 0, 0, -ref.SIZE_CLAMP],
            [lg(65537), lg(ref.GAP_MAX + 1), ref.S, ref.FLOW_ONE, ref.FLOW_ONE, 0, ref.S, ref.FLOW_ONE, 0]]
    model_vecs += [dict(x=e, y=qm.score(e)) for e in edge]
    sequences = []

    def seq_entry(name, spends):
        res, first, _ = ref.run_onchain(qm, spends)
        return dict(name=name, firstFlag=-1 if first is None else first, spends=[
            dict(amount=s["amount"], balance=s["balance"], features=r["features"],
                 flagged=r["flagged"], score=r["score"], to=ADDR.get(s["to"]) or
                 "0x" + format(0x5000 + int(s["to"]), "040x"), ts=s["ts"])
            for s, r in zip(spends, res)])
    sequences.append(seq_entry("demo", demo))
    sequences.append(seq_entry("demo-normal-only", demo[:len(DEMO["normal"])]))
    picked = {}
    for i in test_seq:
        k = seqs[i][0]
        if k not in picked:
            picked[k] = i
    for k, i in picked.items():
        sequences.append(seq_entry(f"test-{k}", seqs[i][1][:45]))
    # One vector / one spend per line keeps diffs readable. Keys are sorted because
    # Foundry decodes JSON objects into structs in alphabetical key order.
    def line(o):
        return json.dumps(o, sort_keys=True, separators=(",", ":"))
    seq_txt = ",\n".join(
        '{"firstFlag":%d,"name":%s,"spends":[\n%s\n]}' % (q["firstFlag"], json.dumps(q["name"]),
                                                          ",\n".join(line(sp) for sp in q["spends"]))
        for q in sequences)
    (ROOT / "test" / "vectors.json").write_text(
        '{"threshold":%d,\n"model":[\n%s\n],\n"sequences":[\n%s\n]}\n'
        % (thr, ",\n".join(line(v) for v in model_vecs), seq_txt), encoding="utf-8", newline="\n")

    write_report(hidden, thr, cands, vsum, nn_test, nn_spend, bl_test, bl_spend, b_name, b_sign, b_thr, base,
                 nn_auc, bl_auc, quant_err, demo, demo_res, demo_first, grid, qhash, curve)
    print(f"selected hidden={hidden} threshold={thr} test detection={nn_test['detection_rate']:.3f} "
          f"vs baseline {b_name}{'>=' if b_sign > 0 else '<='} {b_thr if b_sign > 0 else -b_thr}: "
          f"{bl_test['detection_rate']:.3f}; demo first flag={demo_first}")


# ------------------------------------------------------------------ writers
def write_solidity(w1, b1, w2, b2, thr, hidden, qhash):
    lines = []
    for i in range(hidden):
        terms = " ".join(f"{'+' if w >= 0 else '-'} {abs(int(w))} * x[{j}]" for j, w in enumerate(w1[i]))
        lines.append(f"        h = {int(b1[i])} {terms};\n"
                     f"        if (h > 0) acc += {int(w2[i])} * (h >> {Q1 - 8});")
    body = "\n".join(lines)
    consts = "\n".join(f"    {'int256' if k == 'SIZE_CLAMP' else 'uint256'} internal constant {k} = {getattr(ref, k)};"
                       for k in CONSTANTS)
    src = f"""// SPDX-License-Identifier: MIT
// GENERATED by model/train.py -- do not edit by hand. quantized_sha256={qhash}
pragma solidity ^0.8.29;

/// @title SpendguardModel
/// @notice Fixed-point MLP ({len(w1[0])} -> {hidden} ReLU -> 1) and feature constants.
/// Inputs are the integer features built by SpendguardReactive. Layer-1 weights carry scale 2^{Q1},
/// layer-2 weights 2^{Q2}. The result is the logit * 256.
library SpendguardModel {{
    bytes32 internal constant WEIGHTS_SHA256 = 0x{qhash};
    int256 internal constant THRESHOLD = {thr};
{consts}

    function score(int256[{len(w1[0])}] memory x) internal pure returns (int256) {{
        int256 acc = {int(b2)};
        int256 h;
{body}
        return acc >> {Q2};
    }}
}}
"""
    (ROOT / "src" / "SpendguardModel.sol").write_text(src, encoding="utf-8", newline="\n")


def pct(x):
    return f"{100 * x:.1f}%"


def write_report(hidden, thr, cands, vsum, nn, nns, bl, bls, b_name, b_sign, b_thr, base, nn_auc, bl_auc,
                 quant_err, demo, demo_res, demo_first, grid, qhash, curve):
    rule = f"`{b_name} {'>=' if b_sign > 0 else '<='} {b_thr if b_sign > 0 else -b_thr}`"
    L = []
    L.append("# Model report\n")
    L.append("Generated by `model/train.py`. **All data is synthetic** (`model/generate.py`, seed "
             f"{gen.SEED}); these numbers show the mechanism works on the patterns we generated, "
             "not real-world accuracy.\n")
    L.append("## Setup\n")
    L.append(f"- 6,000 sequences (3,000 normal, 600 per attack pattern), split 60/20/20 by sequence.")
    L.append(f"- Features ({len(ref.FEATURES)}): {', '.join('`'+f+'`' for f in ref.FEATURES)} "
             "(definitions in the README).")
    L.append(f"- Architecture chosen on validation: {ref.N_FEATURES} -> **{hidden}** ReLU -> 1. Candidates: " +
             ", ".join(f"{c[2]} hidden = {pct(c[0])} val detection" for c in sorted(cands, key=lambda c: c[2])) + ".")
    L.append(f"- Quantized weights: layer 1 x2^{Q1}, layer 2 x2^{Q2}; score = logit x 256. "
             f"|float - integer| score difference over all spends: mean {quant_err[0]:.2f}, p99 {quant_err[1]:.2f}, "
             f"max {quant_err[2]:.2f} score units (256 = 1.0 logit).")
    L.append(f"- Threshold **{thr}** (logit {thr / 256:.2f}): the lowest value that freezes at most "
             f"{pct(FREEZE_BUDGET)} of *validation* normal sequences (60 spends each). The baseline gets "
             "the same budget.")
    L.append(f"- Freeze latency assumed for the loss columns: {LATENCY_S} s after the flagged spend "
             "(spends inside that window still go through).")
    L.append(f"- weights sha256 (quantized section): `{qhash}`\n")
    L.append("## Held-out test set: sequence level\n")
    L.append("A sequence is *frozen* at its first spend with score >= threshold. For attack sequences, a freeze "
             "before the attack starts counts as a false freeze, not a detection.\n")
    L.append(f"| Metric | Neural net | Baseline {rule} |\n|---|---|---|")
    L.append(f"| Normal sequences frozen (false freeze) | {pct(nn['normal_freeze_rate'])} | {pct(bl['normal_freeze_rate'])} |")
    L.append(f"| Attack prefix frozen early (false freeze) | {pct(nn['prefix_false_freeze'])} | {pct(bl['prefix_false_freeze'])} |")
    L.append(f"| Attacks detected | **{pct(nn['detection_rate'])}** | {pct(bl['detection_rate'])} |")
    L.append(f"| Mean share of balance lost (no guard: {pct(nn['mean_unguarded_lost_frac'])}) | "
             f"{pct(nn['mean_lost_frac'])} | {pct(bl['mean_lost_frac'])} |")
    L.append("")
    L.append("| Attack pattern | n | NN detected | NN median leaked spends | NN lost | Baseline detected | Baseline lost | No guard lost |\n|---|---|---|---|---|---|---|---|")
    for k in gen.ATTACKS:
        a, b = nn["by_kind"][k], bl["by_kind"][k]
        L.append(f"| {k} | {a['n']} | {pct(a['rate'])} | {a['median_leaked']:.0f} | {pct(a['mean_lost_frac'])} | "
                 f"{pct(b['rate'])} | {pct(b['mean_lost_frac'])} | {pct(a['mean_unguarded_lost_frac'])} |")
    L.append("\n*Leaked spends* = attacker spends that executed before the freeze landed (including the flagged one). "
             "Undetected attacks count all their spends.\n")
    L.append("## Held-out test set: per spend\n")
    L.append("Every spend scored with its full history (no freeze applied), positive = attacker spend.\n")
    L.append(f"| | Neural net | Baseline |\n|---|---|---|")
    for key, lab in (("precision", "Precision"), ("recall", "Recall"), ("fpr", "False-positive rate")):
        L.append(f"| {lab} | {nns[key]:.4f} | {bls[key]:.4f} |")
    L.append(f"| ROC AUC (threshold-free) | {nn_auc:.4f} | {bl_auc:.4f} |")
    L.append("")
    L.append("Confusion matrices (rows = truth, columns = prediction):\n")
    for nm, m in (("Neural net", nns), ("Baseline", bls)):
        L.append(f"{nm}:\n\n| | flagged | not flagged |\n|---|---|---|\n"
                 f"| attacker spend | {m['tp']} | {m['fn']} |\n| normal spend | {m['fp']} | {m['tn']} |\n")
    L.append("## Single-feature baselines (validation, same freeze budget)\n")
    L.append("| Rule | Val detection |\n|---|---|")
    for det, name, sign, t, _ in base:
        L.append(f"| `{name} {'>=' if sign > 0 else '<='} {t if sign > 0 else -t}` | {pct(det)} |")
    L.append("")
    L.append("## Other false-freeze budgets (test set; thresholds re-chosen on validation)\n")
    L.append("| Budget | NN detected | NN false freeze | NN lost | Best single rule | Rule detected | Rule false freeze | Rule lost |\n"
             "|---|---|---|---|---|---|---|---|")
    for c in curve:
        a, b = c["nn"], c["bl"]
        L.append(f"| {pct(c['budget'])} | {pct(a['detection_rate'])} | {pct(a['normal_freeze_rate'])} | "
                 f"{pct(a['mean_lost_frac'])} | `{c['rule']}` | {pct(b['detection_rate'])} | "
                 f"{pct(b['normal_freeze_rate'])} | {pct(b['mean_lost_frac'])} |")
    L.append("")
    L.append("## What the network adds\n")
    diff = nn["detection_rate"] - bl["detection_rate"]
    wins = [k for k in gen.ATTACKS if nn["by_kind"][k]["rate"] > bl["by_kind"][k]["rate"]]
    losses = [k for k in gen.ATTACKS if nn["by_kind"][k]["rate"] < bl["by_kind"][k]["rate"]]
    L.append(f"At the same false-freeze budget the network detects {pct(nn['detection_rate'])} of held-out attacks "
             f"versus {pct(bl['detection_rate'])} for the best single-feature rule {rule} "
             f"({'+' if diff >= 0 else ''}{100 * diff:.1f} points), and the mean share of balance lost is "
             f"{pct(nn['mean_lost_frac'])} versus {pct(bl['mean_lost_frac'])} (no guard: "
             f"{pct(nn['mean_unguarded_lost_frac'])}).\n")
    L.append(f"- Patterns where the network detects more: {', '.join(wins) or 'none'}.")
    L.append(f"- Patterns where the single rule detects more: {', '.join(losses) or 'none'}.")
    L.append("- Each single feature tracks one symptom: a short gap or a high short-window outflow also occurs in "
             "normal batch payments, and a new recipient occurs whenever the agent adds a vendor. The network "
             "weighs several symptoms together; on this data that gives more detections at the same false-freeze "
             "budget, at every budget in the table above.")
    L.append("- `mimic` (normal amounts at normal pace to one new address) looks like a normal agent adding a "
             "vendor under these features; neither method catches it reliably. The per-call cap is the only "
             "bound on that pattern.\n")
    L.append("## Live demo plan (exact replay of the reference implementation)\n")
    L.append(f"Wallet funded with {DEMO['funding']} tokens, per-call cap {DEMO['cap']}. Scores below assume the "
             "listed gaps; live block timestamps will differ by a few seconds.\n")
    L.append("| # | phase | gap (s) | amount | to | features | score | flagged |\n|---|---|---|---|---|---|---|---|")
    plan = [("normal", *x) for x in DEMO["normal"]] + [("attack", *x) for x in DEMO["attack"]]
    for i, ((ph, amt, gap, to), r) in enumerate(zip(plan, demo_res)):
        L.append(f"| {i} | {ph} | {gap} | {amt} | {to} | {r['features']} | {r['score']} | {'**yes**' if r['flagged'] else 'no'} |")
    L.append(f"\nFirst flagged spend: index {demo_first}.\n")
    L.append("Timing robustness (first flagged index; normal spends are indices 0-5):\n")
    L.append("| normal gap scale | burst gap (s) | first flag | max normal score |\n|---|---|---|---|")
    for g in grid:
        L.append(f"| {g['normal_gap_scale']} | {g['burst_gap_s']} | {g['first_flag']} | {g['max_normal_score']} |")
    L.append("")
    (MODEL_DIR / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Integer-exact reference for Spendguard's on-chain feature extractor and MLP.

Everything here uses Python integers and mirrors `src/SpendguardReactive.sol` and
the generated `src/SpendguardModel.sol` operation by operation:

* all divisions are on non-negative integers (floor == Solidity truncation);
* the only signed operation is the arithmetic right shift (`>>`), which rounds
  toward negative infinity in both Python and Solidity (SAR).

If you change anything here, regenerate the model (`python train.py`) and rerun
`forge test`; the parity tests replay `test/vectors.json` through the contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- constants
S = 256                 # fixed-point one for count-like features
FLOW_ONE = 4096         # fixed-point one for outflow fractions (100 % of balance)
GAP_MAX = 1 << 17       # seconds; gap assigned to the first spend of an epoch
H_BURST = 120           # half-life (s) of the spend counter and the short outflow sum
H_FLOW = 1800           # half-life (s) of the long outflow sum
H_FRESH = 600           # half-life (s) of the fresh-recipient counter and value sum
TRUST_AGE = 600         # a recipient first paid less than this many seconds ago is "fresh"
ACC_MAX = 1 << 17       # clamp for decayed accumulators
SIZE_CLAMP = 128        # clamp for log2(amount / typical) in 1/16 doublings
EMA_SHIFT = 3           # typical-amount EMA weight = 1/8

FEATURES = ["frac", "gap", "burst", "rush", "flow", "recipAge", "fresh", "freshFlow", "size"]
N_FEATURES = len(FEATURES)


def log2fx(n: int) -> int:
    """floor(log2(n)) * 16 + next 4 mantissa bits. Requires n >= 1."""
    assert n >= 1
    k = n.bit_length() - 1
    m = n >> (k - 4) if k >= 4 else n << (4 - k)
    return k * 16 + (m - 16)


def decay(v: int, dt: int, h: int) -> int:
    """v * 2^(-dt/h): exact halvings, then a chord between 1 and 1/2."""
    k = dt // h
    if k >= 64:
        return 0
    v >>= k
    r = dt % h
    return v - (v * r) // (2 * h)


@dataclass
class State:
    """Rolling state (reset on every new epoch) plus the learned profile."""
    count: int = 0
    last_ts: int = 0
    burst: int = 0
    rush: int = 0
    flow: int = 0
    fresh: int = 0
    fresh_flow: int = 0
    # profile: survives epochs
    ema: int = 0
    first_seen: dict = field(default_factory=dict)   # recipient -> first learned timestamp

    def reset_rolling(self) -> None:
        self.count = self.last_ts = self.burst = self.rush = self.flow = self.fresh = self.fresh_flow = 0


def features(st: State, amount: int, balance_before: int, ts: int, to) -> tuple[list[int], dict]:
    """Return the feature vector and the rolling-state update (not yet applied)."""
    if st.count == 0:
        dt = GAP_MAX
    else:
        dt = ts - st.last_ts if ts > st.last_ts else 0
        if dt > GAP_MAX:
            dt = GAP_MAX
    if balance_before == 0 or amount >= balance_before:
        ratio = 1 << 16
        part = FLOW_ONE
    else:
        ratio = amount * (1 << 16) // balance_before
        part = amount * FLOW_ONE // balance_before
    frac = log2fx(ratio + 1)
    gap = log2fx(dt + 1)
    burst = min(decay(st.burst, dt, H_BURST) + S, ACC_MAX)
    rush = min(decay(st.rush, dt, H_BURST) + part, ACC_MAX)
    flow = min(decay(st.flow, dt, H_FLOW) + part, ACC_MAX)
    seen = st.first_seen.get(to, 0)
    if seen == 0:
        age, is_fresh = 0, True
    else:
        a = ts - seen if ts > seen else 0
        age, is_fresh = log2fx(a + 2), a < TRUST_AGE
    fresh = min(decay(st.fresh, dt, H_FRESH) + (S if is_fresh else 0), ACC_MAX)
    fresh_flow = min(decay(st.fresh_flow, dt, H_FRESH) + (part if is_fresh else 0), ACC_MAX)
    if st.ema == 0:
        size = 0
    else:
        size = log2fx(amount + 1) - log2fx(st.ema + 1)
        size = max(-SIZE_CLAMP, min(SIZE_CLAMP, size))
    x = [frac, gap, burst, rush, flow, age, fresh, fresh_flow, size]
    upd = dict(burst=burst, rush=rush, flow=flow, fresh=fresh, fresh_flow=fresh_flow, ts=ts)
    return x, upd


def apply(st: State, upd: dict, amount: int, to, learn: bool) -> None:
    """Commit the rolling update; `learn` adds the spend to the profile."""
    st.burst, st.rush, st.flow = upd["burst"], upd["rush"], upd["flow"]
    st.fresh, st.fresh_flow = upd["fresh"], upd["fresh_flow"]
    if upd["ts"] > st.last_ts:
        st.last_ts = upd["ts"]
    st.count += 1
    if learn:
        if st.first_seen.get(to, 0) == 0:
            st.first_seen[to] = upd["ts"]
        if st.ema == 0:
            st.ema = amount
        elif amount >= st.ema:
            st.ema += (amount - st.ema) >> EMA_SHIFT
        else:
            st.ema -= (st.ema - amount) >> EMA_SHIFT


# ------------------------------------------------------------------ MLP
@dataclass
class QModel:
    w1: list[list[int]]   # [hidden][inputs], scale 2^q1 per raw feature unit
    b1: list[int]         # scale 2^q1
    w2: list[int]         # scale 2^q2
    b2: int               # scale 2^(q2+8)
    q1: int
    q2: int
    threshold: int        # score units (logit * 256)

    @classmethod
    def from_json(cls, d: dict) -> "QModel":
        q = d["quantized"]
        return cls(q["w1"], q["b1"], q["w2"], q["b2"], q["q1"], q["q2"], d["threshold"])

    def score(self, x: list[int]) -> int:
        """Logit * 256, exactly as SpendguardModel.score() computes it."""
        acc = self.b2
        shift = self.q1 - 8
        for row, b, w in zip(self.w1, self.b1, self.w2):
            h = b
            for wij, xj in zip(row, x):
                h += wij * xj
            if h > 0:
                acc += w * (h >> shift)
        return acc >> self.q2


def run_onchain(model: QModel, spends, threshold: int | None = None, st: State | None = None):
    """Replay spends exactly like SpendguardReactive (flagged spends are not learned).

    `spends` is an iterable of dicts {amount, balance, ts, to}. Returns a list of
    per-spend results and the index of the first flagged spend (or None).
    """
    t = model.threshold if threshold is None else threshold
    st = st or State()
    out, first = [], None
    for i, s in enumerate(spends):
        x, upd = features(st, s["amount"], s["balance"], s["ts"], s["to"])
        sc = model.score(x)
        flagged = sc >= t
        apply(st, upd, s["amount"], s["to"], learn=not flagged)
        out.append(dict(features=x, score=sc, flagged=flagged))
        if flagged and first is None:
            first = i
    return out, first, st


def run_learn_all(spends):
    """Features when every spend is learned. Identical to `run_onchain` up to and
    including the first flagged spend, so first-flag outcomes are exact."""
    st = State()
    xs = []
    for s in spends:
        x, upd = features(st, s["amount"], s["balance"], s["ts"], s["to"])
        apply(st, upd, s["amount"], s["to"], learn=True)
        xs.append(x)
    return xs

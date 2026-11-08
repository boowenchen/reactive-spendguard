"""Synthetic spending sequences for a hot / AI-agent wallet.

THE DATA IS SYNTHETIC. There is no public labelled corpus of compromised
agent-wallet spends; these generators encode plausible behaviour so the model
can demonstrate the mechanism (a neural score computed inside a Reactive
contract). Metrics measured on this data say nothing about real-world accuracy.

A sequence is a list of spends {amount, balance, ts, to, label, kind}:
  amount / balance  token wei (18 decimals), balance = balance before the spend
  ts                block timestamp (s)
  to                recipient id (int; mapped to an address only for vectors)
  label             1 for spends made by the attacker, 0 otherwise

Normal agents pay a small pool of regular recipients at irregular intervals,
sometimes add a new recipient, sometimes batch a few quick payments, and are
refilled by the owner. Compromise patterns start after a normal prefix:

  burst      cap-sized spends every 2-15 s to one or two new addresses
  sweep      1-4 large spends (50-100 % of balance, capped) to a new address
  drip       spends just under the per-call cap at the agent's usual pace
  spray      normal-sized spends to a different new address every 5-60 s
  mimic      normal-sized spends at normal pace to one new address (hard case)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

SEED = 20261108
WEI = 10 ** 18
ATTACKS = ["burst", "sweep", "drip", "spray", "mimic"]


def tok(x: float) -> int:
    """Token amount -> wei, micro-token precision."""
    return int(round(max(x, 0.0) * 1e6)) * 10 ** 12


class Agent:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.b0 = 10 ** rng.uniform(2, 4)                       # 100 .. 10,000 tokens
        self.typical = self.b0 * 10 ** rng.uniform(-3.0, -1.7)  # 0.1 % .. 2 % of the funded balance
        self.cap = min(self.typical * rng.uniform(5, 20), 0.10 * self.b0)  # owner-set per-call cap
        self.gap_med = 10 ** rng.uniform(1.7, 3.0)              # 50 s .. 1000 s
        self.p_new = rng.uniform(0.03, 0.12)
        self.p_batch = rng.uniform(0.0, 0.08)
        self.pool = list(range(int(rng.integers(2, 9))))
        self.weights = 1.0 / np.arange(1, len(self.pool) + 1)   # Zipf-like
        self.next_id = 1000
        self.balance = tok(self.b0)
        self.ts = 1_800_000_000 + int(rng.integers(0, 10 ** 6))
        self.batch_left = 0
        self.attacking = False

    def new_recipient(self) -> int:
        self.next_id += 1
        return self.next_id

    def refill(self):
        # The owner tops the wallet up between normal sessions, never mid-attack.
        if not self.attacking and self.balance < tok(0.30 * self.b0):
            self.balance = tok(self.b0)

    def emit(self, amount_tok: float, gap: float, to: int, label: int, kind: str):
        self.refill()
        amount = min(tok(min(amount_tok, self.cap)), self.balance)
        if amount <= 0:
            return None
        self.ts += max(1, int(round(gap)))
        s = dict(amount=amount, balance=self.balance, ts=self.ts, to=to, label=label, kind=kind)
        self.balance -= amount
        return s

    # ---- normal behaviour -------------------------------------------------
    def normal_spend(self):
        r = self.rng
        if self.batch_left > 0:
            self.batch_left -= 1
            gap = r.uniform(3, 30)
        else:
            if r.random() < self.p_batch:
                self.batch_left = int(r.integers(1, 4))
            gap = r.lognormal(math.log(self.gap_med), 1.0)
            if r.random() < 0.02:
                gap += r.exponential(6 * 3600)
            gap = min(max(gap, 2), 86400)
        if r.random() < self.p_new:
            to = self.new_recipient()
            self.pool.append(to)
            self.weights = np.append(self.weights, 1.0 / len(self.pool))
        else:
            p = self.weights / self.weights.sum()
            to = self.pool[int(r.choice(len(self.pool), p=p))]
        amt = self.typical * r.lognormal(0, 0.6)
        if r.random() < 0.03:
            amt *= r.uniform(2, 6)
        return self.emit(amt, gap, to, 0, "normal")

    # ---- attacker behaviour ----------------------------------------------
    def attack(self, kind: str):
        r = self.rng
        out = []
        self.refill()
        self.attacking = True
        if kind == "burst":
            dests = [self.new_recipient() for _ in range(int(r.integers(1, 3)))]
            for _ in range(40):
                s = self.emit(self.cap * r.uniform(0.6, 1.0), r.uniform(2, 15),
                              dests[int(r.integers(len(dests)))], 1, kind)
                if s is None:
                    break
                out.append(s)
        elif kind == "sweep":
            to = self.new_recipient()
            for _ in range(int(r.integers(1, 5))):
                s = self.emit(self.balance / WEI * r.uniform(0.5, 1.0), r.uniform(20, 300), to, 1, kind)
                if s: out.append(s)
        elif kind == "drip":
            to = self.new_recipient()
            for _ in range(40):
                s = self.emit(self.cap * r.uniform(0.85, 0.99),
                              min(max(r.lognormal(math.log(self.gap_med), 0.5), 5), 7200), to, 1, kind)
                if s: out.append(s)
        elif kind == "spray":
            for _ in range(int(r.integers(10, 41))):
                s = self.emit(self.typical * r.lognormal(0, 0.6), r.uniform(5, 60), self.new_recipient(), 1, kind)
                if s: out.append(s)
        elif kind == "mimic":
            to = self.new_recipient()
            for _ in range(int(r.integers(20, 41))):
                s = self.emit(self.typical * r.lognormal(0, 0.6),
                              min(max(r.lognormal(math.log(self.gap_med), 1.0), 2), 86400), to, 1, kind)
                if s: out.append(s)
        return out


def normal_sequence(rng, n=60):
    a = Agent(rng)
    seq = [a.normal_spend() for _ in range(n)]
    return [s for s in seq if s]


def attack_sequence(rng, kind):
    while True:  # deterministic retry in the rare case the attack phase is empty
        a = Agent(rng)
        prefix = [a.normal_spend() for _ in range(int(rng.integers(5, 41)))]
        attack = a.attack(kind)
        if attack:
            return [s for s in prefix if s] + attack


def build(seed: int = SEED, n_normal: int = 3000, n_per_attack: int = 600):
    """Deterministic list of (kind, sequence). Split is by sequence index."""
    rng = np.random.default_rng(seed)
    seqs = [("normal", normal_sequence(rng)) for _ in range(n_normal)]
    for kind in ATTACKS:
        seqs += [(kind, attack_sequence(rng, kind)) for _ in range(n_per_attack)]
    order = rng.permutation(len(seqs))
    seqs = [seqs[i] for i in order]
    split = np.empty(len(seqs), dtype="<U5")
    split[:] = "train"
    split[int(0.6 * len(seqs)):int(0.8 * len(seqs))] = "val"
    split[int(0.8 * len(seqs)):] = "test"
    return seqs, split


def main():
    seqs, split = build()
    out = Path(__file__).with_name("data")
    out.mkdir(exist_ok=True)
    summary = {"seed": SEED, "sequences": len(seqs), "by_kind": {}, "spends": 0, "attack_spends": 0}
    for kind, s in seqs:
        summary["by_kind"][kind] = summary["by_kind"].get(kind, 0) + 1
        summary["spends"] += len(s)
        summary["attack_spends"] += sum(x["label"] for x in s)
    summary["split"] = {k: int((split == k).sum()) for k in ("train", "val", "test")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with open(out / "sample.jsonl", "w", encoding="utf-8") as f:
        for kind, s in seqs[:20]:
            f.write(json.dumps({"kind": kind, "spends": [{**x, "amount": str(x["amount"]), "balance": str(x["balance"])} for x in s]}) + "\n")
    json.dump(summary, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()

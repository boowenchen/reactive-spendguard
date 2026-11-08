"""Recompute Spendguard scores from on-chain Spend logs and compare with Scored logs.

Anyone can check the live system: fetch the wallet's Spend logs on Base Sepolia and the
reactive contract's Scored logs on Lasna, then

    python model/replay.py --spend-logs spend.json [--scored-logs scored.json]

Both files are `cast logs --json` output (a JSON array of log objects with `topics` and
`data`). Only the standard library and `spendguard_ref.py` are needed (no numpy).
Exits non-zero if any recomputed score, feature or flag differs from the on-chain one.

Example (addresses from the README):
    cast logs --json --rpc-url https://sepolia.base.org --from-block <b> --address $WALLET \
      "Spend(uint256 indexed,address indexed,uint256,uint256,uint256,uint256)" > spend.json
    cast logs --json --rpc-url https://lasna-rpc.rnk.dev/ --from-block <b> --address $REACTIVE \
      "Scored(uint256 indexed,uint256 indexed,int256,int256[9],bool)" > scored.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spendguard_ref as ref  # noqa: E402


def words(data: str) -> list[int]:
    h = data[2:] if data.startswith("0x") else data
    return [int(h[i:i + 64], 16) for i in range(0, len(h), 64)]


def signed(v: int) -> int:
    return v - (1 << 256) if v >= 1 << 255 else v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spend-logs", required=True)
    ap.add_argument("--scored-logs")
    ap.add_argument("--weights", default=str(HERE / "weights.json"))
    ap.add_argument("--threshold", type=int, help="override (must match the deployed reactive contract)")
    a = ap.parse_args()

    model = ref.QModel.from_json(json.loads(Path(a.weights).read_text(encoding="utf-8")))
    thr = model.threshold if a.threshold is None else a.threshold
    spends = json.loads(Path(a.spend_logs).read_text(encoding="utf-8"))
    spends.sort(key=lambda l: (int(str(l["blockNumber"]), 0), int(str(l["logIndex"]), 0)))
    onchain = {}
    if a.scored_logs:
        for l in json.loads(Path(a.scored_logs).read_text(encoding="utf-8")):
            w = words(l["data"])
            onchain[int(l["topics"][2], 16)] = (signed(w[0]), [signed(v) for v in w[1:10]], bool(w[10]))

    st, epoch, bad = ref.State(), 0, 0
    print("nonce  epoch  score  flagged  features  [on-chain check]")
    for l in spends:
        nonce = int(l["topics"][1], 16)
        to = "0x" + l["topics"][2][-40:].lower()
        amount, balance, ts, e = words(l["data"])[:4]
        if e > epoch:
            epoch = e
            st.reset_rolling()
        elif e < epoch:
            continue
        x, upd = ref.features(st, amount, balance, ts, to)
        sc = model.score(x)
        flagged = sc >= thr
        ref.apply(st, upd, amount, to, learn=not flagged)
        check = ""
        if nonce in onchain:
            ok = onchain[nonce] == (sc, x, flagged)
            bad += not ok
            check = "match" if ok else f"MISMATCH on-chain={onchain[nonce]}"
        elif onchain:
            check = "no Scored log"
        print(f"{nonce:5d}  {e:5d}  {sc:5d}  {str(flagged):7s}  {x}  {check}")
    if bad:
        print(f"{bad} mismatches", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

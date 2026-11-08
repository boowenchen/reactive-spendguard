// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

import {AbstractReactive} from "@reactive/src/base/AbstractReactive.sol";
import {ISystemContract} from "@reactive/src/interfaces/ISystemContract.sol";
import {SpendguardModel as M} from "./SpendguardModel.sol";

/// @title SpendguardReactive (Reactive Lasna)
/// @notice Neural circuit breaker for one GuardedWallet. For every `Spend` log the contract
/// builds 9 integer features from its rolling state, runs the fixed-point MLP in
/// `SpendguardModel`, emits `Scored`, and requests a `freeze` callback when the score
/// reaches `threshold`. The feature code mirrors `model/spendguard_ref.py` exactly.
contract SpendguardReactive is AbstractReactive {
    uint256 public constant SPEND_TOPIC = uint256(keccak256("Spend(uint256,address,uint256,uint256,uint256,uint256)"));
    uint256 public constant UNFROZEN_TOPIC = uint256(keccak256("Unfrozen(uint256)"));
    uint256 public constant N_FEATURES = 9;

    address public immutable owner;
    uint256 public immutable destinationChain;
    int256 public immutable threshold;
    uint64 public immutable callbackGas;
    /// If the wallet keeps spending this many seconds (event time) after a freeze request,
    /// the request is assumed lost and sent again.
    uint256 public immutable retryAfter;
    address public wallet;

    /// Rolling state; cleared on every new wallet epoch. Packs into one storage slot.
    struct Rolling {
        uint64 lastTs;
        uint32 count;
        uint32 burst;
        uint32 rush;
        uint32 flow;
        uint32 fresh;
        uint32 freshFlow;
    }

    Rolling public rolling;
    /// Learned profile; survives epochs. Only spends scored below threshold are learned.
    uint256 public ema;
    mapping(address => uint256) public firstSeen;
    mapping(bytes32 => bool) public seen;
    uint256 public epoch;
    /// Event timestamp of the last freeze request in this epoch (0 = none).
    uint256 public trippedAt;
    uint256 public scored;

    event Scored(uint256 indexed epoch, uint256 indexed nonce, int256 score, int256[9] features, bool flagged);
    event FreezeRequested(uint256 indexed epoch, uint256 indexed nonce, int256 score);
    event EpochAdvanced(uint256 indexed epoch);
    event WalletLinked(address indexed wallet);

    modifier onlyOwner() {
        require(msg.sender == owner, "owner");
        _;
    }

    constructor(address owner_, uint256 destination, int256 threshold_, uint64 gas_, uint256 retryAfter_) payable {
        require(owner_ != address(0) && destination != 0 && gas_ >= 100000 && retryAfter_ > 0, "config");
        owner = owner_;
        destinationChain = destination;
        threshold = threshold_;
        callbackGas = gas_;
        retryAfter = retryAfter_;
    }

    /// @notice One-time link to the wallet (deployed after this contract, because the wallet
    /// pins this address as its only callback sender). Subscribes to its Spend and Unfrozen logs.
    function setWallet(address w) external onlyOwner {
        require(wallet == address(0) && w != address(0), "wallet");
        wallet = w;
        emit WalletLinked(w);
        SYSTEM.subscribe(destinationChain, w, SPEND_TOPIC, REACTIVE_IGNORE, REACTIVE_IGNORE, REACTIVE_IGNORE);
        SYSTEM.subscribe(destinationChain, w, UNFROZEN_TOPIC, REACTIVE_IGNORE, REACTIVE_IGNORE, REACTIVE_IGNORE);
    }

    /// @notice Anyone can re-run the deployed network on a feature vector (eth_call).
    /// Returns unused lREACT to the owner. Withdrawing everything stops the guard from paying for itself.
    function withdraw(address payable to, uint256 amount) external onlyOwner {
        require(to != address(0), "to");
        _pay(to, amount);
    }

    function scoreFeatures(int256[9] calldata x) external pure returns (int256) {
        return M.score(x);
    }

    function modelHash() external pure returns (bytes32) {
        return M.WEIGHTS_SHA256;
    }

    function react(LogRecord calldata log) external onlySystem {
        if (wallet == address(0) || log.chainId != destinationChain || log.contractAddress != wallet) return;
        if (log.topic0 == SPEND_TOPIC) {
            bytes32 key = keccak256(abi.encode(log.chainId, log.txHash, log.logIndex));
            if (seen[key]) return;
            seen[key] = true;
            (uint256 amount, uint256 before, uint256 ts, uint256 e) =
                abi.decode(log.data, (uint256, uint256, uint256, uint256));
            _sync(e);
            if (e < epoch) return; // a spend from before an unfreeze, delivered late
            // forge-lint: disable-next-line(unsafe-typecast) -- topic2 is an indexed address
            _score(log.topic1, address(uint160(log.topic2)), amount, before, ts);
        } else if (log.topic0 == UNFROZEN_TOPIC) {
            _sync(log.topic1);
        }
    }

    // ------------------------------------------------------------------ internals

    function _sync(uint256 e) internal {
        if (e <= epoch) return;
        epoch = e;
        delete rolling;
        trippedAt = 0;
        emit EpochAdvanced(e);
    }

    function _score(uint256 nonce, address to, uint256 amount, uint256 before, uint256 ts) internal {
        Rolling memory r = rolling;
        int256[9] memory x;

        uint256 dt;
        if (r.count == 0) {
            dt = M.GAP_MAX;
        } else {
            dt = ts > r.lastTs ? ts - r.lastTs : 0;
            if (dt > M.GAP_MAX) dt = M.GAP_MAX;
        }
        uint256 ratio;
        uint256 part;
        if (before == 0 || amount >= before) {
            ratio = 1 << 16;
            part = M.FLOW_ONE;
        } else {
            ratio = amount * (1 << 16) / before;
            part = amount * M.FLOW_ONE / before;
        }
        x[0] = _log2fx(ratio + 1); // frac: log2 of amount / balance
        x[1] = _log2fx(dt + 1); // gap: log2 of seconds since previous spend
        // Accumulators are clamped to ACC_MAX (2^17), so the uint32 casts below cannot truncate.
        r.burst = _u32(_decay(r.burst, dt, M.H_BURST) + M.S); // decayed spend count
        r.rush = _u32(_decay(r.rush, dt, M.H_BURST) + part); // outflow, short window
        r.flow = _u32(_decay(r.flow, dt, M.H_FLOW) + part); // outflow, long window
        x[2] = _i(r.burst);
        x[3] = _i(r.rush);
        x[4] = _i(r.flow);

        uint256 seenAt = firstSeen[to];
        bool isFresh;
        if (seenAt == 0) {
            isFresh = true; // x[5] stays 0: never paid before
        } else {
            uint256 age = ts > seenAt ? ts - seenAt : 0;
            x[5] = _log2fx(age + 2); // recipAge
            isFresh = age < M.TRUST_AGE;
        }
        r.fresh = _u32(_decay(r.fresh, dt, M.H_FRESH) + (isFresh ? M.S : 0)); // fresh-recipient count
        r.freshFlow = _u32(_decay(r.freshFlow, dt, M.H_FRESH) + (isFresh ? part : 0)); // outflow to them
        x[6] = _i(r.fresh);
        x[7] = _i(r.freshFlow);

        uint256 typical = ema;
        if (typical != 0) {
            int256 size = _log2fx(amount + 1) - _log2fx(typical + 1); // log2 of amount / typical
            if (size > M.SIZE_CLAMP) size = M.SIZE_CLAMP;
            if (size < -M.SIZE_CLAMP) size = -M.SIZE_CLAMP;
            x[8] = size;
        }

        int256 s = M.score(x);
        bool flagged = s >= threshold;

        // forge-lint: disable-next-line(unsafe-typecast) -- block timestamps fit in 64 bits
        if (ts > r.lastTs) r.lastTs = uint64(ts);
        if (r.count < type(uint32).max) r.count++;
        rolling = r;
        if (!flagged) {
            if (seenAt == 0) firstSeen[to] = ts;
            if (typical == 0) {
                ema = amount;
            } else if (amount >= typical) {
                ema = typical + ((amount - typical) >> M.EMA_SHIFT);
            } else {
                ema = typical - ((typical - amount) >> M.EMA_SHIFT);
            }
        }
        scored++;
        emit Scored(epoch, nonce, s, x, flagged);

        // First flag of the epoch requests a freeze. A spend observed `retryAfter` seconds
        // after a request proves the wallet is still live, so the request is repeated.
        if (trippedAt == 0 ? flagged : ts >= trippedAt + retryAfter) {
            trippedAt = ts == 0 ? 1 : ts;
            emit FreezeRequested(epoch, nonce, s);
            SYSTEM.requestCallbackV_1_0(
                ISystemContract.CallbackConfiguration_V_1_0(
                    destinationChain,
                    wallet,
                    callbackGas,
                    abi.encodeWithSignature("freeze(address,uint256,uint256,int256)", address(0), epoch, nonce, s)
                )
            );
        }
    }

    /// Clamp to ACC_MAX and narrow for packed storage.
    function _u32(uint256 v) internal pure returns (uint32) {
        // forge-lint: disable-next-line(unsafe-typecast) -- ACC_MAX < 2^32
        return uint32(v > M.ACC_MAX ? M.ACC_MAX : v);
    }

    function _i(uint32 v) internal pure returns (int256) {
        return int256(uint256(v));
    }

    /// v * 2^(-dt/h): whole half-lives as shifts, the remainder as a chord (v - v*r/2h).
    function _decay(uint256 v, uint256 dt, uint256 h) internal pure returns (uint256) {
        uint256 k = dt / h;
        if (k >= 64) return 0;
        v >>= k;
        uint256 r = dt % h;
        return v - (v * r) / (2 * h);
    }

    /// floor(log2 n) * 16 + the next 4 mantissa bits. n >= 1.
    function _log2fx(uint256 n) internal pure returns (int256) {
        uint256 k = 0;
        uint256 v = n;
        if (v >= 1 << 128) { v >>= 128; k += 128; }
        if (v >= 1 << 64) { v >>= 64; k += 64; }
        if (v >= 1 << 32) { v >>= 32; k += 32; }
        if (v >= 1 << 16) { v >>= 16; k += 16; }
        if (v >= 1 << 8) { v >>= 8; k += 8; }
        if (v >= 1 << 4) { v >>= 4; k += 4; }
        if (v >= 1 << 2) { v >>= 2; k += 2; }
        if (v >= 1 << 1) { k += 1; }
        uint256 m = k >= 4 ? n >> (k - 4) : n << (4 - k);
        // forge-lint: disable-next-line(unsafe-typecast) -- at most 255 * 16 + 15
        return int256(k * 16 + m - 16);
    }
}

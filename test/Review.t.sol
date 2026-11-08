// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

import {Test, Vm, console} from "forge-std/Test.sol";
import {GuardedWallet} from "../src/GuardedWallet.sol";
import {SpendguardReactive} from "../src/SpendguardReactive.sol";
import {TestToken} from "../src/TestToken.sol";
import {IReactive} from "@reactive/src/interfaces/IReactive.sol";
import {IPayer} from "@reactive/src/interfaces/IPayer.sol";
import {MockSystem} from "./Spendguard.t.sol";

/// Models the documented callback-proxy billing (dev.reactive.network/economy, /callback-contracts):
/// the recipient is billed after every delivery regardless of outcome; when reserves are empty the
/// proxy calls pay(); a recipient with debt receives nothing. Not the real proxy: an illustration.
/// Callback proxy stand-in: reports no debt so spends are not blocked by the fee fail-safe.
contract MockProxy {
    receive() external payable {}

    function debt(address) external pure returns (uint256) {
        return 0;
    }
}

contract BillingProxy {
    uint256 public immutable fee;
    mapping(address => uint256) public debt;

    constructor(uint256 f) {
        fee = f;
    }

    receive() external payable {}

    function deliver(address target, address requester, bytes memory payload)
        external
        returns (bool delivered, bool ok)
    {
        if (debt[target] > 0) return (false, false);
        assembly {
            mstore(add(payload, 36), requester) // proxy injects the requesting reactive's address
        }
        (ok,) = target.call(payload);
        try IPayer(payable(target)).pay(fee) {} catch {
            debt[target] += fee;
        }
        delivered = true;
    }
}

contract ReviewTest is Test {
    address constant SYSTEM = 0x8888888888888888888888888888888888888888;
    address constant PROXY = address(0x100);
    address constant AGENT = address(0xA6E47);
    uint256 constant BASE = 84532;
    bytes32 constant SCORED = keccak256("Scored(uint256,uint256,int256,int256[9],bool)");

    MockSystem system;
    TestToken token;
    GuardedWallet wallet;
    SpendguardReactive rc;
    int256 thr;
    uint256 txid;

    function setUp() public {
        vm.warp(1_800_000_000);
        vm.etch(SYSTEM, address(new MockSystem()).code);
        system = MockSystem(SYSTEM);
        thr = vm.parseJsonInt(vm.readFile("test/vectors.json"), ".threshold");
        rc = new SpendguardReactive(address(this), BASE, thr, 200000, 60);
        token = new TestToken("T", "T", 1_000_000 ether, address(this));
        vm.etch(PROXY, address(new MockProxy()).code);
        wallet = new GuardedWallet(PROXY, address(rc), address(this), token, AGENT, 100 ether);
        token.transfer(address(wallet), 1000 ether);
        rc.setWallet(address(wallet));
    }

    function _log(address w, uint256 nonce, address to, uint256 amount, uint256 bal, uint256 ts, uint256 e)
        internal
        returns (IReactive.LogRecord memory l)
    {
        l.chainId = BASE;
        l.contractAddress = w;
        l.topic0 = rc.SPEND_TOPIC();
        l.topic1 = nonce;
        l.topic2 = uint256(uint160(to));
        l.data = abi.encode(amount, bal, ts, e);
        l.txHash = ++txid;
    }

    function _react(IReactive.LogRecord memory l) internal returns (bool found, int256 s, int256[9] memory x, bool f) {
        vm.recordLogs();
        vm.prank(SYSTEM);
        rc.react(l);
        Vm.Log[] memory logs = vm.getRecordedLogs();
        for (uint256 i; i < logs.length; i++) {
            if (logs[i].emitter == address(rc) && logs[i].topics[0] == SCORED) {
                (s, x, f) = abi.decode(logs[i].data, (int256, int256[9], bool));
                found = true;
            }
        }
    }

    function _deliver(address sender) internal {
        bytes memory p = system.payload();
        assembly {
            mstore(add(p, 36), sender)
        }
        vm.prank(PROXY);
        (bool ok,) = address(wallet).call(p);
        assertTrue(ok);
    }

    // H-1: the freeze is detective only. All cap-sized spends fit in one block, before the first
    // Spend log can even reach Lasna, so the whole balance leaves regardless of the score.
    function test_H1_WindowLimitBoundsSameBlockDrain() public {
        // Default window: 3 x cap per hour. Ten same-block spends of the cap stop after three.
        for (uint256 i; i < 3; i++) {
            vm.prank(AGENT);
            wallet.spend(address(0xBAD), 100 ether);
        }
        vm.prank(AGENT);
        vm.expectRevert("window");
        wallet.spend(address(0xBAD), 100 ether);
        assertEq(token.balanceOf(address(wallet)), 700 ether, "at most 3 x cap left before any freeze");
    }

    function test_H1_WindowCountAndReset() public {
        // Variant: many small spends hit the count limit; the window resets after its length.
        wallet.setWindow(1 hours, 1000 ether, 5);
        for (uint256 i; i < 5; i++) {
            vm.prank(AGENT);
            wallet.spend(address(0xBAD), 1);
        }
        vm.prank(AGENT);
        vm.expectRevert("window");
        wallet.spend(address(0xBAD), 1);
        vm.warp(block.timestamp + 1 hours);
        vm.prank(AGENT);
        wallet.spend(address(0xBAD), 1);
    }

    function test_H2_WalletInDebtStopsSpending() public {
        BillingProxy bp = new BillingProxy(0.001 ether);
        GuardedWallet w = new GuardedWallet{value: 0.01 ether}(address(bp), address(rc), address(this), token, AGENT, 100 ether);
        token.transfer(address(w), 1000 ether);
        address foreignRc = address(0xA77AC4); // a third party's reactive contract on Lasna
        bytes memory p = abi.encodeWithSignature("freeze(address,uint256,uint256,int256)", address(0), 0, 0, 0);
        for (uint256 i; i < 11; i++) {
            (bool delivered, bool ok) = bp.deliver(address(w), foreignRc, p);
            assertTrue(delivered);
            assertFalse(ok, "unauthorized sender reverts, but is still billed");
        }
        assertGt(bp.debt(address(w)), 0, "wallet in debt");
        // The genuine freeze could not be delivered now, so the wallet must not keep spending.
        vm.prank(AGENT);
        vm.expectRevert("guard unfunded");
        w.spend(address(0xBAD), 100 ether);
        // Owner recovery still works while in debt.
        w.withdraw(address(this), 1000 ether);
        assertEq(token.balanceOf(address(w)), 0);
    }

    function test_H2_FundedWalletSpendsNormally() public {
        // Variant: with no debt the fail-safe does not interfere.
        BillingProxy bp = new BillingProxy(0.001 ether);
        GuardedWallet w = new GuardedWallet{value: 0.01 ether}(address(bp), address(rc), address(this), token, AGENT, 100 ether);
        token.transfer(address(w), 1000 ether);
        vm.prank(AGENT);
        w.spend(address(0xBAD), 100 ether);
        assertEq(token.balanceOf(address(w)), 900 ether);
    }

    // L-1: README says a recipient the guard flagged does not become known. In the demo, X (0xdead)
    // gets an unflagged spend first (spend 6, score 286), so it is learned and trusted 600 s later.
    function test_L1_DemoAttackRecipientIsLearnedFromUnflaggedSpend() public {
        address x = address(0xdead);
        uint256 t0 = 1800000000;
        _react(_log(address(wallet), 1, address(0xa11a), 4 ether, 1000 ether, t0, 0));
        _react(_log(address(wallet), 2, address(0xb22b), 6 ether, 996 ether, t0 + 180, 0));
        _react(_log(address(wallet), 3, address(0xa11a), 3.5 ether, 990 ether, t0 + 380, 0));
        _react(_log(address(wallet), 4, address(0xc33c), 5 ether, 986.5 ether, t0 + 530, 0));
        _react(_log(address(wallet), 5, address(0xb22b), 7 ether, 981.5 ether, t0 + 750, 0));
        _react(_log(address(wallet), 6, address(0xa11a), 4.5 ether, 974.5 ether, t0 + 920, 0));
        (, int256 s6,, bool f6) = _react(_log(address(wallet), 7, x, 96 ether, 970 ether, t0 + 1070, 0));
        assertEq(s6, 286);
        assertFalse(f6);
        (, int256 s7,, bool f7) = _react(_log(address(wallet), 8, x, 98 ether, 874 ether, t0 + 1076, 0));
        assertEq(s7, 509);
        assertTrue(f7);
        assertEq(rc.firstSeen(x), t0 + 1070, "flagged recipient is already known");

        // Owner unfreezes; 10 minutes later X counts as a trusted, non-fresh recipient.
        IReactive.LogRecord memory u;
        u.chainId = BASE;
        u.contractAddress = address(wallet);
        u.topic0 = rc.UNFROZEN_TOPIC();
        u.topic1 = 1;
        u.txHash = ++txid;
        _react(u);
        (,, int256[9] memory f,) = _react(_log(address(wallet), 10, x, 90 ether, 700 ether, t0 + 1700, 1));
        assertGt(f[5], 0, "recipAge set");
        assertEq(f[6], 0, "not counted as fresh");
    }

    // L-2: replay key is (chainId, txHash, logIndex), not the wallet nonce. If a reorg re-includes
    // the same spend under another block/logIndex, or a log is re-delivered with another txHash,
    // the same nonce is scored twice.
    function test_L2_SameNonceScoredTwiceUnderDifferentTxHash() public {
        _react(_log(address(wallet), 1, address(0xa11a), 4 ether, 1000 ether, block.timestamp, 0));
        _react(_log(address(wallet), 1, address(0xa11a), 4 ether, 1000 ether, block.timestamp, 0));
        assertEq(rc.scored(), 2);
        (, uint32 count, uint32 burst,,,,) = rc.rolling();
        assertEq(count, 2);
        assertEq(burst, 512);
    }

    // I-1: extreme values. balanceBefore == 0 and timestamps going backwards are handled; amounts
    // near 2^244+ overflow the feature arithmetic and react() reverts (spend never scored).
    function test_I1_ExtremeInputs() public {
        _react(_log(address(wallet), 9, address(0xa11a), 4 ether, 1000 ether, block.timestamp - 2000, 0)); // unflagged: sets ema
        assertGt(rc.ema(), 0);
        (bool f0,,,) = _react(_log(address(wallet), 1, address(1), 1 ether, 0, block.timestamp, 0));
        assertTrue(f0, "before == 0 handled");
        (bool f1,, int256[9] memory x,) = _react(_log(address(wallet), 2, address(1), 1 ether, 5 ether, block.timestamp - 1000, 0));
        assertTrue(f1, "backwards timestamp handled");
        assertEq(x[1], 0, "dt clamps to 0 (log2fx(1) = 0)");

        IReactive.LogRecord memory big = _log(address(wallet), 3, address(1), 1 << 245, 1 << 250, block.timestamp, 0);
        vm.prank(SYSTEM);
        vm.expectRevert(); // panic 0x11: amount * 2^16
        rc.react(big);

        IReactive.LogRecord memory maxed =
            _log(address(wallet), 4, address(1), type(uint256).max, type(uint256).max, block.timestamp, 0);
        vm.prank(SYSTEM);
        vm.expectRevert(); // panic 0x11: amount + 1 in the size feature (ema != 0)
        rc.react(maxed);
    }

    /// Realistic ranges (balances < 2^200, timestamps < 2^63, any order) never revert.
    function testFuzz_ReactNoRevertRealisticRanges(uint256[3] memory amt, uint256[3] memory bal, uint64[3] memory ts)
        public
    {
        for (uint256 i; i < 3; i++) {
            uint256 b = bound(bal[i], 0, 1 << 200);
            uint256 a = bound(amt[i], 1, b == 0 ? 1 : b);
            uint256 t = bound(uint256(ts[i]), 0, 1 << 63);
            (bool found,,,) = _react(_log(address(wallet), i + 1, address(uint160(0x1000 + i)), a, b, t, 0));
            assertTrue(found);
        }
    }

    // L-3: each spend is one billed reactive transaction on Lasna. Slow dust spends to a known
    // recipient are scored but (here) not flagged, so a compromised agent can run down lREACT.
    function test_L3_SlowDustSpendsAreScoredWithoutFreeze() public {
        uint256 t = block.timestamp;
        for (uint256 i; i < 5; i++) {
            t += 300;
            _react(_log(address(wallet), i + 1, address(0xa11a), 4 ether, 1000 ether, t, 0));
        }
        uint256 c = system.count();
        for (uint256 i; i < 40; i++) {
            t += 300;
            _react(_log(address(wallet), 100 + i, address(0xa11a), 1, 980 ether, t, 0));
        }
        console.log("dust spends scored", rc.scored(), "freeze requests", system.count() - c);
        assertEq(system.count(), c, "no freeze for slow dust");
    }
}

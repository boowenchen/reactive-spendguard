// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

import {Test, Vm, console} from "forge-std/Test.sol";
import {StdInvariant} from "forge-std/StdInvariant.sol";
import {GuardedWallet} from "../src/GuardedWallet.sol";
import {SpendguardReactive} from "../src/SpendguardReactive.sol";
import {TestToken} from "../src/TestToken.sol";
import {IReactive} from "@reactive/src/interfaces/IReactive.sol";
import {ISystemContract} from "@reactive/src/interfaces/ISystemContract.sol";

/// Local stand-in for the Reactive system contract: records subscriptions and callback requests.
/// Callback proxy stand-in: reports no debt so spends are not blocked by the fee fail-safe.
contract MockProxy {
    receive() external payable {}

    function debt(address) external pure returns (uint256) {
        return 0;
    }
}

contract MockSystem {
    uint256 public count;
    uint256 public subscriptions;
    bytes public payload;
    address public recipient;
    uint256 public chain;
    uint64 public gasLimit;

    function subscribe(uint256, address, uint256, uint256, uint256, uint256) external {
        subscriptions++;
    }

    function unsubscribe(uint256, address, uint256, uint256, uint256, uint256) external {
        subscriptions--;
    }

    function requestCallbackV_1_0(ISystemContract.CallbackConfiguration_V_1_0 memory c) external {
        count++;
        payload = c.payload;
        recipient = c.recipient;
        chain = c.chainId;
        gasLimit = c.gasLimit;
    }
}

contract SpendguardTest is Test {
    address constant SYSTEM = 0x8888888888888888888888888888888888888888;
    address constant PROXY = address(0x100);
    address constant AGENT = address(0xA6E47);
    uint256 constant BASE = 84532;
    uint256 constant RETRY = 60;
    bytes32 constant SCORED = keccak256("Scored(uint256,uint256,int256,int256[9],bool)");

    struct ModelVec {
        int256[] x;
        int256 y;
    }

    struct VSpend {
        uint256 amount;
        uint256 balance;
        int256[] features;
        bool flagged;
        int256 score;
        address to;
        uint256 ts;
    }

    struct VSeq {
        int256 firstFlag;
        string name;
        VSpend[] spends;
    }

    MockSystem system;
    TestToken token;
    GuardedWallet wallet;
    SpendguardReactive rc;
    string json;
    int256 thr;
    uint256 txid;

    function setUp() public {
        vm.warp(1_800_000_000);
        MockSystem mock = new MockSystem();
        vm.etch(SYSTEM, address(mock).code);
        system = MockSystem(SYSTEM);
        json = vm.readFile("test/vectors.json");
        thr = vm.parseJsonInt(json, ".threshold");
        rc = new SpendguardReactive(address(this), BASE, thr, 200000, RETRY);
        token = new TestToken("Spendguard Demo USD", "sgUSD", 1_000_000 ether, address(this));
        vm.etch(PROXY, address(new MockProxy()).code);
        wallet = new GuardedWallet(PROXY, address(rc), address(this), token, AGENT, 100 ether);
        wallet.setWindow(1 hours, 1_000_000 ether, 1000); // window limits are tested separately
        token.transfer(address(wallet), 1000 ether);
        rc.setWallet(address(wallet));
    }

    // ------------------------------------------------------------------ helpers

    function _seq(string memory name) internal view returns (VSeq memory) {
        VSeq[] memory all = abi.decode(vm.parseJson(json, ".sequences"), (VSeq[]));
        for (uint256 i; i < all.length; i++) {
            if (keccak256(bytes(all[i].name)) == keccak256(bytes(name))) return all[i];
        }
        revert("no such sequence");
    }

    function _spendLog(address w, uint256 nonce, address to, uint256 amount, uint256 bal, uint256 ts, uint256 e)
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

    /// Delivers `l` to `r` as the system contract and returns the Scored event, if any.
    function _react(SpendguardReactive r, IReactive.LogRecord memory l)
        internal
        returns (bool found, int256 score, int256[9] memory x, bool flagged)
    {
        vm.recordLogs();
        vm.prank(SYSTEM);
        r.react(l);
        Vm.Log[] memory logs = vm.getRecordedLogs();
        for (uint256 i; i < logs.length; i++) {
            if (logs[i].emitter == address(r) && logs[i].topics[0] == SCORED) {
                (score, x, flagged) = abi.decode(logs[i].data, (int256, int256[9], bool));
                found = true;
            }
        }
    }

    /// Agent spends through the real wallet; the emitted Spend log is relayed to `rc`.
    function _walletSpend(address to, uint256 amount) internal returns (int256 score, bool flagged) {
        vm.recordLogs();
        vm.prank(AGENT);
        wallet.spend(to, amount);
        Vm.Log[] memory logs = vm.getRecordedLogs();
        IReactive.LogRecord memory l;
        for (uint256 i; i < logs.length; i++) {
            if (logs[i].emitter == address(wallet) && uint256(logs[i].topics[0]) == rc.SPEND_TOPIC()) {
                l.chainId = BASE;
                l.contractAddress = address(wallet);
                l.topic0 = uint256(logs[i].topics[0]);
                l.topic1 = uint256(logs[i].topics[1]);
                l.topic2 = uint256(logs[i].topics[2]);
                l.data = logs[i].data;
                l.txHash = ++txid;
            }
        }
        bool found;
        (found, score,, flagged) = _react(rc, l);
        assertTrue(found, "scored");
    }

    function _unfrozenLog(uint256 e) internal returns (IReactive.LogRecord memory l) {
        l.chainId = BASE;
        l.contractAddress = address(wallet);
        l.topic0 = rc.UNFROZEN_TOPIC();
        l.topic1 = e;
        l.txHash = ++txid;
    }

    /// Plays the callback proxy: injects the reactive address into the recorded payload.
    function _deliver(address sender) internal {
        bytes memory p = system.payload();
        assembly {
            mstore(add(p, 36), sender)
        }
        vm.prank(PROXY);
        (bool ok,) = address(wallet).call(p);
        assertTrue(ok, "callback");
    }

    function _burst(uint256 n, uint256 gap) internal {
        for (uint256 i; i < n; i++) {
            vm.warp(block.timestamp + gap);
            _walletSpend(address(0xBAD), 95 ether);
        }
    }

    /// Near-cap spends to a new address every `gap` seconds until a freeze is requested.
    function _burstUntilRequest(uint256 maxN, uint256 gap) internal returns (uint256 n) {
        uint256 c = system.count();
        while (system.count() == c) {
            require(n < maxN, "no freeze requested");
            vm.warp(block.timestamp + gap);
            _walletSpend(address(0xBAD), 95 ether);
            n++;
        }
    }

    // ------------------------------------------------------------------ parity

    function test_ModelParityWithPythonVectors() public view {
        ModelVec[] memory v = abi.decode(vm.parseJson(json, ".model"), (ModelVec[]));
        assertGt(v.length, 200);
        for (uint256 i; i < v.length; i++) {
            int256[9] memory x;
            for (uint256 j; j < 9; j++) x[j] = v[i].x[j];
            assertEq(rc.scoreFeatures(x), v[i].y, "score mismatch");
        }
    }

    function test_SequenceParityFeaturesScoresAndFirstFreeze() public {
        VSeq[] memory all = abi.decode(vm.parseJson(json, ".sequences"), (VSeq[]));
        assertGe(all.length, 7);
        for (uint256 q; q < all.length; q++) {
            SpendguardReactive r = new SpendguardReactive(address(this), BASE, thr, 200000, RETRY);
            address w = address(uint160(0xC0FFEE + q));
            r.setWallet(w);
            int256 firstReq = -1;
            for (uint256 i; i < all[q].spends.length; i++) {
                VSpend memory s = all[q].spends[i];
                uint256 before = system.count();
                (bool found, int256 score, int256[9] memory x, bool flagged) =
                    _react(r, _spendLog(w, i + 1, s.to, s.amount, s.balance, s.ts, 0));
                assertTrue(found);
                for (uint256 j; j < 9; j++) assertEq(x[j], s.features[j], "feature mismatch");
                assertEq(score, s.score, "score mismatch");
                assertEq(flagged, s.flagged, "flag mismatch");
                if (firstReq < 0 && system.count() > before) firstReq = int256(i);
            }
            assertEq(firstReq, all[q].firstFlag, all[q].name);
        }
    }

    // ------------------------------------------------------------------ end to end

    function test_DemoFreezesAtExpectedSpendThenBlocksSpending() public {
        VSeq memory d = _seq("demo");
        assertGt(d.firstFlag, 0);
        uint256 freezeAt = uint256(d.firstFlag);
        for (uint256 i; i < d.spends.length; i++) {
            VSpend memory s = d.spends[i];
            vm.warp(s.ts);
            if (wallet.frozen()) {
                vm.prank(AGENT);
                vm.expectRevert(bytes("frozen"));
                wallet.spend(s.to, s.amount);
                continue;
            }
            assertEq(token.balanceOf(address(wallet)), s.balance, "balance before");
            uint256 before = system.count();
            (int256 score, bool flagged) = _walletSpend(s.to, s.amount);
            assertEq(score, s.score);
            assertEq(flagged, s.flagged);
            if (i < freezeAt) assertEq(system.count(), before, "no request before expected spend");
            if (i == freezeAt) {
                assertEq(system.count(), before + 1, "request at expected spend");
                assertEq(system.recipient(), address(wallet));
                assertEq(system.chain(), BASE);
                _deliver(address(rc));
                assertTrue(wallet.frozen());
            }
        }
        uint256 spent;
        for (uint256 i; i <= freezeAt; i++) spent += d.spends[i].amount;
        assertEq(token.balanceOf(address(wallet)), 1000 ether - spent);
    }

    function test_NormalSequencesNeverRequestFreeze() public {
        string[2] memory names = ["demo-normal-only", "test-normal"];
        for (uint256 n; n < 2; n++) {
            VSeq memory q = _seq(names[n]);
            assertEq(q.firstFlag, -1);
            SpendguardReactive r = new SpendguardReactive(address(this), BASE, thr, 200000, RETRY);
            address w = address(uint160(0xF00D + n));
            r.setWallet(w);
            uint256 before = system.count();
            for (uint256 i; i < q.spends.length; i++) {
                VSpend memory s = q.spends[i];
                _react(r, _spendLog(w, i + 1, s.to, s.amount, s.balance, s.ts, 0));
            }
            assertEq(system.count(), before, names[n]);
            assertEq(r.trippedAt(), 0);
        }
    }

    function test_SpendRevertsWhileFrozenOwnerCanStillWithdraw() public {
        _burstUntilRequest(4, 5);
        _deliver(address(rc));
        assertTrue(wallet.frozen());
        vm.prank(AGENT);
        vm.expectRevert(bytes("frozen"));
        wallet.spend(address(0xBAD), 1 ether);
        uint256 bal = token.balanceOf(address(wallet));
        wallet.withdraw(address(this), bal);
        assertEq(token.balanceOf(address(wallet)), 0);
    }

    function test_UnfreezeAdvancesEpochAndStaleCallbackIsIgnored() public {
        _burstUntilRequest(4, 5);
        uint256 requested = system.count();
        // Owner unfreezes before the queued freeze lands: that callback is now stale.
        wallet.unfreeze();
        assertEq(wallet.epoch(), 1);
        _deliver(address(rc));
        assertFalse(wallet.frozen(), "stale freeze applied");
        // Reactive side resets on the Unfrozen log.
        _react(rc, _unfrozenLog(1));
        assertEq(rc.epoch(), 1);
        assertEq(rc.trippedAt(), 0);
        (uint64 lastTs, uint32 count,,,,,) = rc.rolling();
        assertEq(count, 0);
        assertEq(lastTs, 0);
        // Spends now carry epoch 1; a freeze requested now is valid.
        _burstUntilRequest(4, 5);
        assertEq(system.count(), requested + 1);
        _deliver(address(rc));
        assertTrue(wallet.frozen());
    }

    function test_SpendLogWithNewerEpochResetsStateEvenWithoutUnfrozenLog() public {
        _burst(2, 5);
        wallet.unfreeze();
        vm.warp(block.timestamp + 5);
        _walletSpend(address(0xA11A), 1 ether); // carries epoch 1
        assertEq(rc.epoch(), 1);
        (, uint32 count,,,,,) = rc.rolling();
        assertEq(count, 1);
    }

    function test_LateSpendFromOldEpochIsIgnored() public {
        _react(rc, _unfrozenLog(2));
        uint256 n = rc.scored();
        (bool found,,,) = _react(rc, _spendLog(address(wallet), 9, address(0xBAD), 1 ether, 10 ether, block.timestamp, 1));
        assertFalse(found);
        assertEq(rc.scored(), n);
    }

    // ------------------------------------------------------------------ authorization

    function test_OnlyProxyCarryingThisReactiveCanFreeze() public {
        vm.expectRevert();
        wallet.freeze(address(rc), 0, 1, 999);
        vm.prank(PROXY);
        vm.expectRevert();
        wallet.freeze(address(0xEE), 0, 1, 999);
        assertFalse(wallet.frozen());
        vm.prank(PROXY);
        wallet.freeze(address(rc), 0, 1, 999);
        assertTrue(wallet.frozen());
    }

    function testFuzz_UnauthorizedCallerCannotFreeze(address caller, address sender) public {
        vm.assume(caller != PROXY);
        vm.prank(caller);
        vm.expectRevert();
        wallet.freeze(sender, 0, 1, 1);
        assertFalse(wallet.frozen());
    }

    function test_ReactOnlySystemAndWrongOriginIgnored() public {
        IReactive.LogRecord memory l = _spendLog(address(wallet), 1, address(1), 1 ether, 10 ether, block.timestamp, 0);
        vm.expectRevert();
        rc.react(l);
        l.contractAddress = address(0x999);
        (bool found,,,) = _react(rc, l);
        assertFalse(found);
        l.contractAddress = address(wallet);
        l.chainId = 1;
        (found,,,) = _react(rc, l);
        assertFalse(found);
        assertEq(rc.scored(), 0);
    }

    function test_ReplayedLogIsNoop() public {
        IReactive.LogRecord memory l = _spendLog(address(wallet), 1, address(1), 1 ether, 10 ether, block.timestamp, 0);
        (bool found,,,) = _react(rc, l);
        assertTrue(found);
        (uint64 a, uint32 b, uint32 c, uint32 d, uint32 e, uint32 f, uint32 g) = rc.rolling();
        uint256 ema = rc.ema();
        (found,,,) = _react(rc, l);
        assertFalse(found);
        (uint64 a2, uint32 b2, uint32 c2, uint32 d2, uint32 e2, uint32 f2, uint32 g2) = rc.rolling();
        assertEq(abi.encode(a, b, c, d, e, f, g), abi.encode(a2, b2, c2, d2, e2, f2, g2));
        assertEq(rc.ema(), ema);
        assertEq(rc.scored(), 1);
    }

    function test_CapAndAgentEnforced() public {
        vm.prank(AGENT);
        vm.expectRevert(bytes("cap"));
        wallet.spend(address(1), 100 ether + 1);
        vm.expectRevert(bytes("agent"));
        wallet.spend(address(1), 1 ether);
        vm.prank(AGENT);
        vm.expectRevert(bytes("cap"));
        wallet.spend(address(1), 0);
        vm.prank(AGENT);
        vm.expectRevert(bytes("owner"));
        wallet.setCap(1000 ether);
        wallet.setCap(1 ether);
        vm.prank(AGENT);
        vm.expectRevert(bytes("cap"));
        wallet.spend(address(1), 2 ether);
        wallet.setAgent(address(0xB0B));
        vm.prank(AGENT);
        vm.expectRevert(bytes("agent"));
        wallet.spend(address(1), 1 ether);
    }

    function test_RetryWhenWalletKeepsSpendingAfterRequest() public {
        _burstUntilRequest(4, 5);
        uint256 c = system.count();
        assertEq(c, 1, "one request");
        _burst(1, 5); // inside retry window: no duplicate
        assertEq(system.count(), c);
        vm.warp(block.timestamp + RETRY);
        _walletSpend(address(0xA11A), 1 ether); // freeze evidently did not land
        assertEq(system.count(), c + 1, "retried");
    }

    function test_CallbackFeesCannotTouchTokenPrincipal() public {
        vm.deal(address(wallet), 1 ether);
        vm.prank(PROXY);
        wallet.pay(0.1 ether);
        assertEq(token.balanceOf(address(wallet)), 1000 ether);
    }

    function test_SubscriptionsOnLink() public view {
        assertEq(rc.wallet(), address(wallet));
        assertEq(system.subscriptions(), 2);
    }

    // ------------------------------------------------------------------ gas

    /// Gas of react() itself for every demo spend. The system contract is replaced by a no-op
    /// so the recording mock's own storage writes are not counted; the real system contract's
    /// cost for requestCallbackV_1_0 is not measurable locally.
    function test_ReactGas() public {
        vm.etch(SYSTEM, address(new NoopSystem()).code);
        uint256 maxGas;
        VSeq memory d = _seq("demo");
        for (uint256 i; i < d.spends.length; i++) {
            VSpend memory s = d.spends[i];
            IReactive.LogRecord memory l = _spendLog(address(wallet), i + 1, s.to, s.amount, s.balance, s.ts, 0);
            vm.prank(SYSTEM);
            uint256 g = gasleft();
            rc.react(l);
            g -= gasleft();
            console.log("react() gas, demo spend", i, s.flagged ? "flagged" : "-", g);
            if (g > maxGas) maxGas = g;
        }
        console.log("react() max gas", maxGas);
        assertLt(maxGas, 300000);
    }
}

contract NoopSystem {
    function subscribe(uint256, address, uint256, uint256, uint256, uint256) external {}
    function requestCallbackV_1_0(ISystemContract.CallbackConfiguration_V_1_0 memory) external {}
}

// ---------------------------------------------------------------------- invariant

contract WalletHandler is Test {
    address constant PROXY = address(0x100);
    address constant RC = address(0x500);
    GuardedWallet public wallet;
    TestToken public token;
    uint256 public spent;
    uint256 public spendsWhileFrozen;
    uint256 public callbackMoves;

    constructor() {
        token = new TestToken("T", "T", 1000 ether, address(this));
        // This handler is both owner and agent.
        vm.etch(PROXY, address(new MockProxy()).code);
        wallet = new GuardedWallet(PROXY, RC, address(this), token, address(this), 20 ether);
        wallet.setWindow(1 hours, 1_000_000 ether, 1000);
        token.transfer(address(wallet), 1000 ether);
    }

    function spend(uint256 amount, uint8 who) external {
        amount = bound(amount, 1, 25 ether);
        bool wasFrozen = wallet.frozen();
        uint256 b = token.balanceOf(address(wallet));
        try wallet.spend(address(uint160(0x1000 + who)), amount) {
            spent += amount;
        } catch {}
        if (wasFrozen && token.balanceOf(address(wallet)) < b) spendsWhileFrozen++;
    }

    function freezeCallback(uint8 mode, int256 score) external {
        uint256 b = token.balanceOf(address(wallet));
        uint256 e = wallet.epoch();
        uint256 n = wallet.nonce(); // read before the prank: a prank applies to the next call only
        if (mode % 3 == 0 && e > 0) e--; // stale epoch
        vm.prank(PROXY);
        wallet.freeze(RC, e, n, score);
        if (token.balanceOf(address(wallet)) != b) callbackMoves++;
    }

    function unfreeze() external {
        wallet.unfreeze();
    }
}

contract SpendguardInvariantTest is StdInvariant, Test {
    WalletHandler handler;

    function setUp() public {
        handler = new WalletHandler();
        bytes4[] memory sel = new bytes4[](3);
        sel[0] = WalletHandler.spend.selector;
        sel[1] = WalletHandler.freezeCallback.selector;
        sel[2] = WalletHandler.unfreeze.selector;
        targetSelector(FuzzSelector({addr: address(handler), selectors: sel}));
        targetContract(address(handler));
    }

    function invariant_FrozenWalletCannotSpend() public view {
        assertEq(handler.spendsWhileFrozen(), 0);
    }

    function invariant_CallbacksNeverMoveFunds() public view {
        assertEq(handler.callbackMoves(), 0);
    }

    function invariant_BalanceConservation() public view {
        assertEq(handler.token().balanceOf(address(handler.wallet())) + handler.spent(), 1000 ether);
    }
}

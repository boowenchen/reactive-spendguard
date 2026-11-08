// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

import {Test} from "forge-std/Test.sol";
import {Deploy} from "../script/Deploy.s.sol";
import {TestToken} from "../src/TestToken.sol";
import {GuardedWallet} from "../src/GuardedWallet.sol";
import {SpendguardReactive} from "../src/SpendguardReactive.sol";
import {SpendguardModel} from "../src/SpendguardModel.sol";
import {MockSystem} from "./Spendguard.t.sol";

/// Runs every Deploy.s.sol stage offline (chain ids faked with vm.chainId). Nothing is sent
/// to a network; this checks wiring, env handling and the testnet-only guards.
contract DeployScriptTest is Test {
    uint256 constant KEY = 0xA11CE;
    address owner;

    function setUp() public {
        owner = vm.addr(KEY);
        vm.deal(owner, 10 ether);
        vm.etch(0x8888888888888888888888888888888888888888, address(new MockSystem()).code);
        vm.setEnv("PRIVATE_KEY", vm.toString(KEY));
        vm.setEnv("OWNER", vm.toString(owner));
        vm.setEnv("AGENT", vm.toString(address(0xA6E47)));
        vm.setEnv("TOKEN_SUPPLY_WEI", "1000000000000000000000000");
        vm.setEnv("FUND_WEI", "1000000000000000000000");
        vm.setEnv("CAP_WEI", "100000000000000000000");
        vm.setEnv("REACTIVE_FUND_WEI", "1000000000000000000");
        vm.setEnv("CALLBACK_FUND_WEI", "10000000000000000");
    }

    function _run(string memory step, uint256 chain) internal {
        vm.chainId(chain);
        vm.setEnv("STEP", step);
        vm.setEnv("EXPECTED_CHAIN_ID", vm.toString(chain));
        new Deploy().run();
    }

    function test_AllStagesWireTogetherAndGuardsHold() public {
        uint64 n = vm.getNonce(owner);
        _run("token", 84532);
        TestToken token = TestToken(vm.computeCreateAddress(owner, n));
        assertEq(token.balanceOf(owner), 1_000_000 ether);
        vm.setEnv("TOKEN", vm.toString(address(token)));

        n = vm.getNonce(owner);
        _run("reactive", 5318007);
        SpendguardReactive rc = SpendguardReactive(payable(vm.computeCreateAddress(owner, n)));
        assertEq(rc.owner(), owner);
        assertEq(rc.destinationChain(), 84532);
        assertEq(rc.threshold(), SpendguardModel.THRESHOLD);
        assertEq(address(rc).balance, 1 ether);
        vm.setEnv("REACTIVE", vm.toString(address(rc)));

        n = vm.getNonce(owner);
        _run("wallet", 84532);
        GuardedWallet w = GuardedWallet(payable(vm.computeCreateAddress(owner, n)));
        assertEq(w._CALLBACK_SENDER(), address(rc));
        assertEq(address(w._SERVICE_PROVIDER()), 0x476F44b11F5C942c58f4c8E3d0300E5710743585);
        assertEq(token.balanceOf(address(w)), 1000 ether);
        assertEq(w.cap(), 100 ether);
        assertEq(w.agent(), address(0xA6E47));
        vm.setEnv("WALLET", vm.toString(address(w)));

        _run("wallet", 84532); // rerun with WALLET set: no new wallet, no double funding
        assertEq(token.balanceOf(address(w)), 1000 ether);

        _run("link", 5318007);
        assertEq(rc.wallet(), address(w));
        _run("link", 5318007); // idempotent

        _run("unfreeze", 84532);
        assertEq(w.epoch(), 1);

        _rejectsNonTestnetAndWrongChain();
    }

    /// Part of the single test above: vm.setEnv is process-wide, so env-driven cases must not
    /// run as separate (parallel) tests.
    function _rejectsNonTestnetAndWrongChain() internal {
        vm.chainId(11155111); // a testnet, but not one Spendguard deploys to
        vm.setEnv("STEP", "token");
        vm.setEnv("EXPECTED_CHAIN_ID", "11155111");
        Deploy d = new Deploy();
        vm.expectRevert(bytes("testnet only"));
        d.run();
        vm.setEnv("EXPECTED_CHAIN_ID", "84532");
        vm.expectRevert(bytes("wrong chain"));
        d.run();
        vm.chainId(5318007);
        vm.setEnv("EXPECTED_CHAIN_ID", "5318007");
        vm.expectRevert(bytes("Base"));
        d.run();
    }
}

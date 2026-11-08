// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

import {Script} from "forge-std/Script.sol";
import {TestToken} from "../src/TestToken.sol";
import {GuardedWallet} from "../src/GuardedWallet.sol";
import {SpendguardReactive} from "../src/SpendguardReactive.sol";
import {SpendguardModel} from "../src/SpendguardModel.sol";

/// Staged deployment. Select STEP; setting an existing address env var makes a stage restartable.
///   STEP=token     (Base Sepolia)  demo ERC20 minted to OWNER
///   STEP=reactive  (Lasna)         SpendguardReactive, funded with REACTIVE_FUND_WEI
///   STEP=wallet    (Base Sepolia)  GuardedWallet bound to REACTIVE, funded with FUND_WEI tokens
///   STEP=link      (Lasna)         reactive.setWallet(WALLET): subscribes to Spend + Unfrozen
///   STEP=unfreeze  (Base Sepolia)  owner lifts a freeze and starts a new epoch
contract Deploy is Script {
    function run() external {
        string memory cfg = vm.readFile("config/networks.json");
        string memory step = vm.envString("STEP");
        uint256 base = vm.parseJsonUint(cfg, ".base.chainId");
        uint256 lasna = vm.parseJsonUint(cfg, ".lasna.chainId");
        uint256 expected = vm.envUint("EXPECTED_CHAIN_ID");
        require(block.chainid == expected, "wrong chain");
        require(expected == base || expected == lasna, "testnet only");
        address owner = vm.envAddress("OWNER");
        uint256 key = vm.envUint("PRIVATE_KEY");
        require(vm.addr(key) == owner, "owner/key mismatch");

        vm.startBroadcast(key);
        if (eq(step, "token")) {
            require(expected == base, "Base");
            address old = vm.envOr("TOKEN", address(0));
            if (old == address(0)) new TestToken("Spendguard Demo USD", "sgUSD", vm.envUint("TOKEN_SUPPLY_WEI"), owner);
            else require(old.code.length > 0, "missing token");
        } else if (eq(step, "reactive")) {
            require(expected == lasna, "Lasna");
            address old = vm.envOr("REACTIVE", address(0));
            if (old == address(0)) {
                new SpendguardReactive{value: vm.envUint("REACTIVE_FUND_WEI")}(
                    owner,
                    base,
                    vm.envOr("THRESHOLD", SpendguardModel.THRESHOLD),
                    uint64(vm.envOr("CALLBACK_GAS", uint256(500000))),
                    vm.envOr("RETRY_SECONDS", uint256(60))
                );
            } else {
                require(SpendguardReactive(payable(old)).owner() == owner, "existing owner");
            }
        } else if (eq(step, "wallet")) {
            require(expected == base, "Base");
            TestToken t = TestToken(vm.envAddress("TOKEN"));
            address w = vm.envOr("WALLET", address(0));
            if (w == address(0)) {
                w = address(
                    new GuardedWallet{value: vm.envUint("CALLBACK_FUND_WEI")}(
                        vm.parseJsonAddress(cfg, ".base.callbackProxy"),
                        vm.envAddress("REACTIVE"),
                        owner,
                        t,
                        vm.envAddress("AGENT"),
                        vm.envUint("CAP_WEI")
                    )
                );
            } else {
                GuardedWallet g = GuardedWallet(payable(w));
                require(g.owner() == owner && address(g.token()) == address(t), "existing wallet");
            }
            uint256 desired = vm.envUint("FUND_WEI");
            uint256 present = t.balanceOf(w);
            if (present < desired) require(t.transfer(w, desired - present), "fund");
        } else if (eq(step, "link")) {
            require(expected == lasna, "Lasna");
            SpendguardReactive rc = SpendguardReactive(payable(vm.envAddress("REACTIVE")));
            address w = vm.envAddress("WALLET");
            if (rc.wallet() == address(0)) rc.setWallet(w);
            else require(rc.wallet() == w, "wrong wallet");
        } else if (eq(step, "unfreeze")) {
            require(expected == base, "Base");
            GuardedWallet(payable(vm.envAddress("WALLET"))).unfreeze();
        } else {
            revert("unknown STEP");
        }
        vm.stopBroadcast();
    }

    function eq(string memory a, string memory b) internal pure returns (bool) {
        return keccak256(bytes(a)) == keccak256(bytes(b));
    }
}

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

import {Script} from "forge-std/Script.sol";
import {GuardedWallet} from "../src/GuardedWallet.sol";

/// One agent spend on Base Sepolia: WALLET.spend(TO, AMOUNT_WEI), signed with AGENT_PRIVATE_KEY.
contract Spend is Script {
    function run() external {
        string memory cfg = vm.readFile("config/networks.json");
        require(block.chainid == vm.parseJsonUint(cfg, ".base.chainId"), "Base Sepolia only");
        GuardedWallet w = GuardedWallet(payable(vm.envAddress("WALLET")));
        uint256 key = vm.envUint("AGENT_PRIVATE_KEY");
        require(vm.addr(key) == w.agent(), "not the wallet agent");
        require(!w.frozen(), "wallet is frozen");
        vm.startBroadcast(key);
        w.spend(vm.envAddress("TO"), vm.envUint("AMOUNT_WEI"));
        vm.stopBroadcast();
    }
}

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

import {AbstractCallback} from "@reactive/src/base/AbstractCallback.sol";
import {IPayable} from "@reactive/src/interfaces/IPayable.sol";
import {TestToken} from "./TestToken.sol";

/// @title GuardedWallet (Base Sepolia)
/// @notice A token wallet for a hot key (`agent`, e.g. an AI agent) with a cold `owner`.
/// Every agent spend emits `Spend`. Spendguard's reactive contract on Reactive Lasna scores
/// each spend with an on-chain neural network and, above its threshold, sends a callback
/// that sets `frozen`. A callback can only freeze: no callback path moves tokens.
/// Native ETH held here pays callback fees (AbstractPayer); token principal is separate.
contract GuardedWallet is AbstractCallback {
    address public immutable owner;
    TestToken public immutable token;
    address public agent;
    /// Hard per-call limit; bounds what a single spend can move before any freeze lands.
    uint256 public cap;
    /// Advanced by `unfreeze`; freeze callbacks carrying an older epoch are ignored.
    uint256 public epoch;
    /// Number of spends so far; the n-th spend carries nonce n.
    uint256 public nonce;
    bool public frozen;
    /// Hard rolling limits per fixed window. The network only freezes after a spend is seen, so
    /// these bound what a compromised agent can move before the freeze lands (e.g. many spends in
    /// one block). Defaults: 1 hour, 3 x cap, 20 spends.
    uint256 public windowSeconds;
    uint256 public windowAmount;
    uint256 public windowMaxSpends;
    uint256 public windowStart;
    uint256 public windowSpent;
    uint256 public windowCount;

    event Spend(
        uint256 indexed nonce, address indexed to, uint256 amount, uint256 balanceBefore, uint256 timestamp, uint256 epoch
    );
    event Frozen(uint256 indexed nonce, int256 score);
    event Unfrozen(uint256 indexed epoch);
    event AgentSet(address indexed agent);
    event CapSet(uint256 cap);
    event Withdrawn(address indexed to, uint256 amount);
    event WindowSet(uint256 windowSeconds, uint256 amount, uint256 maxSpends);

    modifier onlyOwner() {
        require(msg.sender == owner, "owner");
        _;
    }

    constructor(address proxy, address reactive, address owner_, TestToken token_, address agent_, uint256 cap_)
        payable
        AbstractCallback(IPayable(payable(proxy)), reactive)
    {
        require(proxy != address(0) && reactive != address(0) && owner_ != address(0), "address");
        require(address(token_) != address(0) && agent_ != address(0) && cap_ > 0, "config");
        owner = owner_;
        token = token_;
        agent = agent_;
        cap = cap_;
        emit AgentSet(agent_);
        emit CapSet(cap_);
        _setWindow(1 hours, 3 * cap_, 20);
    }

    /// @notice Agent pays `amount` tokens to `to`. Reverts while frozen, above the cap or the window
    /// limits, or while this wallet owes callback fees (an unpaid wallet could not receive a freeze,
    /// so spending stops instead of running unguarded).
    function spend(address to, uint256 amount) external {
        require(msg.sender == agent, "agent");
        require(!frozen, "frozen");
        require(amount > 0 && amount <= cap, "cap");
        require(_SERVICE_PROVIDER.debt(address(this)) == 0, "guard unfunded");
        if (block.timestamp >= windowStart + windowSeconds) {
            windowStart = block.timestamp;
            windowSpent = 0;
            windowCount = 0;
        }
        require(windowSpent + amount <= windowAmount && windowCount < windowMaxSpends, "window");
        windowSpent += amount;
        windowCount++;
        require(to != address(0) && to != address(this), "to");
        uint256 before = token.balanceOf(address(this));
        require(amount <= before, "balance");
        uint256 n = ++nonce;
        emit Spend(n, to, amount, before, block.timestamp, epoch);
        require(token.transfer(to, amount), "transfer");
    }

    /// @notice Reactive callback. The proxy injects the sender reactive contract as `rc`.
    /// Idempotent; ignores callbacks for an older epoch (requested before an unfreeze).
    function freeze(address rc, uint256 e, uint256 n, int256 score)
        external
        onlyServiceProvider
        onlyCallbackSender(rc)
    {
        if (e != epoch || frozen) return;
        frozen = true;
        emit Frozen(n, score);
    }

    /// @notice Owner lifts a freeze (or just resets the guard). Starts a new epoch, which
    /// makes queued freeze callbacks stale and tells the reactive contract to reset its
    /// rolling state.
    function unfreeze() external onlyOwner {
        frozen = false;
        epoch++;
        emit Unfrozen(epoch);
    }

    function setAgent(address a) external onlyOwner {
        require(a != address(0), "agent");
        agent = a;
        emit AgentSet(a);
    }

    function setCap(uint256 c) external onlyOwner {
        require(c > 0, "cap");
        cap = c;
        emit CapSet(c);
    }

    function setWindow(uint256 seconds_, uint256 amount, uint256 maxSpends) external onlyOwner {
        _setWindow(seconds_, amount, maxSpends);
    }

    function _setWindow(uint256 seconds_, uint256 amount, uint256 maxSpends) internal {
        require(seconds_ > 0 && amount > 0 && maxSpends > 0, "window");
        windowSeconds = seconds_;
        windowAmount = amount;
        windowMaxSpends = maxSpends;
        emit WindowSet(seconds_, amount, maxSpends);
    }

    /// @notice Owner recovery path; works while frozen.
    function withdraw(address to, uint256 amount) external onlyOwner {
        require(to != address(0), "to");
        emit Withdrawn(to, amount);
        require(token.transfer(to, amount), "transfer");
    }

    /// @notice Owner recovers unused callback-fee ETH.
    function withdrawNative(address payable to, uint256 amount) external onlyOwner {
        require(to != address(0), "to");
        (bool ok,) = to.call{value: amount}("");
        require(ok, "native");
    }
}
